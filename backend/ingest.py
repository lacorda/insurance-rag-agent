# coding: utf-8
"""文档加载、递归分块、Chroma 增量索引。

向量库为 Chroma。正文用 LangChain RecursiveCharacterTextSplitter
按段落 / 中英句读 / 空白切开。
"""

import hashlib
import json
import os

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
CHROMA_DIR = os.path.join(PROJECT_ROOT, "chroma")
MANIFEST_PATH = os.path.join(CHROMA_DIR, "manifest.json")
COLLECTION_NAME = "insurance_kb"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
CHUNK_SPEC = f"rcts-{CHUNK_SIZE}-{CHUNK_OVERLAP}"


# 文本切分器，返回格式为：[chunk1, chunk2, chunk3, ...]
# 每个chunk包含：
# - text: 切分后的文本内容
# - id_: 切分后的文本的唯一标识
# - metadata: 切分后的文本的元数据
# - node_id: 切分后的文本的节点标识
# - file_name: 切分后的文本的文件名
# - file_path: 切分后的文本的文件路径
# - file_type: 切分后的文本的文件类型
# - chunk_id: 切分后的文本的块标识
# - content_hash: 切分后的文本的内容哈希
_TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", "。", "！", "？", "；", "!", "?", " ", ""],
)


def split_chunks(text: str) -> list:
    """用 RecursiveCharacterTextSplitter 切块。

    参数:
        text: 原始文档正文
    返回:
        分块后的字符串列表，part.strip() 去除空格和换行符
    """

    return [part.strip() for part in _TEXT_SPLITTER.split_text(text) if part.strip()]


def _file_sha256(path: str) -> str:
    """计算文件字节 hash，用于增量判断是否变化。

    参数:
        path: 文件绝对路径
    返回:
        hex 摘要
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for piece in iter(lambda: handle.read(8192), b""):
            digest.update(piece)
    return digest.hexdigest()


def _load_manifest() -> dict:
    """读取 chroma 目录下的增量清单。"""
    if not os.path.exists(MANIFEST_PATH):
        return {"files": {}}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_manifest(manifest: dict) -> None:
    """把增量清单写回磁盘。

    参数:
        manifest: 含 files 字段的字典
    """
    os.makedirs(CHROMA_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def _open_collection():
    """打开（或创建）持久化 Chroma collection。

    返回:
        (PersistentClient, collection)
    """
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection


def _delete_chunk_ids(index: VectorStoreIndex, chunk_ids: list) -> None:
    """从向量库删除一组 chunk。

    参数:
        index: LlamaIndex 向量索引
        chunk_ids: 要删除的 node id 列表
    """
    if not chunk_ids:
        return
    index.delete_nodes(chunk_ids, delete_from_docstore=True)


def _nodes_from_document(file_path: str, text: str, file_name: str, file_hash: str) -> list:
    """把一篇文档做成带 metadata 的 TextNode 列表。

    参数:
        file_path: 源文件路径
        text: 解析后的正文
        file_name: 文件名
        file_hash: 文件内容 hash
    返回:
        TextNode 列表
    """
    file_type = os.path.splitext(file_name)[1].lstrip(".").lower() or "txt"
    chunks = split_chunks(text)
    nodes = []
    for idx, chunk in enumerate(chunks):
        node_id = f"{hashlib.sha256(f'{file_name}:{file_hash}:{idx}'.encode('utf-8')).hexdigest()[:20]}"
        nodes.append(TextNode(
            text=chunk,
            id_=node_id,
            metadata={
                "file_name": file_name,
                "file_path": file_path,
                "file_type": file_type,
                "chunk_id": idx,
                "content_hash": file_hash,
            },
        ))
    return nodes


def build_chroma_index(docs_dir: str = DOCS_DIR, chroma_dir: str = CHROMA_DIR) -> VectorStoreIndex:
    """扫描文档目录，按文件 hash 做增量更新，返回挂在 Chroma 上的索引。

    未变化的文件跳过；变化或新增则重建该文件的 chunks；目录中已删除的文件从库中清掉。

    参数:
        docs_dir: 知识库目录
        chroma_dir: Chroma 持久化目录
    返回:
        VectorStoreIndex
    """
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"文档目录不存在: {docs_dir}，请先放入待检索文件")

    # 打开 Chroma 数据库
    _client, collection = _open_collection()
    # 创建向量存储
    vector_store = ChromaVectorStore(chroma_collection=collection)
    # 创建存储上下文
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # 创建索引
    # 至此，索引创建完成：连接 Chroma 数据库，创建向量存储，创建存储上下文，创建索引
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

    # 加载文档
    documents = SimpleDirectoryReader(docs_dir).load_data()
    if not documents:
        raise ValueError(f"文档目录为空: {docs_dir}，请先放入待检索文件")

    # 加载清单：_load_manifest() 是用于加载 Chroma 数据库中的清单文件，以跟踪哪些文件已经索引过
    manifest = _load_manifest()
    files_meta = manifest.get("files", {})
    seen_names = set()

    for doc in documents:
        file_path = doc.metadata.get("file_path") or ""
        file_name = doc.metadata.get("file_name") or os.path.basename(file_path)
        if not file_path or not os.path.isfile(file_path):
            raise FileNotFoundError(f"无法定位源文件: {file_name}")
        seen_names.add(file_name)
        file_hash = _file_sha256(file_path)
        prev = files_meta.get(file_name)
        if (
            prev
            and prev.get("hash") == file_hash
            and prev.get("chunk_spec") == CHUNK_SPEC
        ):
            continue
        if prev and prev.get("chunk_ids"):
            _delete_chunk_ids(index, prev["chunk_ids"])
        nodes = _nodes_from_document(file_path, doc.text, file_name, file_hash)
        if not nodes:
            raise ValueError(f"文档分块结果为空: {file_name}")
        index.insert_nodes(nodes)
        files_meta[file_name] = {
            "hash": file_hash,
            "chunk_ids": [node.node_id for node in nodes],
            "file_type": os.path.splitext(file_name)[1].lstrip(".").lower(),
            "chunk_spec": CHUNK_SPEC,
        }
        print(f"[Chroma] 已索引 {file_name}，{len(nodes)} 个 chunk")

    removed = [name for name in list(files_meta.keys()) if name not in seen_names]
    for name in removed:
        _delete_chunk_ids(index, files_meta[name].get("chunk_ids") or [])
        del files_meta[name]
        print(f"[Chroma] 已删除离库文件 {name}")

    manifest["files"] = files_meta
    _save_manifest(manifest)
    print(f"[Chroma] 索引就绪，文档 {len(files_meta)} 个，目录 {chroma_dir}")
    return index


def load_all_chunks(index: VectorStoreIndex) -> list:
    """从索引取出全部 chunk，供 BM25 建库。

    参数:
        index: Chroma 上的 VectorStoreIndex
    返回:
        dict 列表，每项含 text / file_name / file_type / chunk_id / node_id
    """
    vector_store = index.vector_store
    collection = vector_store._collection
    raw = collection.get(include=["documents", "metadatas"])
    rows = []
    ids = raw.get("ids") or []
    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []
    for node_id, text, meta in zip(ids, documents, metadatas):
        meta = meta or {}
        rows.append({
            "node_id": node_id,
            "text": text or "",
            "file_name": meta.get("file_name", "未知"),
            "file_type": meta.get("file_type", ""),
            "chunk_id": meta.get("chunk_id"),
        })
    if not rows:
        raise ValueError("Chroma 中没有 chunk，请先放入文档并重建索引")
    return rows
