# coding: utf-8
"""Query 改写、混合检索、Rerank、Lost-in-Middle、滑动窗口记忆。

BM25 与向量检索用 LCEL RunnableParallel 并行，再 RRF 融合。
"""

import jieba
from langchain_core.documents import Document as LCDocument
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableParallel
from rank_bm25 import BM25Okapi

MEMORY_WINDOW = 6
VECTOR_TOP_K = 8
BM25_TOP_K = 8
RERANK_TOP_K = 5
RRF_K = 60
RERANK_RETRY = 3


def slide_history(history: list, window: int = MEMORY_WINDOW) -> list:
    """只保留最近 N 条消息，避免 Token 随对话线性膨胀。

    参数:
        history: [{role, content}, ...]，按时间升序
        window: 保留条数，默认 6（约 3 轮）
    返回:
        截断后的 history
    """
    if window <= 0:
        raise ValueError("window 必须大于 0")
    return history[-window:]


def to_lc_messages(history: list) -> list:
    """把落库 history 转成 LangChain 消息。

    参数:
        history: [{role, content}, ...]
    返回:
        HumanMessage / AIMessage 列表
    """
    messages = []
    for item in history:
        if item["role"] == "user":
            messages.append(HumanMessage(content=item["content"]))
        elif item["role"] == "assistant":
            messages.append(AIMessage(content=item["content"]))
    return messages


def create_rewrite_chain(llm):
    """用历史对话把指代消解成可检索问句。

    参数:
        llm: ChatTongyi 等聊天模型
    返回:
        LCEL 链，输入 {question, history}，输出改写后的检索问句
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你负责把用户当前问题改写成适合检索保险知识库的完整问句。
结合历史对话消解「它」「这个产品」「能不能赔」等指代，补全险种或主体名称。
只输出一个问句，不要答案，不要来源，不要解释，不要换行。"""),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])
    return prompt | llm | StrOutputParser()


def rewrite_query(rewrite_chain, question: str, history: list) -> str:
    """执行 Query 改写。无历史时直接返回原问句。

    参数:
        rewrite_chain: create_rewrite_chain 的返回值
        question: 本轮用户问题
        history: 滑动窗口后的历史
    返回:
        用于检索的问句
    """
    if not history:
        return question
    rewritten = rewrite_chain.invoke({
        "question": question,
        "history": to_lc_messages(history),
    })
    rewritten = (rewritten or "").strip()
    if not rewritten:
        raise ValueError("Query 改写结果为空")
    return rewritten


def _tokenize(text: str) -> list:
    """中文分词，给 BM25 用。

    参数:
        text: 待分词文本
    返回:
        token 列表
    """

    # jieba.lcut 是 jieba 库的默认分词模式，它会根据词典和语料库进行分词，并返回一个列表。
    # 这个列表中的每个元素都是分词后的一个词。
    # 例如，对于文本 "这是一个测试", jieba.lcut 会返回 ["这", "是", "一个", "测试"]。
    # 这个列表中的每个元素都是分词后的一个词。
    return [tok for tok in jieba.lcut(text) if tok.strip()]


def build_bm25(chunks: list) -> tuple:
    """用全部 chunk 建 BM25 索引。

    参数:
        chunks: ingest.load_all_chunks 的返回值
    返回:
        (BM25Okapi, chunks)
    """
    corpus = [_tokenize(row["text"]) for row in chunks]

    # BM25Okapi 是 BM25 算法的实现类，它接受一个列表作为参数，列表中的每个元素都是分词后的一个词。
    # 例如，对于文本 "这是一个测试", BM25Okapi 会返回一个 BM25Okapi 对象，它接受一个列表作为参数，列表中的每个元素都是分词后的一个词。
    return BM25Okapi(corpus), chunks


def _chunk_to_lc(row: dict, score) -> LCDocument:
    """把内部 chunk 转成 LangChain Document。

    参数:
        row: 含 text / file_name 等字段
        score: 检索分
    返回:
        LCDocument
    """
    return LCDocument(
        page_content=row["text"],
        metadata={
            # 将 score 四舍五入到小数点后 4 位，如果 score 为 None，则返回 None。
            # 例如，对于 score 0.123456789，round(float(score), 4) 会返回 0.1235。
            "score": round(float(score), 4) if score is not None else None,
            "file_name": row["file_name"],
            "file_type": row.get("file_type", ""),
            "chunk_id": row.get("chunk_id"),
            "node_id": row.get("node_id"),
        },
    )


def bm25_search(bm25, chunks: list, query: str, top_k: int = BM25_TOP_K) -> list:
    """BM25 关键词检索。

    参数:
        bm25: BM25Okapi
        chunks: 与 bm25 语料顺序一致的 chunk 列表
        query: 检索问句
        top_k: 返回条数
    返回:
        LCDocument 列表
    """

    # bm25.get_scores 是 BM25Okapi 对象的 get_scores 方法，它接受一个列表作为参数，列表中的每个元素都是分词后的一个词。
    # 返回：[0.1, 0.2, 0.3, 0.4, 0.5]
    scores = bm25.get_scores(_tokenize(query))

    # 将 scores 中的元素和索引打包成元组，然后按分数降序排序，最后截取前 top_k 个元素。
    # 返回：[(0, 0.5), (1, 0.4), (2, 0.3), (3, 0.2), (4, 0.1)]
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
    return [_chunk_to_lc(chunks[idx], score) for idx, score in ranked]


def vector_search(index, query: str, top_k: int = VECTOR_TOP_K) -> list:
    """Chroma 向量检索，结果转为 LangChain Document。

    参数:
        index: VectorStoreIndex
        query: 检索问句
        top_k: 返回条数
    返回:
        LCDocument 列表
    """
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    docs = []
    for node in nodes:
        docs.append(LCDocument(
            page_content=node.text,
            metadata={
                "score": round(node.score, 4) if node.score is not None else None,
                "file_name": node.metadata.get("file_name", "未知"),
                "file_type": node.metadata.get("file_type", ""),
                "chunk_id": node.metadata.get("chunk_id"),
                "node_id": node.node_id,
            },
        ))
    return docs


def rrf_fuse(result_lists: list, rrf_k: int = RRF_K) -> list:
    """倒数排名融合多路检索结果。

    参数:
        result_lists: 多路 LCDocument 列表
        rrf_k: RRF 平滑常数
    返回:
        按融合分降序的 LCDocument 列表
    """
    scores = {}
    docs = {}
    for result_list in result_lists:
        for rank, doc in enumerate(result_list, start=1):
            key = doc.metadata.get("node_id") or (doc.metadata.get("file_name"), doc.page_content[:80])
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            docs[key] = doc
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    fused = []
    for key, score in ranked:
        doc = docs[key]
        doc.metadata["score"] = round(score, 6)
        fused.append(doc)
    return fused


def rerank_dashscope(api_key: str, query: str, docs: list, top_k: int = RERANK_TOP_K) -> list:
    """调用 DashScope Rerank 对融合结果重排，失败则有限次重试。

    参数:
        api_key: DashScope Key
        query: 检索问句
        docs: 待重排文档
        top_k: 截断条数
    返回:
        重排后的 LCDocument 列表
    """
    if not docs:
        raise ValueError("没有可供重排的文档")
    import dashscope
    from dashscope import TextReRank

    dashscope.api_key = api_key
    texts = [doc.page_content for doc in docs]
    last_error = None
    resp = None
    for _ in range(RERANK_RETRY):
        try:
            resp = TextReRank.call(
                model="qwen3-rerank",
                query=query,
                documents=texts,
                top_n=min(top_k, len(texts)),
                return_documents=False,
            )
            if resp.status_code == 200:
                break
            last_error = RuntimeError(f"Rerank 失败: {resp.code} {resp.message}")
        except Exception as exc:
            last_error = exc
            resp = None
    if resp is None or resp.status_code != 200:
        raise RuntimeError(f"Rerank 重试 {RERANK_RETRY} 次仍失败: {last_error}")
    results = resp.output.get("results") if isinstance(resp.output, dict) else resp.output.results
    reranked = []
    for item in results:
        if isinstance(item, dict):
            idx = item["index"]
            score = item.get("relevance_score")
        else:
            idx = item.index
            score = getattr(item, "relevance_score", None)
        doc = docs[idx]
        doc.metadata["score"] = round(float(score), 4) if score is not None else doc.metadata.get("score")
        reranked.append(doc)
    return reranked


def lost_in_middle_reorder(docs: list) -> list:
    """缓解 Lost-in-Middle：高分放两端，较低分放中间。

    参数:
        docs: 已按相关性降序的文档
    返回:
        重排后的文档列表
    """
    if len(docs) <= 2:
        return list(docs)
    even = docs[0::2]
    odd = docs[1::2]
    odd.reverse()
    return even + odd


def format_context(docs: list) -> str:
    """拼给 LLM 的带溯源 context。

    参数:
        docs: 检索文档
    返回:
        带 [来源: 文件名] 前缀的文本
    """
    parts = []
    for doc in docs:
        parts.append(f"[来源: {doc.metadata['file_name']}]\n{doc.page_content}")
    return "\n\n".join(parts)


def docs_to_sources(docs: list) -> list:
    """转成前端 / 评估用的 sources 结构。

    参数:
        docs: 检索文档
    返回:
        [{file_name, score, preview}, ...]
    """
    return [
        {
            "file_name": doc.metadata["file_name"],
            "score": doc.metadata.get("score"),
            "preview": doc.page_content[:200],
        }
        for doc in docs
    ]


def create_hybrid_retriever(index, bm25, chunks: list):
    """BM25 与向量检索并行的 LCEL RunnableParallel。

    参数:
        index: VectorStoreIndex
        bm25: BM25Okapi
        chunks: BM25 语料对应的 chunk 列表
    返回:
        输入 query 字符串，输出 {"bm25": [...], "vector": [...]}
    """

    # 带 score 的 BM25 检索和向量检索。
    return RunnableParallel(
        bm25=RunnableLambda(lambda query: bm25_search(bm25, chunks, query)),
        vector=RunnableLambda(lambda query: vector_search(index, query)),
    )


def hybrid_retrieve(parallel, api_key: str, query: str, top_k: int = RERANK_TOP_K) -> list:
    """并行检索 → RRF → Rerank → Lost-in-Middle。

    参数:
        parallel: create_hybrid_retriever 返回的 Runnable
        api_key: DashScope Key
        query: 改写后的检索问句
        top_k: 最终条数
    返回:
        供生成使用的 LCDocument 列表
    """
    bundled = parallel.invoke(query)
    fused = rrf_fuse([bundled["bm25"], bundled["vector"]])
    if not fused:
        raise ValueError(f"混合检索无结果: {query}")
    reranked = rerank_dashscope(api_key, query, fused, top_k=top_k)
    return lost_in_middle_reorder(reranked)
