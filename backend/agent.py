# coding: utf-8
"""RAG 运行时门面：启动时建索引，问答走 graph.stream_rag。

检索、分块、ReAct 已拆到 ingest / retrieve / graph。本文件只给 Flask 调用。
"""

import os
from dataclasses import dataclass

from langchain_community.chat_models import ChatTongyi
from llama_index.core import Settings
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from llama_index.llms.dashscope import DashScope

from graph import create_qa_chain, create_report_chain, invoke_rag, stream_rag
from ingest import build_chroma_index, load_all_chunks
from retrieve import build_bm25, create_hybrid_retriever, create_rewrite_chain

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _require_api_key() -> str:
    """读取 DashScope Key，未设置则启动失败。

    返回:
        API Key 字符串
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")
    return api_key


def setup_llamaindex():
    """配置 LlamaIndex 全局 LLM 与 Embedding，供 Chroma 向量检索使用。

    返回:
        (DashScope llm, DashScopeEmbedding)
    """
    api_key = _require_api_key()
    llm = DashScope(
        model="deepseek-v3",
        api_key=api_key,
        temperature=0.1,
        top_p=0.8,
    )
    embed_model = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    )
    Settings.llm = llm
    Settings.embed_model = embed_model
    return llm, embed_model


@dataclass
class RagRuntime:
    """一次进程内复用的 RAG 运行时。"""

    api_key: str
    index: object
    chunks: list
    bm25: object
    parallel: object
    llm: object
    qa_chain: object
    report_chain: object
    rewrite_chain: object


def init_agent() -> RagRuntime:
    """进程启动时调用一次：Chroma 增量索引 + 混合检索器 + LCEL 链。

    返回:
        RagRuntime
    """
    setup_llamaindex()
    api_key = _require_api_key()
    index = build_chroma_index()
    chunks = load_all_chunks(index)
    bm25, chunks = build_bm25(chunks)
    parallel = create_hybrid_retriever(index, bm25, chunks)
    llm = ChatTongyi(model_name="deepseek-v3", dashscope_api_key=api_key)
    return RagRuntime(
        api_key=api_key,
        index=index,
        chunks=chunks,
        bm25=bm25,
        parallel=parallel,
        llm=llm,
        qa_chain=create_qa_chain(llm),
        report_chain=create_report_chain(llm),
        rewrite_chain=create_rewrite_chain(llm),
    )


def answer_question(runtime: RagRuntime, question: str, history: list) -> dict:
    """非流式问答，给评估脚本和 POST /api/chat 使用。

    参数:
        runtime: init_agent 返回值
        question: 用户问题
        history: 同一 thread_id 的已落库记录
    返回:
        {answer, sources, report, rewritten_query, saved_path}
    """
    return invoke_rag(runtime, question, history)


def stream_answer(runtime: RagRuntime, question: str, history: list):
    """流式问答，给 POST /api/chat/stream 使用。

    参数:
        runtime: init_agent 返回值
        question: 用户问题
        history: 同一 thread_id 的已落库记录
    产出:
        graph.stream_rag 的事件
    """
    yield from stream_rag(runtime, question, history)
