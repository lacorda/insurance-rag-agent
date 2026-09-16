# coding: utf-8
"""保险 RAG 检索评估：HitRate@K、MRR、RAGAS faithfulness / answer_relevancy。

在 backend 目录执行: python eval_rag.py
"""

import json
import os
import sys

from flask.cli import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
for _proxy_key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_proxy_key, None)

from agent import answer_question, init_agent
from retrieve import hybrid_retrieve, rewrite_query

GOLD_PATH = os.path.join(os.path.dirname(__file__), "gold_set.json")
TOP_K = 5


def _hit(retrieved_files: list, expected_files: list) -> bool:
    """检索到的文件名是否命中任一期望来源。

    参数:
        retrieved_files: 实际召回的 file_name 列表（按排序）
        expected_files: gold 中的期望文件名
    返回:
        是否命中
    """
    expected = set(expected_files)
    return any(name in expected for name in retrieved_files)


def _mrr(retrieved_files: list, expected_files: list) -> float:
    """第一份相关文档的倒数排名。

    参数:
        retrieved_files: 实际召回的 file_name 列表（按排序）
        expected_files: gold 中的期望文件名
    返回:
        1/rank，未命中为 0
    """
    expected = set(expected_files)
    for rank, name in enumerate(retrieved_files, start=1):
        if name in expected:
            return 1.0 / rank
    return 0.0


def _run_ragas(rows: list, runtime) -> dict:
    """用 RAGAS 算 faithfulness 与 answer_relevancy。

    参数:
        rows: 含 question / answer / contexts / ground_truth 的样本
        runtime: RagRuntime，复用同一套 LLM / Embedding
    返回:
        ragas evaluate 的分数字典
    """
    from datasets import Dataset
    from langchain_community.embeddings import DashScopeEmbeddings
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, faithfulness

    dataset = Dataset.from_list(rows)
    llm = LangchainLLMWrapper(runtime.llm)
    embeddings = LangchainEmbeddingsWrapper(
        DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key=runtime.api_key)
    )
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
    )
    return dict(result)


def main() -> None:
    """加载 gold 集，跑检索指标和 RAGAS。"""
    with open(GOLD_PATH, "r", encoding="utf-8") as handle:
        gold = json.load(handle)

    runtime = init_agent()
    hits = []
    mrrs = []
    ragas_rows = []

    for item in gold:
        question = item["question"]
        expected = item["expected_files"]
        rewritten = rewrite_query(runtime.rewrite_chain, question, [])
        docs = hybrid_retrieve(runtime.parallel, runtime.api_key, rewritten, top_k=TOP_K)
        files = [doc.metadata["file_name"] for doc in docs]
        hit = _hit(files, expected)
        mrr = _mrr(files, expected)
        hits.append(1.0 if hit else 0.0)
        mrrs.append(mrr)
        print(f"Q: {question}")
        print(f"  rewritten: {rewritten}")
        print(f"  files: {files}")
        print(f"  HitRate@{TOP_K}={int(hit)}  RR={mrr:.4f}")

        result = answer_question(runtime, question, [])
        ragas_rows.append({
            "user_input": question,
            "response": result["answer"],
            "retrieved_contexts": [doc.page_content for doc in docs],
            "reference": item["ground_truth"],
        })

    hit_rate = sum(hits) / len(hits)
    mrr = sum(mrrs) / len(mrrs)
    print("\n=== 检索指标 ===")
    print(f"HitRate@{TOP_K}: {hit_rate:.4f}")
    print(f"MRR: {mrr:.4f}")

    print("\n=== RAGAS ===")
    scores = _run_ragas(ragas_rows, runtime)
    for key, value in scores.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    sys.exit(main())
