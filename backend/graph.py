# coding: utf-8
"""保险 RAG 的 LangGraph ReAct：检索、报告、落盘。

死循环：连续相同 (tool, args) 直接停。最大步数走 recursion_limit。
"""

import json
import os
import time
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from retrieve import (
    docs_to_sources,
    format_context,
    hybrid_retrieve,
    rewrite_query,
    slide_history,
    to_lc_messages,
)
from safety import INJECTION_RULE, wrap_user_input

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
MAX_REACT_STEPS = 8
TOOL_RETRY = 3


class DeadLoopError(RuntimeError):
    """连续重复调用同一工具与参数。"""


class LoopGuard:
    """记录工具调用，检测死循环与步数上限。"""

    def __init__(self, max_steps: int = MAX_REACT_STEPS):
        self.max_steps = max_steps
        self.calls = []

    def before_call(self, name: str, args: dict) -> None:
        """在真正执行工具前检查。

        参数:
            name: 工具名
            args: 工具参数
        """
        if len(self.calls) >= self.max_steps:
            raise RuntimeError(f"超过最大执行步数 {self.max_steps}")
        key = (name, json.dumps(args, ensure_ascii=False, sort_keys=True))
        if self.calls and self.calls[-1] == key:
            raise DeadLoopError(f"检测到死循环：连续重复调用 {name}")
        self.calls.append(key)


class ReactState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def _tool_args(tool_call: dict) -> dict:
    """从 LangGraph tool_call 取出参数字典。

    参数:
        tool_call: 模型产出的一次工具调用
    返回:
        工具参数 dict
    """
    raw = tool_call.get("args") or tool_call.get("arguments") or {}
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def _invoke_tool(selected, args: dict):
    """执行单个工具，瞬时失败则有限次重试；死循环与步数上限立即抛出。

    参数:
        selected: LangChain Tool
        args: 工具参数
    返回:
        工具执行结果
    """
    last_error = None
    for _ in range(TOOL_RETRY):
        try:
            return selected.invoke(args)
        except DeadLoopError:
            raise
        except RuntimeError as exc:
            if str(exc).startswith("超过最大执行步数"):
                raise
            last_error = exc
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"{selected.name} 重试 {TOOL_RETRY} 次仍失败: {last_error}")


def build_rag_tools(runtime, guard: LoopGuard, question: str):
    """注册 retrieve_docs / generate_report / save_report。

    参数:
        runtime: agent.RagRuntime
        guard: 本次请求的死循环检测器
        question: 用户原问句
    返回:
        (tools 列表, 可变 bucket 字典，用来回传 docs/report/saved_path)
    """
    bucket = {"docs": [], "report": "", "saved_path": ""}

    @tool
    def retrieve_docs(query: str) -> str:
        """从保险知识库检索相关文档片段。输入应是完整检索问句。

        参数:
            query: 检索问句
        返回:
            带来源标注的文档正文
        """
        guard.before_call("retrieve_docs", {"query": query})
        docs = hybrid_retrieve(runtime.parallel, runtime.api_key, query)
        bucket["docs"] = docs
        return format_context(docs)

    @tool
    def generate_report(answer: str, sources: str) -> str:
        """根据本轮问答生成结构化分析报告。

        参数:
            answer: 助手已经给出的回答
            sources: 来源文件名，逗号分隔
        返回:
            分析报告正文
        """
        guard.before_call("generate_report", {"answer": answer, "sources": sources})
        report = runtime.report_chain.invoke({
            "question": question,
            "answer": answer,
            "sources": sources,
        })
        bucket["report"] = report
        return report

    @tool
    def save_report(content: str, filename: str = "") -> str:
        """把分析报告保存到本地 reports 目录。

        参数:
            content: 报告正文
            filename: 可选文件名，空则按时间生成
        返回:
            保存路径
        """
        guard.before_call("save_report", {"content": content, "filename": filename})
        os.makedirs(REPORTS_DIR, exist_ok=True)
        name = filename.strip() or f"report-{int(time.time())}.txt"
        if "/" in name or "\\" in name:
            raise ValueError("filename 不能包含路径分隔符")
        path = os.path.join(REPORTS_DIR, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
        bucket["saved_path"] = path
        return path

    return [retrieve_docs, generate_report, save_report], bucket


def build_react_graph(llm, tools):
    """LangGraph ReAct：模型决定调工具，直到给出最终回复。

    参数:
        llm: 支持 bind_tools 的聊天模型
        tools: 三个 @tool
    返回:
        编译后的 StateGraph
    """
    tools_by_name = {t.name: t for t in tools}
    bound = llm.bind_tools(tools)

    def agent_node(state: ReactState) -> dict:
        """调用 LLM，可能产生 tool_calls。

        参数:
            state: 含 messages 的图状态
        返回:
            追加的 AIMessage
        """
        response = bound.invoke(state["messages"])
        return {"messages": [response]}

    def tools_node(state: ReactState) -> dict:
        """执行本轮全部 tool_calls。

        参数:
            state: 含 messages 的图状态
        返回:
            ToolMessage 列表
        """
        last = state["messages"][-1]
        outputs = []
        for call in last.tool_calls:
            name = call["name"]
            args = _tool_args(call)
            selected = tools_by_name[name]
            try:
                result = _invoke_tool(selected, args)
            except DeadLoopError as exc:
                result = str(exc)
            except RuntimeError as exc:
                result = str(exc)
            outputs.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        return {"messages": outputs}

    def should_continue(state: ReactState) -> str:
        """有 tool_calls 则进工具节点，否则结束。

        参数:
            state: 图状态
        返回:
            tools 或 end
        """
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "end"

    graph = StateGraph(ReactState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def create_qa_chain(llm):
    """从原 agent.py 拷贝的 LCEL 问答链，要求基于检索上下文作答。

    参数:
        llm: 聊天模型
    返回:
        LCEL 链
    """
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.output_parsers import StrOutputParser

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的保险产品顾问。
根据以下检索到的文档内容回答用户的问题。
只依据检索上下文作答，不要编造条款。如果文档中没有相关信息，请如实说明。
""" + INJECTION_RULE + """
回答中用文件名标注溯源，例如（来源：雇主责任险.txt）。请用中文回复。

检索到的文档内容:
{context}"""),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])
    return qa_prompt | llm | StrOutputParser()


def create_report_chain(llm):
    """从原 agent.py 拷贝的 LCEL 报告链。

    参数:
        llm: 聊天模型
    返回:
        LCEL 链
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    report_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的保险产品分析师。
根据以下问答记录，生成一份结构化的分析报告。
报告需包含：概述、核心要点、适用场景、注意事项。
请用中文撰写，语言简洁专业。

问答记录:
问题: {question}
回答: {answer}

相关文档来源: {sources}"""),
        ("human", "请生成分析报告"),
    ])
    return report_prompt | llm | StrOutputParser()


def stream_rag(runtime, question: str, history: list):
    """流式跑完：改写 → 混合检索 → 问答 token → ReAct 报告/存文件。

    参数:
        runtime: RagRuntime
        question: 用户问题
        history: 落库历史（未截断）
    产出:
        事件 dict：rewrite / sources / token / report / done
    """
    history = slide_history(history)
    rewritten = rewrite_query(runtime.rewrite_chain, question, history)
    yield {"type": "rewrite", "query": rewritten}

    docs = hybrid_retrieve(runtime.parallel, runtime.api_key, rewritten)
    sources = docs_to_sources(docs)
    yield {"type": "sources", "sources": sources}

    context = format_context(docs)
    answer_parts = []
    for chunk in runtime.qa_chain.stream({
        "context": context,
        "question": wrap_user_input(question),
        "history": to_lc_messages(history),
    }):
        answer_parts.append(chunk)
        yield {"type": "token", "text": chunk}
    answer = "".join(answer_parts)

    guard = LoopGuard()
    tools, bucket = build_rag_tools(runtime, guard, question)
    bucket["docs"] = docs
    graph = build_react_graph(runtime.llm, tools)
    source_names = ", ".join(sorted({item["file_name"] for item in sources}))
    react_input = {
        "messages": [
            SystemMessage(content=(
                "你是保险知识库 Agent。检索与回答已经完成。\n"
                f"{INJECTION_RULE}\n"
                "必须先调用 generate_report，再调用 save_report 把报告落到本地。"
                "不要重复调用同一工具；不要再检索，除非报告明显缺少来源。"
            )),
            HumanMessage(content=(
                f"问题: {wrap_user_input(question)}\n"
                f"改写问句: {rewritten}\n"
                f"回答: {answer}\n"
                f"来源: {source_names}"
            )),
        ]
    }
    graph.invoke(react_input, {"recursion_limit": MAX_REACT_STEPS})
    report = bucket["report"]
    if not report:
        raise RuntimeError("Agent 未调用 generate_report，请检查工具调用")
    if not bucket.get("saved_path"):
        raise RuntimeError("Agent 未调用 save_report，请检查工具调用")
    yield {
        "type": "report",
        "report": report,
        "saved_path": bucket.get("saved_path") or "",
    }
    yield {
        "type": "done",
        "answer": answer,
        "sources": sources,
        "report": report,
        "rewritten_query": rewritten,
        "saved_path": bucket.get("saved_path") or "",
    }


def invoke_rag(runtime, question: str, history: list) -> dict:
    """非流式入口，给评估脚本和兼容接口用。

    参数:
        runtime: RagRuntime
        question: 用户问题
        history: 落库历史
    返回:
        {answer, sources, report, rewritten_query, saved_path}
    """
    result = None
    for event in stream_rag(runtime, question, history):
        if event["type"] == "done":
            result = event
    if result is None:
        raise RuntimeError("RAG 图没有产生 done 事件")
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "report": result["report"],
        "rewritten_query": result.get("rewritten_query", ""),
        "saved_path": result.get("saved_path", ""),
    }
