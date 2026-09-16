#!/usr/bin/env python
# coding: utf-8
"""Flask 路由：问答 invoke / SSE stream，按 thread_id 落库。"""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from flask.cli import load_dotenv

from agent import answer_question, init_agent, stream_answer
from models import ChatMessage, db

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
VITE_ORIGIN = "http://127.0.0.1:5173"
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
# 清空代理环境变量：防止代理影响请求。
for _proxy_key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_proxy_key, None)


def _is_dev() -> bool:
    """是否走 Vite 开发脚本。

    返回:
        RAG_ENV 不是 prod 时为 True
    """
    return os.getenv("RAG_ENV", "dev") != "prod"


def _proxy_vite(path: str):
    """开发态把 Vite 资源转到 5173，脚本与页面同源。

    参数:
        path: 浏览器请求路径
    返回:
        Flask Response
    """
    url = f"{VITE_ORIGIN}/{path}"
    qs = request.query_string.decode()
    if qs:
        url = f"{url}?{qs}"
    try:
        with urlopen(Request(url, method="GET"), timeout=10) as resp:
            return Response(
                resp.read(),
                status=resp.status,
                content_type=resp.headers.get("Content-Type", "application/javascript"),
            )
    except HTTPError as exc:
        return Response(exc.read(), status=exc.code)
    except URLError:
        return jsonify({"error": "Vite 未启动，请先 npm run dev"}), 502


def _require_thread_id(raw) -> str:
    """校验 thread_id。

    参数:
        raw: 请求里的 thread_id
    返回:
        去空白后的 thread_id
    """
    thread_id = (raw or "").strip()
    if not thread_id:
        raise ValueError("thread_id 不能为空")
    return thread_id


def _load_history(thread_id: str) -> list:
    """按会话取出已落库消息，供滑动窗口记忆使用。

    参数:
        thread_id: 会话 id
    返回:
        [{role, content}, ...]
    """
    rows = (
        db.session.query(ChatMessage)
        .filter_by(thread_id=thread_id)
        .order_by(ChatMessage.id.asc())
        .all()
    )
    return [row.to_history() for row in rows]


def _save_turn(thread_id: str, question: str, result: dict) -> None:
    """写入本轮用户问题与助手回答。

    参数:
        thread_id: 会话 id
        question: 用户原文
        result: invoke/stream 完成后的 {answer, sources, report, rewritten_query}
    """
    db.session.add(ChatMessage(
        thread_id=thread_id,
        role="user",
        content=question,
    ))
    db.session.add(ChatMessage(
        thread_id=thread_id,
        role="assistant",
        content=result["answer"],
        sources_json=json.dumps(result.get("sources") or [], ensure_ascii=False),
        report=result.get("report") or "",
        rewritten_query=result.get("rewritten_query") or "",
    ))
    db.session.commit()


def create_app():
    """组装 Flask 应用并在启动时初始化 RAG 运行时。

    返回:
        Flask app
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    app = Flask(
        __name__,
        template_folder=os.path.join(PROJECT_ROOT, "templates"),
        static_folder=os.path.join(PROJECT_ROOT, "static"),
    )
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(DATA_DIR, "app.db")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)

    # 创建数据库表。
    with app.app_context():
        db.create_all()

    runtime = init_agent()

    @app.get("/")
    def home():
        return render_template("index.html", dev=_is_dev())

    @app.get("/api/chat/messages")
    def chat_messages():
        try:
            thread_id = _require_thread_id(request.args.get("thread_id"))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        rows = (
            db.session.query(ChatMessage)
            .filter_by(thread_id=thread_id)
            .order_by(ChatMessage.id.asc())
            .all()
        )
        return jsonify({
            "thread_id": thread_id,
            "messages": [row.to_client() for row in rows],
        })

    @app.post("/api/chat")
    def chat():
        body = request.get_json(force=True)
        question = (body.get("question") or "").strip()
        try:
            thread_id = _require_thread_id(body.get("thread_id"))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        if not question:
            return jsonify({"error": "question 不能为空"}), 400

        history = _load_history(thread_id)
        result = answer_question(runtime, question, history)
        _save_turn(thread_id, question, result)
        result["thread_id"] = thread_id
        return jsonify(result)

    @app.post("/api/chat/stream")
    def chat_stream():
        body = request.get_json(force=True)
        question = (body.get("question") or "").strip()
        try:
            thread_id = _require_thread_id(body.get("thread_id"))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        if not question:
            return jsonify({"error": "question 不能为空"}), 400

        history = _load_history(thread_id)

        def generate():
            collected = {
                "answer": "",
                "sources": [],
                "report": "",
                "rewritten_query": "",
            }
            try:
                for event in stream_answer(runtime, question, history):
                    if event["type"] == "token":
                        collected["answer"] += event.get("text") or ""
                    elif event["type"] == "rewrite":
                        collected["rewritten_query"] = event.get("query") or ""
                    elif event["type"] == "sources":
                        collected["sources"] = event.get("sources") or []
                    elif event["type"] == "report":
                        collected["report"] = event.get("report") or ""
                    elif event["type"] == "done":
                        collected["answer"] = event.get("answer") or collected["answer"]
                        collected["sources"] = event.get("sources") or collected["sources"]
                        collected["report"] = event.get("report") or collected["report"]
                        collected["rewritten_query"] = (
                            event.get("rewritten_query") or collected["rewritten_query"]
                        )
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                _save_turn(thread_id, question, collected)
            except Exception as exc:
                db.session.rollback()
                yield f"data: {json.dumps({'type': 'error', 'error': str(exc)}, ensure_ascii=False)}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    if _is_dev():
        @app.get("/<path:path>")
        def vite_dev_proxy(path):
            return _proxy_vite(path)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
