# coding: utf-8
"""对话记录。库文件在项目根 data/app.db，与向量索引分离。"""

import json
from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class ChatMessage(db.Model):
    """一条对话消息。同一 thread_id 即同一条问答会话。

    参数由 SQLAlchemy 列定义，不经过构造函数传入业务字段。
    """

    __tablename__ = "chat_messages"

    id = db.Column(db.Integer, primary_key=True)
    thread_id = db.Column(db.String(64), nullable=False, index=True)
    role = db.Column(db.String(16), nullable=False)
    content = db.Column(db.Text, nullable=False)
    sources_json = db.Column(db.Text)
    report = db.Column(db.Text)
    rewritten_query = db.Column(db.Text)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    def to_client(self):
        """转成前端气泡结构。

        返回:
            {role, text, sources, report, rewrittenQuery}
        """
        sources = json.loads(self.sources_json) if self.sources_json else []
        return {
            "role": self.role,
            "text": self.content,
            "sources": sources,
            "report": self.report or "",
            "rewrittenQuery": self.rewritten_query or "",
        }

    def to_history(self):
        """转成检索改写用的滑动窗口条目。

        返回:
            {role, content}
        """
        return {"role": self.role, "content": self.content}
