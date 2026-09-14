# 保险知识库 RAG 智能问答 Agent

模拟保险企业私有知识库：多格式文档检索、文档溯源、多轮问答、报告生成。

## 启动

1. 复制 `.env.example` 为 `.env`，填入 `DASHSCOPE_API_KEY`
2. 后端

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

3. 前端（另开终端）

```bash
cd frontend
npm install
npm run dev
```

浏览器打开 `http://127.0.0.1:5000`。

检索评估：

```bash
cd backend
source .venv/bin/activate
python eval_rag.py
```
