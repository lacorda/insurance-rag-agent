import { useEffect, useState } from "react";
import { askQuestionStream, listMessages } from "./api.js";

const THREAD_KEY = "insurance-rag-thread-id";

function emptyAssistant() {
  return {
    role: "assistant",
    text: "",
    sources: [],
    report: "",
    rewrittenQuery: "",
  };
}

export default function Chat() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [historyReady, setHistoryReady] = useState(false);
  const [error, setError] = useState("");
  const [threadId] = useState(() => {
    const existing = localStorage.getItem(THREAD_KEY);
    if (existing) {
      return existing;
    }
    const id = crypto.randomUUID();
    localStorage.setItem(THREAD_KEY, id);
    return id;
  });

  useEffect(() => {
    listMessages(threadId)
      .then((data) => {
        setMessages(
          (data.messages || []).map((msg) => ({
            role: msg.role,
            text: msg.text,
            sources: msg.sources || [],
            report: msg.report || "",
            rewrittenQuery: msg.rewrittenQuery || "",
          }))
        );
      })
      .catch((err) => setError(err.message))
      .finally(() => setHistoryReady(true));
  }, [threadId]);

  async function onSubmit(e) {
    e.preventDefault();
    const text = question.trim();
    if (!text || loading) {
      return;
    }
    setError("");
    setLoading(true);
    setQuestion("");
    setMessages((prev) => [...prev, { role: "user", text }, emptyAssistant()]);
    try {
      const finalEvent = await askQuestionStream(text, threadId, (event) => {
        if (event.type === "rewrite") {
          setMessages((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            next[next.length - 1] = { ...last, rewrittenQuery: event.query || "" };
            return next;
          });
        } else if (event.type === "token") {
          setMessages((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            next[next.length - 1] = { ...last, text: (last.text || "") + (event.text || "") };
            return next;
          });
        } else if (event.type === "sources") {
          setMessages((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            next[next.length - 1] = { ...last, sources: event.sources || [] };
            return next;
          });
        } else if (event.type === "report") {
          setMessages((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            next[next.length - 1] = { ...last, report: event.report || "" };
            return next;
          });
        } else if (event.type === "done") {
          setMessages((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            next[next.length - 1] = {
              ...last,
              text: event.answer || last.text,
              sources: event.sources || last.sources,
              report: event.report || last.report,
              rewrittenQuery: event.rewritten_query || last.rewrittenQuery,
            };
            return next;
          });
        }
      });
      if (finalEvent && finalEvent.type === "error") {
        throw new Error(finalEvent.error || "生成失败");
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function onNewThread() {
    const id = crypto.randomUUID();
    localStorage.setItem(THREAD_KEY, id);
    window.location.reload();
  }

  return (
    <>
      <header className="header">
        <h1>保险知识库 RAG 智能问答</h1>
        <div className="toolbar">
          <button type="button" className="text-btn" onClick={onNewThread}>
            新对话
          </button>
        </div>
        <p className="sub">多格式文档检索、来源溯源、多轮指代消解、分析报告</p>
      </header>
      <main className="chat">
        {!historyReady && <p className="hint">正在加载对话记录...</p>}
        {historyReady && messages.length === 0 && (
          <p className="hint">例如：雇主责任险的保障范围有哪些？没有签劳动合同的临时工能不能赔？</p>
        )}
        {messages.map((msg, i) => (
          <article key={i} className={`bubble ${msg.role}`}>
            <div className="label">{msg.role === "user" ? "你" : "助手"}</div>
            {msg.role === "assistant" && msg.rewrittenQuery && (
              <div className="rewrite">检索问句：{msg.rewrittenQuery}</div>
            )}
            <div className="text">{msg.text}</div>
            {msg.sources && msg.sources.length > 0 && (
              <ul className="sources">
                {msg.sources.map((src, j) => (
                  <li key={j}>
                    <strong>{src.file_name}</strong>
                    {src.score != null ? ` · ${src.score}` : ""}
                    <div className="preview">{src.preview}</div>
                  </li>
                ))}
              </ul>
            )}
            {msg.role === "assistant" && msg.report && (
              <div className="report">
                <div className="report-title">分析报告</div>
                <div className="report-body">{msg.report}</div>
              </div>
            )}
          </article>
        ))}
        {loading && <p className="hint">正在改写、检索并流式生成...</p>}
        {error && <p className="error">{error}</p>}
      </main>

      <form className="composer" onSubmit={onSubmit}>
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="输入保险产品或理赔问题"
          disabled={loading || !historyReady}
        />
        <button type="submit" disabled={loading || !historyReady || !question.trim()}>
          发送
        </button>
      </form>
    </>
  );
}
