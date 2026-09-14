async function request(url, options = {}) {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || "请求失败");
  }
  return data;
}

export function listMessages(threadId) {
  return request(`/api/chat/messages?thread_id=${encodeURIComponent(threadId)}`);
}

export function askQuestion(question, threadId) {
  return request("/api/chat", {
    method: "POST",
    body: JSON.stringify({ question, thread_id: threadId }),
  });
}

export async function askQuestionStream(question, threadId, onEvent) {
  const res = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, thread_id: threadId }),
  });
  if (!res.ok) {
    let data = {};
    try {
      data = await res.json();
    } catch {
      data = {};
    }
    throw new Error(data.error || "请求失败");
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buf += decoder.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop();
    for (const part of parts) {
      const line = part.split("\n").find((item) => item.startsWith("data: "));
      if (!line) {
        continue;
      }
      const event = JSON.parse(line.slice(6));
      onEvent(event);
      if (event.type === "done" || event.type === "error") {
        return event;
      }
    }
  }
  throw new Error("流式响应意外结束");
}
