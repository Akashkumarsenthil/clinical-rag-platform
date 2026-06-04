"""Serves a single-page chat UI at the root path."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["UI"])

CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Clinical RAG Platform</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #0f172a; --surface: #1e293b; --surface2: #334155;
    --border: #475569; --text: #e2e8f0; --text-dim: #94a3b8;
    --accent: #38bdf8; --accent-dim: #0284c7;
    --green: #34d399; --red: #f87171;
    --user-bg: #1e3a5f; --bot-bg: #1a2332;
    --radius: 12px; --font: 'Inter', -apple-system, system-ui, sans-serif;
  }

  body { font-family: var(--font); background: var(--bg); color: var(--text); height: 100vh; display: flex; flex-direction: column; }

  /* Header */
  .header {
    padding: 16px 24px; background: var(--surface); border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 12px; flex-shrink: 0;
  }
  .header-icon {
    width: 40px; height: 40px; background: linear-gradient(135deg, var(--accent), #818cf8);
    border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 20px;
  }
  .header-text h1 { font-size: 18px; font-weight: 600; }
  .header-text p { font-size: 12px; color: var(--text-dim); margin-top: 2px; }
  .status { margin-left: auto; display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--text-dim); }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); }
  .status-dot.offline { background: var(--red); }

  /* Chat area */
  .chat-container { flex: 1; overflow-y: auto; padding: 20px 24px; display: flex; flex-direction: column; gap: 16px; }

  .message { display: flex; gap: 12px; max-width: 85%; animation: fadeIn 0.3s ease; }
  .message.user { align-self: flex-end; flex-direction: row-reverse; }
  .message.bot { align-self: flex-start; }

  .avatar {
    width: 32px; height: 32px; border-radius: 8px; display: flex; align-items: center;
    justify-content: center; font-size: 14px; flex-shrink: 0; font-weight: 600;
  }
  .message.user .avatar { background: var(--accent-dim); }
  .message.bot .avatar { background: var(--surface2); }

  .bubble { padding: 12px 16px; border-radius: var(--radius); line-height: 1.6; font-size: 14px; }
  .message.user .bubble { background: var(--user-bg); border-bottom-right-radius: 4px; }
  .message.bot .bubble { background: var(--bot-bg); border: 1px solid var(--surface2); border-bottom-left-radius: 4px; }

  .meta { margin-top: 8px; font-size: 11px; color: var(--text-dim); display: flex; gap: 12px; flex-wrap: wrap; }
  .meta span { display: flex; align-items: center; gap: 4px; }

  /* Sources */
  .sources { margin-top: 10px; }
  .sources-toggle {
    background: none; border: 1px solid var(--border); color: var(--accent); padding: 4px 10px;
    border-radius: 6px; font-size: 11px; cursor: pointer; font-family: var(--font);
  }
  .sources-toggle:hover { background: var(--surface2); }
  .sources-list { margin-top: 8px; display: none; }
  .sources-list.open { display: block; }
  .source-item {
    background: var(--surface); border: 1px solid var(--surface2); border-radius: 8px;
    padding: 10px 12px; margin-bottom: 6px; font-size: 12px; line-height: 1.5;
  }
  .source-item .source-header { color: var(--accent); font-weight: 500; margin-bottom: 4px; }
  .source-item .source-text { color: var(--text-dim); max-height: 80px; overflow: hidden; }

  /* Typing indicator */
  .typing { display: flex; gap: 4px; padding: 8px 0; }
  .typing span { width: 8px; height: 8px; background: var(--text-dim); border-radius: 50%; animation: bounce 1.4s infinite; }
  .typing span:nth-child(2) { animation-delay: 0.2s; }
  .typing span:nth-child(3) { animation-delay: 0.4s; }

  /* Input area */
  .input-area {
    padding: 16px 24px; background: var(--surface); border-top: 1px solid var(--border);
    flex-shrink: 0;
  }
  .input-wrap {
    display: flex; gap: 10px; max-width: 900px; margin: 0 auto;
  }
  .input-wrap textarea {
    flex: 1; background: var(--bg); border: 1px solid var(--border); color: var(--text);
    padding: 12px 16px; border-radius: var(--radius); font-family: var(--font); font-size: 14px;
    resize: none; height: 48px; outline: none; transition: border-color 0.2s;
  }
  .input-wrap textarea:focus { border-color: var(--accent); }
  .input-wrap textarea::placeholder { color: var(--text-dim); }
  .send-btn {
    width: 48px; height: 48px; background: var(--accent); border: none; border-radius: var(--radius);
    cursor: pointer; display: flex; align-items: center; justify-content: center; transition: background 0.2s;
  }
  .send-btn:hover { background: var(--accent-dim); }
  .send-btn:disabled { background: var(--surface2); cursor: not-allowed; }
  .send-btn svg { width: 20px; height: 20px; fill: var(--bg); }

  /* Welcome */
  .welcome { text-align: center; padding: 60px 20px; color: var(--text-dim); }
  .welcome h2 { font-size: 22px; color: var(--text); margin-bottom: 8px; }
  .welcome p { font-size: 14px; max-width: 500px; margin: 0 auto 20px; line-height: 1.6; }
  .examples { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; }
  .example-btn {
    background: var(--surface); border: 1px solid var(--border); color: var(--text);
    padding: 8px 14px; border-radius: 8px; font-size: 13px; cursor: pointer;
    font-family: var(--font); transition: all 0.2s;
  }
  .example-btn:hover { border-color: var(--accent); color: var(--accent); }

  @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
  @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }

  .chat-container::-webkit-scrollbar { width: 6px; }
  .chat-container::-webkit-scrollbar-track { background: transparent; }
  .chat-container::-webkit-scrollbar-thumb { background: var(--surface2); border-radius: 3px; }
</style>
</head>
<body>

<div class="header">
  <div class="header-icon">&#x1F3E5;</div>
  <div class="header-text">
    <h1>Clinical RAG Platform</h1>
    <p>Hybrid retrieval + LangGraph agent over medical literature</p>
  </div>
  <div class="status">
    <div class="status-dot" id="statusDot"></div>
    <span id="statusText">Checking...</span>
  </div>
</div>

<div class="chat-container" id="chat">
  <div class="welcome" id="welcome">
    <h2>Ask a clinical question</h2>
    <p>This system uses hybrid dense+sparse retrieval, cross-encoder reranking,
       and a LangGraph agent to answer questions from indexed medical papers.</p>
    <div class="examples">
      <button class="example-btn" onclick="askExample(this)">How are LLMs used in clinical medicine?</button>
      <button class="example-btn" onclick="askExample(this)">What is biomedical named entity recognition?</button>
      <button class="example-btn" onclick="askExample(this)">What benchmarks evaluate medical LLMs?</button>
      <button class="example-btn" onclick="askExample(this)">What are the ethical risks of AI in healthcare?</button>
    </div>
  </div>
</div>

<div class="input-area">
  <div class="input-wrap">
    <textarea id="input" placeholder="Ask a clinical question..." rows="1"
      onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendMessage();}"></textarea>
    <button class="send-btn" id="sendBtn" onclick="sendMessage()">
      <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
    </button>
  </div>
</div>

<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const sendBtn = document.getElementById('sendBtn');
const welcome = document.getElementById('welcome');
let sessionId = 'session_' + Date.now();

async function checkHealth() {
  try {
    const r = await fetch('/health');
    const d = await r.json();
    document.getElementById('statusDot').className = 'status-dot';
    document.getElementById('statusText').textContent =
      `v${d.version} | Qdrant: ${d.services.qdrant} | Redis: ${d.services.redis}`;
  } catch {
    document.getElementById('statusDot').className = 'status-dot offline';
    document.getElementById('statusText').textContent = 'Offline';
  }
}
checkHealth();
setInterval(checkHealth, 30000);

function askExample(btn) { input.value = btn.textContent; sendMessage(); }

function addMessage(role, content, extra) {
  if (welcome) welcome.style.display = 'none';
  const msg = document.createElement('div');
  msg.className = `message ${role}`;
  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? 'You' : 'AI';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = content;

  if (extra) {
    const meta = document.createElement('div');
    meta.className = 'meta';
    if (extra.confidence !== undefined)
      meta.innerHTML += `<span>Confidence: ${(extra.confidence * 100).toFixed(0)}%</span>`;
    if (extra.latency_ms !== undefined)
      meta.innerHTML += `<span>Latency: ${(extra.latency_ms / 1000).toFixed(1)}s</span>`;
    if (extra.sources && extra.sources.length > 0) {
      meta.innerHTML += `<span>${extra.sources.length} sources</span>`;
    }
    bubble.appendChild(meta);

    if (extra.sources && extra.sources.length > 0) {
      const srcDiv = document.createElement('div');
      srcDiv.className = 'sources';
      const toggle = document.createElement('button');
      toggle.className = 'sources-toggle';
      toggle.textContent = 'Show sources';
      const list = document.createElement('div');
      list.className = 'sources-list';
      toggle.onclick = () => {
        list.classList.toggle('open');
        toggle.textContent = list.classList.contains('open') ? 'Hide sources' : 'Show sources';
      };
      extra.sources.forEach((s, i) => {
        const item = document.createElement('div');
        item.className = 'source-item';
        const src = s.metadata.source ? s.metadata.source.split('/').pop() : 'unknown';
        const page = s.metadata.page_number || '?';
        item.innerHTML = `<div class="source-header">[${i+1}] ${src} — p.${page} (score: ${s.score.toFixed(2)})</div>
          <div class="source-text">${s.content.substring(0, 200)}...</div>`;
        list.appendChild(item);
      });
      srcDiv.appendChild(toggle);
      srcDiv.appendChild(list);
      bubble.appendChild(srcDiv);
    }
  }

  msg.appendChild(avatar);
  msg.appendChild(bubble);
  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
  return msg;
}

function addTyping() {
  const msg = document.createElement('div');
  msg.className = 'message bot';
  msg.id = 'typing';
  msg.innerHTML = '<div class="avatar">AI</div><div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>';
  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
}

function removeTyping() {
  const t = document.getElementById('typing');
  if (t) t.remove();
}

async function sendMessage() {
  const q = input.value.trim();
  if (!q) return;
  input.value = '';
  sendBtn.disabled = true;
  addMessage('user', q);
  addTyping();

  try {
    const r = await fetch('/api/v1/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, top_k: 5, session_id: sessionId }),
    });
    removeTyping();
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    addMessage('bot', d.answer.replace(/\\n/g, '<br>'), d);
  } catch (e) {
    removeTyping();
    addMessage('bot', `<span style="color:var(--red)">Error: ${e.message}</span>`);
  }
  sendBtn.disabled = false;
  input.focus();
}

input.focus();
</script>
</body>
</html>"""


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def chat_ui() -> str:
    """Serve the chat UI at the root path."""
    return CHAT_HTML
