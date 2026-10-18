"""
Streaming Gradio Frontend — Agentic RAG v3
Streams tokens in real-time with beautiful pipeline status display.
"""
from __future__ import annotations
import os, json, time
import requests
import gradio as gr

API_URL = os.getenv("API_URL", "http://localhost:8000")
PASS_THRESHOLD = 3.5

# ── CSS ───────────────────────────────────────────────────────────
CSS = """
/* ── Base ─────────────────────────────────────────────────────── */
body, .gradio-container { background: #0f0f13 !important; }
.gradio-container { max-width: 1400px !important; margin: 0 auto !important; }

/* ── Header ───────────────────────────────────────────────────── */
.app-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(139,92,246,0.15);
}
.app-header h1 { color: #fff; font-size: 2rem; margin: 0 0 8px 0; }
.app-header p  { color: #a78bfa; margin: 0; font-size: 0.95rem; }

/* ── Upload panel ─────────────────────────────────────────────── */
.upload-panel {
    background: #13131f;
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 14px;
    padding: 20px;
}
.upload-panel label { color: #c4b5fd !important; font-weight: 600 !important; }

/* ── Chat container ───────────────────────────────────────────── */
.chat-wrap { background: #13131f !important; border-radius: 14px !important;
             border: 1px solid rgba(139,92,246,0.2) !important; }

/* ── Pipeline tracker ─────────────────────────────────────────── */
.pipeline-box {
    background: #0d0d18;
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 12px;
    padding: 14px 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    min-height: 160px;
    color: #94a3b8;
    line-height: 1.8;
}

/* ── Input row ────────────────────────────────────────────────── */
.msg-input textarea {
    background: #1a1a2e !important;
    border: 1px solid rgba(139,92,246,0.3) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 0.95rem !important;
    resize: none !important;
}
.msg-input textarea:focus {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,0.2) !important;
}

/* ── Send button ──────────────────────────────────────────────── */
.send-btn {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    transition: all 0.2s !important;
    min-height: 56px !important;
}
.send-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 20px rgba(124,58,237,0.4) !important;
}

/* ── Upload button ────────────────────────────────────────────── */
.upload-btn button {
    background: linear-gradient(135deg, #059669, #0d9488) !important;
    border-radius: 10px !important;
    color: white !important;
    font-weight: 600 !important;
    width: 100% !important;
}

/* ── Status card ──────────────────────────────────────────────── */
.status-card {
    background: #0d1117;
    border-left: 3px solid #8b5cf6;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    color: #c4b5fd;
    font-size: 0.85rem;
    margin-top: 8px;
}

/* ── Tabs ─────────────────────────────────────────────────────── */
.tab-nav { border-bottom: 2px solid rgba(139,92,246,0.3) !important; }
.tab-nav button { color: #94a3b8 !important; font-weight: 500 !important; }
.tab-nav button.selected { color: #a78bfa !important;
                            border-bottom: 2px solid #8b5cf6 !important; }

/* ── Message bubbles ──────────────────────────────────────────── */
.message.user     { background: linear-gradient(135deg,#1e1b4b,#312e81) !important;
                    border-radius: 16px 16px 4px 16px !important; }
.message.bot      { background: #1a1a2e !important;
                    border-radius: 16px 16px 16px 4px !important;
                    border-left: 3px solid #7c3aed !important; }

/* ── Eval badge ───────────────────────────────────────────────── */
.eval-pass { color: #34d399; font-weight: 700; }
.eval-warn { color: #fbbf24; font-weight: 700; }

/* ── Scrollbar ────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f0f13; }
::-webkit-scrollbar-thumb { background: #4c1d95; border-radius: 3px; }
"""

# ── Node display config ───────────────────────────────────────────
NODE_ICONS = {
    "input_guard":       ("🛡️", "Input Guard",       "#ef4444"),
    "retrieval_guard":   ("🔍", "Retrieval Guard",   "#f97316"),
    "output_guard":      ("🚫", "Output Guard",      "#eab308"),
    "memory_loader":     ("🧠", "Memory Loader",    "#a78bfa"),
    "query_optimizer":   ("⚡", "Query Optimizer",   "#60a5fa"),
    "source_identifier": ("🔍", "Source Identifier", "#34d399"),
    "retriever":         ("📡", "Hybrid Retriever",  "#f59e0b"),
    "post_processor":    ("🎯", "Post Processor",    "#f472b6"),
    "answer_generator":  ("✍️",  "Answer Generator",  "#a3e635"),
    "evaluator":         ("📊", "Evaluator",         "#fb923c"),
}
NODE_ORDER = [
    "input_guard", "memory_loader", "query_optimizer",
    "source_identifier", "retriever", "retrieval_guard",
    "post_processor", "answer_generator", "output_guard",
]


def _star(n):
    n = int(n)
    return "⭐" * n + "☆" * (5 - n)


def _format_eval(ev: dict) -> str:
    if not ev or not ev.get("faithfulness"):
        return ""
    f  = ev["faithfulness"]
    ar = ev["answer_relevance"]
    cr = ev["context_relevance"]
    avg = round((f + ar + cr) / 3, 1)
    ok  = avg >= PASS_THRESHOLD
    badge = "✅ PASS" if ok else "⚠️ NEEDS REVIEW"
    return (
        f"\n\n---\n"
        f"**📊 Quality Evaluation**\n\n"
        f"| Metric | Score |\n|---|---|\n"
        f"| Faithfulness | {_star(f)} `{f}/5` |\n"
        f"| Answer Relevance | {_star(ar)} `{ar}/5` |\n"
        f"| Context Relevance | {_star(cr)} `{cr}/5` |\n"
        f"| **Overall** | `{avg}/5` **{badge}** |\n"
        + (f"\n> *{ev['reasoning']}*" if ev.get("reasoning") else "")
    )


def _format_sources(sources: list) -> str:
    if not sources:
        return ""
    unique = list(dict.fromkeys(sources))[:5]
    lines  = "\n".join(f"  `{s}`" for s in unique)
    return f"\n\n📎 **Sources**\n{lines}"


def _build_pipeline_html(completed: list[str], active: str | None,
                          statuses: dict[str, str]) -> str:
    lines = []
    for node in NODE_ORDER:
        icon, label, color = NODE_ICONS[node]
        if node in completed:
            lines.append(f'<span style="color:#34d399">✓ {icon} {label}</span>')
        elif node == active:
            status = statuses.get(node, "Running...")
            lines.append(
                f'<span style="color:{color};animation:pulse 1s infinite">'
                f'▶ {icon} {label} — <em>{status}</em></span>'
            )
        else:
            lines.append(f'<span style="color:#374151">○ {icon} {label}</span>')
    return "<br>".join(lines)


# ── API helpers ───────────────────────────────────────────────────

def _post(path, **kwargs):
    r = requests.post(f"{API_URL}{path}", timeout=180, **kwargs)
    r.raise_for_status()
    return r.json()

def _get(path, **kwargs):
    r = requests.get(f"{API_URL}{path}", timeout=30, **kwargs)
    r.raise_for_status()
    return r.json()


# ── Upload handler ────────────────────────────────────────────────

def handle_upload(file, state):
    if file is None:
        return state, "⚠️ Please select a file.", gr.update(), gr.update()
    filename = os.path.basename(file.name)
    try:
        with open(file.name, "rb") as f:
            data = _post("/upload", files={"file": (filename, f)})
        doc_id     = data["doc_id"]
        session_id = _post("/session", json={"doc_id": doc_id})["session_id"]
        new_state  = {"doc_id": doc_id, "session_id": session_id}
        entities   = ", ".join(data.get("key_entities", [])[:5]) or "—"
        status = (
            f"<div class='status-card'>"
            f"✅ <strong>{data['filename']}</strong> ready<br>"
            f"🗂 Chunks: <strong>{data['chunk_count']}</strong> &nbsp;|&nbsp; "
            f"📋 Type: <code>{data.get('content_type','general')}</code> &nbsp;|&nbsp; "
            f"{'📊 Tables detected' if data.get('has_tables') else '📄 No tables'}<br>"
            f"🔑 Entities: {entities}"
            f"</div>"
        )
        return new_state, status, gr.update(visible=True), gr.update(visible=True)
    except Exception as e:
        return state, f"❌ Upload failed: {e}", gr.update(), gr.update()


# ── Streaming chat handler ────────────────────────────────────────

def handle_chat_stream(message, history, state):
    if not message.strip():
        yield history, state, "", "<div class='pipeline-box'>Waiting for input...</div>"
        return
    if not state.get("doc_id"):
        history = history + [[message, "⚠️ Please upload a document first."]]
        yield history, state, "", "<div class='pipeline-box'>No document loaded.</div>"
        return

    # Add user message immediately
    history = history + [[message, ""]]
    yield history, state, "", "<div class='pipeline-box'>Starting pipeline...</div>"

    completed_nodes: list[str] = []
    active_node:     str | None = None
    node_statuses:   dict       = {}

    answer_text  = ""
    sources      = []
    eval_data    = {}
    query_type   = "simple"
    refined      = ""
    filters      = {}
    extra_info   = {}

    def _pipeline_html():
        return (
            "<div class='pipeline-box'>"
            + _build_pipeline_html(completed_nodes, active_node, node_statuses)
            + "</div>"
        )

    try:
        with requests.post(
            f"{API_URL}/chat/stream",
            json={"query": message, "session_id": state["session_id"], "doc_id": state["doc_id"]},
            stream=True,
            timeout=180,
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                except Exception:
                    continue

                etype = event.get("type")

                # ── Node status ────────────────────────────────────
                if etype == "node":
                    node   = event["node"]
                    status = event.get("status", "Running...")
                    if active_node and active_node != node:
                        completed_nodes.append(active_node)
                    active_node          = node
                    node_statuses[node]  = status
                    yield history, state, "", _pipeline_html()

                # ── Query info ─────────────────────────────────────
                elif etype == "query_info":
                    query_type = event.get("query_type", "simple")
                    refined    = event.get("refined", "")
                    subs       = event.get("sub_queries", [])
                    extra_info["query_type"] = query_type
                    extra_info["refined"]    = refined
                    extra_info["sub_queries"] = subs

                # ── Source info ────────────────────────────────────
                elif etype == "source_info":
                    filters = event.get("filters", {})
                    extra_info["filters"] = filters

                # ── Retrieval info ─────────────────────────────────
                elif etype == "retrieval_info":
                    count = event.get("count", 0)
                    node_statuses["retriever"] = f"Found {count} chunks"
                    yield history, state, "", _pipeline_html()

                # ── Safety block ──────────────────────────────────
                elif etype == "safety_block":
                    layer  = event.get("layer",  "Safety")
                    reason = event.get("reason", "")
                    blocked_html = (
                        "<div class='pipeline-box'>"
                        f"<span style='color:#ef4444;font-weight:bold'>"
                        f"🚫 BLOCKED by {layer}</span><br>"
                        f"<span style='color:#94a3b8;font-size:0.8rem'>"
                        f"Reason: {reason}</span>"
                        "</div>"
                    )
                    yield history, state, "", blocked_html

                # ── Safety pass ────────────────────────────────────
                elif etype == "safety_pass":
                    layer = event.get("layer", "")
                    node_statuses[layer] = "✅ Passed"

                # ── Answer replace (output guard blocked) ──────────
                elif etype == "answer_replace":
                    safe_answer        = event.get("answer", "")
                    history[-1][1]     = safe_answer
                    yield history, state, "", _pipeline_html()

                # ── Cache hit ─────────────────────────────────────
                elif etype == "cache_hit":
                    node_statuses["cache"] = "⚡ Cache hit!"
                    yield history, state, "", (
                        "<div class='pipeline-box'>"
                        "<span style='color:#34d399;font-weight:bold'>"
                        "⚡ Response served from cache — 0 API calls!</span>"
                        "</div>"
                    )

                # ── Token — core streaming ─────────────────────────
                elif etype == "token":
                    token       = event.get("token", "")
                    answer_text += token
                    history[-1][1] = answer_text
                    yield history, state, "", _pipeline_html()

                # ── Sources ────────────────────────────────────────
                elif etype == "sources":
                    sources = event.get("sources", [])

                # ── Evaluation ─────────────────────────────────────
                elif etype == "evaluation":
                    eval_data = event.get("data", {})
                    if active_node:
                        completed_nodes.append(active_node)
                    active_node = None

                    # Append eval + sources + metadata to answer
                    suffix = ""
                    suffix += _format_eval(eval_data)
                    suffix += _format_sources(sources)

                    # Query type badge
                    qbadge = {
                        "simple":          "🔵 Simple",
                        "complex":         "🟣 Complex",
                        "conversational":  "🟢 Conversational",
                    }.get(query_type, query_type)
                    suffix += f"\n\n*Query: {qbadge}*"

                    if refined and refined != message:
                        suffix += f"  \n*Refined: \"{refined[:80]}...\"*"
                    if filters:
                        active_f = {k: v for k, v in filters.items() if v}
                        if active_f:
                            suffix += f"  \n*Filters: `{active_f}`*"

                    subs = extra_info.get("sub_queries", [])
                    if subs:
                        suffix += f"  \n*Sub-queries: {len(subs)} executed*"

                    history[-1][1] = answer_text + suffix

                    final_pipeline = (
                        "<div class='pipeline-box'>"
                        + _build_pipeline_html(completed_nodes, None, node_statuses)
                        + "<br><span style='color:#34d399;font-weight:bold'>"
                        + "✅ Pipeline complete</span>"
                        + "</div>"
                    )
                    yield history, state, "", final_pipeline

                # ── Usage / cost ──────────────────────────────────
                elif etype == "usage":
                    calls  = event.get("llm_calls", 0)
                    cost   = event.get("estimated_cost", 0)
                    cached = event.get("cached_tokens", 0)
                    ptok   = event.get("prompt_tokens", 0)
                    cache_pct = round(cached/max(ptok,1)*100)
                    history[-1][1] += (
                        f"\n\n<sub>💰 Cost: `${cost:.5f}` · "
                        f"LLM calls: `{calls}` · "
                        f"Prompt cache: `{cache_pct}%`</sub>"
                    )
                    yield history, state, "", _pipeline_html()

                # ── Done ───────────────────────────────────────────
                elif etype == "done":
                    break

                # ── Error ──────────────────────────────────────────
                elif etype == "error":
                    history[-1][1] = f"❌ Error: {event.get('message', 'Unknown error')}"
                    yield history, state, "", "<div class='pipeline-box'>❌ Pipeline error.</div>"
                    return

    except Exception as e:
        history[-1][1] = f"❌ Connection error: {e}"
        yield history, state, "", "<div class='pipeline-box'>❌ Connection error.</div>"


def handle_reset(state):
    return {}, [], "<div class='status-card'>Session cleared. Upload a new document.</div>", \
           gr.update(visible=False), gr.update(visible=False), \
           "<div class='pipeline-box'>Waiting for document...</div>"


def load_doc_table():
    try:
        docs = _get("/documents")
        if not docs:
            return []
        return [
            [d["filename"], d["file_type"],
             d["chunk_count"],
             "✅ Yes" if d.get("has_tables") else "❌ No",
             d.get("content_type", "general"),
             d["id"][:8] + "...",
             str(d["created_at"])[:16]]
            for d in docs
        ]
    except Exception:
        return []


def add_golden(state, question, expected):
    if not state.get("doc_id"):
        return "⚠️ Upload a document first.", question, expected
    if not question.strip() or not expected.strip():
        return "⚠️ Fill in both fields.", question, expected
    try:
        _post("/golden", json={"doc_id": state["doc_id"],
                               "question": question, "expected": expected})
        return "✅ Golden sample saved!", "", ""
    except Exception as e:
        return f"❌ Error: {e}", question, expected


def load_golden(state):
    if not state.get("doc_id"):
        return []
    try:
        data = _get(f"/golden/{state['doc_id']}")
        return [[d["question"][:80], d["expected"][:80], str(d["created_at"])[:16]] for d in data]
    except Exception:
        return []


def load_stats():
    try:
        usage = _get("/stats/usage")
        cache = _get("/stats/cache")
        emb_hits  = cache.get("embedding_cache", {}).get("total_hits", 0) or 0
        emb_total = cache.get("embedding_cache", {}).get("total_entries", 0) or 0
        resp_hits = cache.get("response_cache", {}).get("total_hits", 0) or 0
        resp_ent  = cache.get("response_cache", {}).get("total_entries", 0) or 0
        total_queries  = usage.get("total_queries", 0) or 0
        total_cost     = usage.get("total_cost", 0) or 0
        avg_cost       = usage.get("avg_cost_per_query", 0) or 0
        cached_tok     = usage.get("total_cached_tokens", 0) or 0
        prompt_tok     = usage.get("total_prompt_tokens", 0) or 0
        cache_hit_rate = usage.get("response_cache_hits", 0) or 0
        cache_pct      = round(cached_tok / max(prompt_tok, 1) * 100)

        html = (
            f"<div style='background:#0d0d18;border:1px solid #4c1d95;"
            f"border-radius:12px;padding:20px;color:#e2e8f0;font-family:monospace'>"
            f"<h3 style='color:#a78bfa;margin:0 0 16px'>📊 API Usage & Cost Stats</h3>"
            f"<table style='width:100%;border-collapse:collapse'>"
            f"<tr><td style='color:#94a3b8;padding:6px'>Total Queries</td>"
            f"<td style='color:#fff;font-weight:bold'>{total_queries}</td></tr>"
            f"<tr><td style='color:#94a3b8;padding:6px'>Total Cost</td>"
            f"<td style='color:#34d399;font-weight:bold'>${total_cost:.4f}</td></tr>"
            f"<tr><td style='color:#94a3b8;padding:6px'>Avg Cost / Query</td>"
            f"<td style='color:#34d399'>${avg_cost:.6f}</td></tr>"
            f"<tr><td style='color:#94a3b8;padding:6px'>Prompt Cache Rate</td>"
            f"<td style='color:#60a5fa'>{cache_pct}% of tokens from cache</td></tr>"
            f"<tr><td style='color:#94a3b8;padding:6px'>Response Cache Hits</td>"
            f"<td style='color:#60a5fa'>{cache_hit_rate} queries served free</td></tr>"
            f"<tr><td style='color:#94a3b8;padding:6px'>Embedding Cache</td>"
            f"<td style='color:#f59e0b'>{emb_hits} hits / {emb_total} entries</td></tr>"
            f"<tr><td style='color:#94a3b8;padding:6px'>Response Cache</td>"
            f"<td style='color:#f59e0b'>{resp_hits} hits / {resp_ent} entries</td></tr>"
            f"</table></div>"
        )
        return html
    except Exception as e:
        return f"<div style='color:#f87171'>❌ Could not load stats: {e}</div>"


def run_eval(state):
    if not state.get("doc_id"):
        return "⚠️ Upload a document first.", gr.update()
    try:
        result = _post(f"/batch-eval/{state['doc_id']}")
        if "error" in result:
            return f"⚠️ {result['error']}", gr.update()

        ok  = result["pass_rate"] >= 0.8
        clr = "#34d399" if ok else "#fbbf24"
        summary = (
            f"<div style='background:#0d0d18;border:1px solid {clr};"
            f"border-radius:12px;padding:20px;color:#e2e8f0'>"
            f"<h3 style='color:{clr};margin:0 0 12px'>📊 Batch Evaluation Results</h3>"
            f"<table style='width:100%;border-collapse:collapse'>"
            f"<tr><td style='padding:6px;color:#94a3b8'>Total Samples</td>"
            f"<td style='color:#fff;font-weight:bold'>{result['total']}</td></tr>"
            f"<tr><td style='padding:6px;color:#94a3b8'>Passed</td>"
            f"<td style='color:{clr};font-weight:bold'>{result['passed']}</td></tr>"
            f"<tr><td style='padding:6px;color:#94a3b8'>Pass Rate</td>"
            f"<td style='color:{clr};font-weight:bold'>{result['pass_rate']*100:.0f}%</td></tr>"
            f"<tr><td style='padding:6px;color:#94a3b8'>Avg Faithfulness</td>"
            f"<td style='color:#a78bfa'>{result['avg_faithfulness']}/5</td></tr>"
            f"<tr><td style='padding:6px;color:#94a3b8'>Avg Answer Relevance</td>"
            f"<td style='color:#a78bfa'>{result['avg_answer_rel']}/5</td></tr>"
            f"<tr><td style='padding:6px;color:#94a3b8'>Avg Context Relevance</td>"
            f"<td style='color:#a78bfa'>{result['avg_context_rel']}/5</td></tr>"
            f"<tr><td style='padding:6px;color:#94a3b8'>Overall Avg</td>"
            f"<td style='color:#fff;font-weight:bold'>{result['avg_overall']}/5</td></tr>"
            f"</table></div>"
        )
        rows = [
            [d["question"][:55],
             f"{_star(d['faithfulness'])} {d['faithfulness']}/5",
             f"{_star(d['answer_relevance'])} {d['answer_relevance']}/5",
             f"{_star(d['context_relevance'])} {d['context_relevance']}/5",
             "✅ PASS" if d["passed"] else "❌ FAIL"]
            for d in result.get("details", [])
        ]
        return summary, gr.update(value=rows)
    except Exception as e:
        return f"❌ Eval failed: {e}", gr.update()


# ── UI ────────────────────────────────────────────────────────────

HEADER = """
<div class="app-header">
  <h1>🤖 Agentic RAG Chatbot</h1>
  <p>
    Upload any <strong>PDF · DOCX · TXT · Markdown</strong> — get intelligent, streaming answers with full pipeline visibility.<br>
    <strong>GPT-4o</strong> (generation) · <strong>GPT-4o-mini</strong> (optimizer / identifier / post-processor) ·
    <strong>pgvector + BM25</strong> · <strong>3-Tier Memory</strong> · <strong>LangGraph 7-node pipeline</strong> · <strong>LLM-as-Judge</strong>
  </p>
</div>
"""

with gr.Blocks(
    title="Agentic RAG",
    css=CSS,
    theme=gr.themes.Base(
        primary_hue="violet",
        secondary_hue="indigo",
        neutral_hue="slate",
        font=["Inter", "ui-sans-serif", "system-ui"],
    ),
) as demo:

    state = gr.State({})
    gr.HTML(HEADER)

    with gr.Tabs(elem_classes=["tab-nav"]):

        # ── TAB 1: Chat ───────────────────────────────────────────
        with gr.Tab("💬 Chat"):
            with gr.Row(equal_height=False):

                # Left sidebar
                with gr.Column(scale=1, min_width=300, elem_classes=["upload-panel"]):
                    gr.Markdown("### 📄 Document")
                    file_in = gr.File(
                        label="Drop PDF · DOCX · TXT · MD here",
                        file_types=[".pdf",".docx",".doc",".txt",".md",".markdown"],
                    )
                    upload_btn = gr.Button("⬆️ Upload & Index", elem_classes=["upload-btn"])
                    status_html = gr.HTML(
                        "<div class='status-card'>No document loaded.</div>"
                    )
                    reset_btn = gr.Button("🔄 New Session", variant="secondary")

                    gr.Markdown("### 🔄 Live Pipeline")
                    pipeline_html = gr.HTML(
                        "<div class='pipeline-box'>Waiting for document...</div>"
                    )

                # Main chat area
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(
                        label="",
                        height=540,
                        visible=False,
                        show_copy_button=True,
                        bubble_full_width=False,
                        elem_classes=["chat-wrap"],
                        avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=rag"),
                        render_markdown=True,
                    )
                    with gr.Row(visible=False, elem_classes=["input-row"]) as input_row:
                        msg_box = gr.Textbox(
                            placeholder="Ask anything about your document…  (Shift+Enter for new line)",
                            lines=2,
                            max_lines=4,
                            label="",
                            scale=5,
                            elem_classes=["msg-input"],
                        )
                        send_btn = gr.Button("Send ➤", scale=1, elem_classes=["send-btn"])

            # Wire upload
            upload_btn.click(
                handle_upload,
                [file_in, state],
                [state, status_html, chatbot, input_row],
            )

            # Wire streaming chat
            send_btn.click(
                handle_chat_stream,
                [msg_box, chatbot, state],
                [chatbot, state, msg_box, pipeline_html],
            )
            msg_box.submit(
                handle_chat_stream,
                [msg_box, chatbot, state],
                [chatbot, state, msg_box, pipeline_html],
            )

            # Wire reset
            reset_btn.click(
                handle_reset,
                [state],
                [state, chatbot, status_html, chatbot, input_row, pipeline_html],
            )

        # ── TAB 2: Document Library ───────────────────────────────
        with gr.Tab("📚 Library"):
            gr.Markdown("### All Indexed Documents")
            refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
            doc_table   = gr.Dataframe(
                headers=["Filename","Type","Chunks","Tables","Content Type","ID","Indexed At"],
                datatype=["str","str","number","str","str","str","str"],
                interactive=False,
            )
            refresh_btn.click(load_doc_table, outputs=doc_table)
            demo.load(load_doc_table, outputs=doc_table)

        # ── TAB 3: Golden Test Data ───────────────────────────────
        with gr.Tab("🧪 Golden Data"):
            gr.Markdown("### Add Golden Q&A Samples for Evaluation")
            with gr.Row():
                q_in = gr.Textbox(label="Question", lines=2,
                                  placeholder="What was total revenue in 2023?")
                e_in = gr.Textbox(label="Expected Answer", lines=2,
                                  placeholder="$383.3 billion")
            add_btn  = gr.Button("➕ Add Sample", variant="primary")
            g_status = gr.Markdown("")
            gr.Markdown("### Saved Samples")
            load_g   = gr.Button("🔄 Load")
            g_table  = gr.Dataframe(
                headers=["Question","Expected Answer","Added At"],
                datatype=["str","str","str"],
                interactive=False,
            )
            add_btn.click(add_golden, [state, q_in, e_in], [g_status, q_in, e_in])
            load_g.click(load_golden, [state], g_table)

        # ── TAB 5: Cost & Cache Stats ─────────────────────────────
        with gr.Tab("💰 Stats"):
            gr.Markdown("### API Usage, Cost & Cache Performance")
            stats_refresh = gr.Button("🔄 Refresh Stats", variant="secondary")
            stats_html    = gr.HTML("<div style='color:#94a3b8'>Click refresh to load stats.</div>")
            stats_refresh.click(load_stats, outputs=stats_html)

        # ── TAB 4: Batch Evaluation ───────────────────────────────
        with gr.Tab("📊 Evaluation"):
            gr.Markdown(
                "### Post-prod Batch Evaluation\n"
                "Runs the full pipeline against all golden samples and scores with LLM-as-Judge."
            )
            run_btn  = gr.Button("▶️ Run Batch Evaluation", variant="primary", size="lg")
            ev_html  = gr.HTML("")
            ev_table = gr.Dataframe(
                headers=["Question","Faithfulness","Answer Rel.","Context Rel.","Result"],
                datatype=["str","str","str","str","str"],
                interactive=False,
            )
            run_btn.click(run_eval, [state], [ev_html, ev_table])


if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        favicon_path=None,
    )
