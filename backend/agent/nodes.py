"""
LangGraph Nodes v4 — OPTIMIZATION 2 + 5 + 7
  - Parallel sub-query retrieval (ThreadPoolExecutor)
  - Parallel embedding calls
  - Conditional evaluator (skip trivial responses)
  - Prompt caching (static system prompts first)
  - Full cost tracking
"""
from __future__ import annotations
import json, logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from openai import OpenAI

from backend.config import (
    OPENAI_API_KEY, LLM_LARGE, LLM_SMALL,
    TOP_K, POST_PROCESS_TOP_K,
)
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.database.db import get_all_documents, log_api_usage
from backend.memory.memory_manager import (
    build_memory_context, maybe_summarise, extract_and_store_facts,
)
from backend.agent.state import AgentState
from backend.llm_retry import with_llm_retry
from backend.prompts.prompt_configs import (
    QUERY_OPTIMIZER_SYSTEM,   QUERY_OPTIMIZER_TEMPLATE,
    SOURCE_IDENTIFIER_SYSTEM, SOURCE_IDENTIFIER_TEMPLATE,
    POST_PROCESSOR_SYSTEM,    POST_PROCESSOR_TEMPLATE,
    ANSWER_GENERATOR_SYSTEM,  ANSWER_GENERATOR_TEMPLATE,
    EVALUATOR_SYSTEM,         EVALUATOR_TEMPLATE,
)

logger  = logging.getLogger(__name__)
_client = OpenAI(api_key=OPENAI_API_KEY)

# ── Cost constants (per million tokens) ──────────────────────────
COST_PER_1M = {
    "gpt-4o":       {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-4o-mini":  {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    "embedding":    {"input": 0.02},
}


def _estimate_cost(model: str, prompt_tokens: int, cached_tokens: int,
                   completion_tokens: int) -> float:
    rates = COST_PER_1M.get(model, COST_PER_1M["gpt-4o-mini"])
    uncached = max(0, prompt_tokens - cached_tokens)
    return (
        (uncached       * rates["input"]         / 1_000_000) +
        (cached_tokens  * rates.get("cached_input", rates["input"]) / 1_000_000) +
        (completion_tokens * rates["output"]     / 1_000_000)
    )


# ── LLM helper with usage tracking ───────────────────────────────

class UsageTracker:
    def __init__(self):
        self.llm_calls        = 0
        self.prompt_tokens    = 0
        self.cached_tokens    = 0
        self.completion_tokens = 0
        self.estimated_cost   = 0.0

    def track(self, resp, model: str):
        self.llm_calls += 1
        usage = resp.usage
        pt = usage.prompt_tokens
        ct = getattr(getattr(usage, 'prompt_tokens_details', None), 'cached_tokens', 0)
        ot = usage.completion_tokens
        self.prompt_tokens     += pt
        self.cached_tokens     += ct
        self.completion_tokens += ot
        self.estimated_cost    += _estimate_cost(model, pt, ct, ot)
        if ct > 0:
            logger.info(f"[Cache] {ct}/{pt} prompt tokens from cache ({ct/pt*100:.0f}%)")


@with_llm_retry
def _create_completion(**kwargs):
    return _client.chat.completions.create(**kwargs)


def _llm(messages, model, json_mode=False, max_tokens=1024,
         tracker: UsageTracker | None = None) -> str:
    kwargs = {"model": model, "messages": messages,
              "temperature": 0.0, "max_tokens": max_tokens}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = _create_completion(**kwargs)
    if tracker:
        tracker.track(resp, model)
    return resp.choices[0].message.content or ""


# ─────────────────────────────────────────────────────────────────
# NODE 0 · Memory Loader
# ─────────────────────────────────────────────────────────────────

def memory_loader_node(state: AgentState) -> AgentState:
    memory_ctx = build_memory_context(
        session_id   = state["session_id"],
        query        = state["query"],
        full_history = state.get("chat_history", []),
    )
    state["memory_context"] = memory_ctx
    state.setdefault("_tracker", UsageTracker())
    logger.info(
        f"[Memory] summary={'yes' if memory_ctx['summary'] else 'no'} "
        f"facts={len(memory_ctx['relevant_facts'])}"
    )
    return state


# ─────────────────────────────────────────────────────────────────
# NODE 1 · Query Optimizer (LLM small)
# ─────────────────────────────────────────────────────────────────

def query_optimizer_node(state: AgentState) -> AgentState:
    tracker      = state.get("_tracker", UsageTracker())
    memory_ctx   = state.get("memory_context", {})
    memory_text  = memory_ctx.get("memory_prompt", "None")
    history      = memory_ctx.get("recent_messages", state.get("chat_history", []))
    history_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history[-4:]) or "None"

    prompt = QUERY_OPTIMIZER_TEMPLATE.format(
        memory_context=memory_text or "None",
        history=history_text,
        query=state["query"],
    )
    try:
        result = json.loads(_llm(
            [{"role": "system", "content": QUERY_OPTIMIZER_SYSTEM},
             {"role": "user",   "content": prompt}],
            model=LLM_SMALL, json_mode=True, tracker=tracker,
        ))
        state["refined_query"] = result.get("refined_query", state["query"])
        state["sub_queries"]   = result.get("sub_queries",   [])
        state["query_type"]    = result.get("query_type",    "simple")
    except Exception as e:
        logger.warning(f"[QueryOptimizer] failed: {e}")
        state["refined_query"] = state["query"]
        state["sub_queries"]   = []
        state["query_type"]    = "simple"
    return state


# ─────────────────────────────────────────────────────────────────
# NODE 2 · Source Identifier (LLM small)
# ─────────────────────────────────────────────────────────────────

def source_identifier_node(state: AgentState) -> AgentState:
    tracker = state.get("_tracker", UsageTracker())
    docs    = get_all_documents()
    if len(docs) <= 1:
        state["relevant_doc_ids"] = [state["doc_id"]]
        state["metadata_filters"] = {}
        return state

    docs_summary = "\n".join(
        f"- id={d['id']} | file={d['filename']} | type={d['file_type']} "
        f"| content_type={d.get('content_type','general')} "
        f"| has_tables={d.get('has_tables',False)}"
        for d in docs
    )
    try:
        result = json.loads(_llm(
            [{"role": "system", "content": SOURCE_IDENTIFIER_SYSTEM},
             {"role": "user",   "content": SOURCE_IDENTIFIER_TEMPLATE.format(
                 query=state["refined_query"], documents=docs_summary)}],
            model=LLM_SMALL, json_mode=True, tracker=tracker,
        ))
        state["relevant_doc_ids"] = result.get("relevant_doc_ids", [state["doc_id"]])
        state["metadata_filters"] = result.get("metadata_filters", {})
    except Exception as e:
        logger.warning(f"[SourceIdentifier] failed: {e}")
        state["relevant_doc_ids"] = [state["doc_id"]]
        state["metadata_filters"] = {}
    return state


# ─────────────────────────────────────────────────────────────────
# NODE 3 · Retriever — OPTIMIZATION 2: PARALLEL EXECUTION
# ─────────────────────────────────────────────────────────────────

def _retrieve_one(doc_id: str, query: str, top_k: int,
                  filters: dict | None) -> List[dict]:
    """Single retrieval task — runs in thread pool."""
    return HybridRetriever(doc_id).retrieve(query, top_k, filters)


def retriever_node(state: AgentState) -> AgentState:
    queries  = state["sub_queries"] if state["sub_queries"] else [state["refined_query"]]
    doc_ids  = state.get("relevant_doc_ids") or [state["doc_id"]]
    filters  = state.get("metadata_filters") or {}

    seen:   set[str]   = set()
    chunks: List[dict] = []

    # ── PARALLEL: submit all (doc_id × query) combinations ───────
    tasks = [
        (doc_id, query)
        for doc_id in doc_ids
        for query  in queries
    ]

    with ThreadPoolExecutor(max_workers=min(8, len(tasks))) as pool:
        futures = {
            pool.submit(_retrieve_one, doc_id, q, TOP_K, filters or None): (doc_id, q)
            for doc_id, q in tasks
        }
        for future in as_completed(futures):
            try:
                for chunk in future.result():
                    if chunk["chunk_id"] not in seen:
                        seen.add(chunk["chunk_id"])
                        chunks.append(chunk)
            except Exception as e:
                logger.warning(f"[Retriever] task failed: {e}")

    state["retrieved_chunks"] = chunks
    logger.info(
        f"[Retriever] {len(chunks)} unique chunks "
        f"from {len(doc_ids)} doc(s) × {len(queries)} query(ies) "
        f"[parallel]"
    )
    return state


# ─────────────────────────────────────────────────────────────────
# NODE 4 · Post-Processor (LLM small)
# ─────────────────────────────────────────────────────────────────

def post_processor_node(state: AgentState) -> AgentState:
    tracker = state.get("_tracker", UsageTracker())
    chunks  = state["retrieved_chunks"]
    if not chunks:
        state["processed_chunks"] = []
        return state

    chunk_preview = "\n".join(
        f"[{i}] ({c.get('chunk_type','text')}) {c['text'][:200].strip()}"
        for i, c in enumerate(chunks)
    )
    try:
        result  = json.loads(_llm(
            [{"role": "system", "content": POST_PROCESSOR_SYSTEM},
             {"role": "user",   "content": POST_PROCESSOR_TEMPLATE.format(
                 query=state["refined_query"], chunks=chunk_preview,
                 top_k=POST_PROCESS_TOP_K)}],
            model=LLM_SMALL, json_mode=True, tracker=tracker,
        ))
        indices = result.get("selected_indices", list(range(len(chunks))))
        valid   = [i for i in indices if isinstance(i, int) and 0 <= i < len(chunks)]
        state["processed_chunks"] = [chunks[i] for i in valid[:POST_PROCESS_TOP_K]]
    except Exception as e:
        logger.warning(f"[PostProcessor] failed: {e}")
        state["processed_chunks"] = chunks[:POST_PROCESS_TOP_K]
    return state


# ─────────────────────────────────────────────────────────────────
# NODE 5 · Answer Generator (LLM large)
# ─────────────────────────────────────────────────────────────────

def answer_generator_node(state: AgentState) -> AgentState:
    tracker     = state.get("_tracker", UsageTracker())
    chunks      = state["processed_chunks"] or state["retrieved_chunks"]
    memory_ctx  = state.get("memory_context", {})
    recent_msgs = memory_ctx.get("recent_messages", state.get("chat_history", []))

    if not chunks:
        state["answer"]  = "I could not find relevant information in the document."
        state["sources"] = []
        return state

    context = "\n\n".join(
        f"[Source {i+1} | {c['filename']} · "
        f"{c.get('chunk_type','text')} chunk {c['chunk_index']}]\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    memory_section = ""
    if memory_ctx.get("memory_prompt"):
        memory_section = (
            f"--- Memory Context ---\n{memory_ctx['memory_prompt']}\n--- End Memory ---\n"
        )

    # OPTIMIZATION 3: static system prompt first → prompt caching
    messages = [{"role": "system", "content": ANSWER_GENERATOR_SYSTEM}]
    for msg in recent_msgs:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({
        "role": "user",
        "content": ANSWER_GENERATOR_TEMPLATE.format(
            memory_section=memory_section,
            context=context,
            query=state["refined_query"],
        ),
    })

    answer = _llm(messages, model=LLM_LARGE, max_tokens=1500, tracker=tracker)
    state["answer"]  = answer
    state["sources"] = [
        f"{c['filename']} ({c.get('chunk_type','text')} chunk {c['chunk_index']})"
        for c in chunks
    ]

    # Post-turn memory
    try:
        maybe_summarise(state["session_id"], state.get("chat_history", []))
        extract_and_store_facts(state["session_id"], state["refined_query"], answer)
    except Exception as e:
        logger.warning(f"[Memory] post-turn update failed: {e}")

    return state


# ─────────────────────────────────────────────────────────────────
# NODE 6 · Evaluator — OPTIMIZATION 5: CONDITIONAL
# ─────────────────────────────────────────────────────────────────

SKIP_PHRASES = [
    "not available in the document",
    "could not find relevant",
    "please upload a document",
]


def evaluator_node(state: AgentState) -> AgentState:
    tracker = state.get("_tracker", UsageTracker())
    answer  = state.get("answer", "")

    # ── Skip trivial / empty responses ───────────────────────────
    should_skip = (
        len(answer) < 100
        or any(p in answer.lower() for p in SKIP_PHRASES)
    )

    if should_skip:
        state["evaluation"] = {
            "faithfulness": 0, "answer_relevance": 0,
            "context_relevance": 0,
            "reasoning": "Skipped — trivial or empty response.",
            "skipped": True,
        }
        logger.info("[Evaluator] Skipped — trivial response (saved 1 LLM call)")
        _flush_usage(state, tracker)
        return state

    chunks  = (state["processed_chunks"] or state["retrieved_chunks"])[:3]
    context = "\n\n".join(c["text"] for c in chunks)[:2000]

    try:
        result = json.loads(_llm(
            [{"role": "system", "content": EVALUATOR_SYSTEM},
             {"role": "user",   "content": EVALUATOR_TEMPLATE.format(
                 query=state["refined_query"], context=context, answer=answer)}],
            model=LLM_SMALL, json_mode=True, tracker=tracker,
        ))
        state["evaluation"] = {
            "faithfulness":      int(result.get("faithfulness",      0)),
            "answer_relevance":  int(result.get("answer_relevance",  0)),
            "context_relevance": int(result.get("context_relevance", 0)),
            "reasoning":         result.get("reasoning", ""),
            "skipped":           False,
        }
    except Exception as e:
        logger.warning(f"[Evaluator] failed: {e}")
        state["evaluation"] = {
            "faithfulness": 0, "answer_relevance": 0,
            "context_relevance": 0, "reasoning": "Unavailable.", "skipped": False,
        }

    _flush_usage(state, tracker)
    return state


def _flush_usage(state: AgentState, tracker: UsageTracker) -> None:
    """Persist usage stats to DB after pipeline completes."""
    try:
        import hashlib
        query_hash = hashlib.md5(state["query"].encode()).hexdigest()
        log_api_usage(
            session_id        = state["session_id"],
            query_hash        = query_hash,
            embedding_calls   = 0,   # tracked separately in embedder
            embedding_cached  = 0,
            llm_calls         = tracker.llm_calls,
            prompt_tokens     = tracker.prompt_tokens,
            cached_tokens     = tracker.cached_tokens,
            completion_tokens = tracker.completion_tokens,
            estimated_cost    = tracker.estimated_cost,
            cache_hit         = False,
        )
        logger.info(
            f"[Usage] LLM calls={tracker.llm_calls} "
            f"tokens={tracker.prompt_tokens}p/{tracker.completion_tokens}c "
            f"cached={tracker.cached_tokens} "
            f"cost=${tracker.estimated_cost:.5f}"
        )
    except Exception as e:
        logger.warning(f"[Usage] logging failed: {e}")
