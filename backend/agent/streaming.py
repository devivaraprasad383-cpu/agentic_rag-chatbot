"""
Streaming Pipeline v5 — safety-aware SSE streaming.
Emits safety events so UI shows which layer blocked + why.
"""
from __future__ import annotations
import json, hashlib, logging
from typing import Generator

from openai import OpenAI

from backend.config import OPENAI_API_KEY, LLM_LARGE
from backend.database.db import (
    get_chat_history, save_message,
    get_cached_response, set_cached_response,
    log_api_usage,
)
from backend.memory.memory_manager import (
    build_memory_context, maybe_summarise, extract_and_store_facts,
)
from backend.agent.nodes import (
    query_optimizer_node, source_identifier_node,
    retriever_node, post_processor_node,
    answer_generator_node, UsageTracker, _estimate_cost,
)
from backend.safety.safety_nodes import (
    input_guard_node, retrieval_guard_node,
    output_guard_node, is_blocked,
)
from backend.prompts.prompt_configs import (
    ANSWER_GENERATOR_SYSTEM, ANSWER_GENERATOR_TEMPLATE,
)
from backend.llm_retry import with_llm_retry

logger  = logging.getLogger(__name__)
_client = OpenAI(api_key=OPENAI_API_KEY)


@with_llm_retry
def _open_stream(**kwargs):
    """Retries only the call that OPENS the stream. Once a token has
    already been yielded to the client we can't transparently retry
    a dropped stream without either re-sending duplicate tokens or
    silently restarting the answer, so a mid-stream failure still
    surfaces as an `error` SSE event — this only protects against the
    common case of a transient failure before any output exists."""
    return _client.chat.completions.create(**kwargs)

SAFETY_LAYER_LABELS = {
    "input_guard":     "🛡️ Input Guard (Layer 1)",
    "retrieval_guard": "🔍 Retrieval Guard (Layer 2)",
    "output_guard":    "🚫 Output Guard (Layer 3)",
}


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _init_state(query, session_id, doc_id, history, tracker):
    return {
        "query": query, "session_id": session_id, "doc_id": doc_id,
        "chat_history": history, "memory_context": {},
        "refined_query": "", "sub_queries": [], "query_type": "",
        "relevant_doc_ids": [], "metadata_filters": {},
        "retrieved_chunks": [], "processed_chunks": [],
        "answer": "", "sources": [], "evaluation": {},
        "safety_blocked": False, "safety_reason": "", "safety_layer": "",
        "_tracker": tracker,
    }


def stream_rag_pipeline(
    query: str, session_id: str, doc_id: str,
) -> Generator[str, None, None]:
    try:
        # ── Response cache check (before any safety processing) ───
        query_hash = hashlib.md5(f"{query.strip().lower()}{doc_id}".encode()).hexdigest()
        cached     = get_cached_response(query_hash, doc_id)

        if cached:
            logger.info(f"[ResponseCache] HIT")
            yield _sse({"type": "cache_hit", "message": "⚡ Served from cache"})
            for word in cached["answer"].split(" "):
                yield _sse({"type": "token", "token": word + " "})
            sources    = cached["sources"] if isinstance(cached["sources"], list) else json.loads(cached["sources"])
            evaluation = cached["evaluation"] if isinstance(cached["evaluation"], dict) else json.loads(cached["evaluation"])
            yield _sse({"type": "sources",    "sources":    sources})
            yield _sse({"type": "evaluation", "data":       evaluation})
            yield _sse({"type": "done"})
            log_api_usage(session_id, query_hash, 0, 0, 0, 0, 0, 0, 0.0, True)
            return

        history = get_chat_history(session_id, limit=20)
        tracker = UsageTracker()
        state   = _init_state(query, session_id, doc_id, history, tracker)

        # ──────────────────────────────────────────────────────────
        # LAYER 1: Input Guard
        # ──────────────────────────────────────────────────────────
        yield _sse({"type": "node", "node": "input_guard",
                    "status": "Scanning query for safety..."})
        state = input_guard_node(state)
        if is_blocked(state):
            yield _sse({
                "type":   "safety_block",
                "layer":  SAFETY_LAYER_LABELS["input_guard"],
                "reason": state["safety_reason"],
            })
            yield _sse({"type": "token", "token": state["answer"]})
            yield _sse({"type": "done"})
            return
        yield _sse({"type": "safety_pass", "layer": "input_guard"})

        # ── Memory ────────────────────────────────────────────────
        yield _sse({"type": "node", "node": "memory_loader",
                    "status": "Loading memory context..."})
        memory_ctx = build_memory_context(session_id, query, history)
        state["memory_context"] = memory_ctx

        # ── Query Optimizer ───────────────────────────────────────
        yield _sse({"type": "node", "node": "query_optimizer",
                    "status": "Optimizing query..."})
        state = query_optimizer_node(state)
        yield _sse({"type": "query_info", "refined": state["refined_query"],
                    "query_type": state["query_type"],
                    "sub_queries": state["sub_queries"]})

        # ── Source Identifier ─────────────────────────────────────
        yield _sse({"type": "node", "node": "source_identifier",
                    "status": "Identifying sources..."})
        state = source_identifier_node(state)
        yield _sse({"type": "source_info", "filters": state["metadata_filters"]})

        # ── Retriever ─────────────────────────────────────────────
        n_q = len(state["sub_queries"]) if state["sub_queries"] else 1
        yield _sse({"type": "node", "node": "retriever",
                    "status": f"Parallel search × {n_q} queries..."})
        state = retriever_node(state)
        yield _sse({"type": "retrieval_info",
                    "count": len(state["retrieved_chunks"])})

        # ──────────────────────────────────────────────────────────
        # LAYER 2: Retrieval Guard
        # ──────────────────────────────────────────────────────────
        yield _sse({"type": "node", "node": "retrieval_guard",
                    "status": "Validating retrieval quality..."})
        state = retrieval_guard_node(state)
        if is_blocked(state):
            yield _sse({
                "type":   "safety_block",
                "layer":  SAFETY_LAYER_LABELS["retrieval_guard"],
                "reason": state["safety_reason"],
            })
            yield _sse({"type": "token", "token": state["answer"]})
            yield _sse({"type": "done"})
            return
        yield _sse({"type": "safety_pass", "layer": "retrieval_guard"})

        # ── Post Processor ────────────────────────────────────────
        yield _sse({"type": "node", "node": "post_processor",
                    "status": "Ranking chunks..."})
        state = post_processor_node(state)

        # ── Answer Generator (streaming) ──────────────────────────
        yield _sse({"type": "node", "node": "answer_generator",
                    "status": "Generating answer..."})

        chunks      = state["processed_chunks"] or state["retrieved_chunks"]
        memory_ctx  = state.get("memory_context", {})
        recent_msgs = memory_ctx.get("recent_messages", history)

        if not chunks:
            state["answer"] = "I could not find relevant information in the document."
            yield _sse({"type": "token", "token": state["answer"]})
            yield _sse({"type": "done"})
            return

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

        full_answer   = ""
        prompt_tokens = cached_tokens = output_tokens = 0

        stream = _open_stream(
            model=LLM_LARGE, messages=messages,
            max_tokens=1500, temperature=0.1, stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                full_answer += delta
                yield _sse({"type": "token", "token": delta})
            if hasattr(chunk, 'usage') and chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens
                if hasattr(chunk.usage, 'prompt_tokens_details'):
                    cached_tokens = getattr(
                        chunk.usage.prompt_tokens_details, 'cached_tokens', 0)

        tracker.llm_calls        += 1
        tracker.prompt_tokens    += prompt_tokens
        tracker.cached_tokens    += cached_tokens
        tracker.completion_tokens += output_tokens
        tracker.estimated_cost   += _estimate_cost(
            LLM_LARGE, prompt_tokens, cached_tokens, output_tokens)

        sources = [
            f"{c['filename']} ({c.get('chunk_type','text')} chunk {c['chunk_index']})"
            for c in chunks
        ]
        state["answer"]  = full_answer
        state["sources"] = sources
        yield _sse({"type": "sources", "sources": sources})

        # ──────────────────────────────────────────────────────────
        # LAYER 3: Output Guard (active gate + evaluation in 1 call)
        # ──────────────────────────────────────────────────────────
        yield _sse({"type": "node", "node": "output_guard",
                    "status": "Safety-checking answer..."})
        state = output_guard_node(state)

        if is_blocked(state):
            yield _sse({
                "type":   "safety_block",
                "layer":  SAFETY_LAYER_LABELS["output_guard"],
                "reason": state["safety_reason"],
            })
            # Replace streamed answer with safe fallback
            yield _sse({"type": "answer_replace", "answer": state["answer"]})
            yield _sse({"type": "done"})
            return

        yield _sse({"type": "safety_pass",  "layer": "output_guard"})
        yield _sse({"type": "evaluation",   "data":  state["evaluation"]})

        # Usage summary
        yield _sse({
            "type":           "usage",
            "llm_calls":      tracker.llm_calls,
            "prompt_tokens":  tracker.prompt_tokens,
            "cached_tokens":  tracker.cached_tokens,
            "output_tokens":  tracker.completion_tokens,
            "estimated_cost": round(tracker.estimated_cost, 5),
            "cache_savings":  round(tracker.cached_tokens * 0.15 / 1_000_000, 6),
        })

        # Cache successful responses only (not blocked ones)
        if full_answer and not state.get("safety_blocked"):
            set_cached_response(query_hash, doc_id, full_answer,
                                sources, state["evaluation"])

        # Persist
        save_message(session_id, "user",      query)
        save_message(session_id, "assistant", full_answer,
                     sources, state["evaluation"], state["query_type"])

        try:
            maybe_summarise(session_id, history)
            extract_and_store_facts(session_id, state["refined_query"], full_answer)
        except Exception as e:
            logger.warning(f"[Memory] {e}")

        yield _sse({"type": "done"})

    except Exception as e:
        logger.exception("Streaming pipeline failed")
        yield _sse({"type": "error", "message": str(e)})
