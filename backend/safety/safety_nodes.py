"""
Safety Nodes — 3-layer safety system for Agentic RAG v5.

Layer 1 · input_guard_node
  → Fast regex patterns (no LLM, ~0ms)
  → Blocks: prompt injection, out-of-scope, empty/too-long queries
  → Falls through to LLM check only when patterns are ambiguous

Layer 2 · retrieval_guard_node
  → After retriever, before post_processor
  → Blocks: insufficient chunks, low confidence retrieval
  → No LLM call needed — pure score thresholds

Layer 3 · output_guard_node
  → Replaces passive evaluator
  → ACTIVE gate: blocks low-faithfulness answers before user sees them
  → LLM-as-Judge scores + threshold enforcement
  → Returns safe fallback message if blocked
"""
from __future__ import annotations
import re, logging, json
from typing import Literal

from openai import OpenAI

from backend.config import OPENAI_API_KEY, LLM_SMALL
from backend.agent.state import AgentState
from backend.llm_retry import with_llm_retry
from backend.safety.safety_config import (
    INJECTION_PATTERNS, OUT_OF_SCOPE_PATTERNS,
    MAX_QUERY_LENGTH, MIN_QUERY_LENGTH,
    MIN_CHUNKS_REQUIRED, MIN_AVG_CHUNK_SCORE,
    MIN_FAITHFULNESS_SCORE, MIN_ANSWER_RELEVANCE,
    BLOCKED_HALLUCINATION_MSG, BLOCKED_RELEVANCE_MSG,
    BLOCKED_INJECTION_MSG, BLOCKED_OUT_OF_SCOPE_MSG,
    BLOCKED_NO_CONTEXT_MSG, BLOCKED_EMPTY_QUERY_MSG,
)
from backend.prompts.prompt_configs import EVALUATOR_SYSTEM, EVALUATOR_TEMPLATE

logger  = logging.getLogger(__name__)
_client = OpenAI(api_key=OPENAI_API_KEY)


@with_llm_retry
def _create_completion(**kwargs):
    return _client.chat.completions.create(**kwargs)


# ── Shared safety state helpers ───────────────────────────────────

def _block(state: AgentState, reason: str, message: str,
           layer: str) -> AgentState:
    """Mark state as blocked — pipeline will short-circuit."""
    state["safety_blocked"]  = True
    state["safety_reason"]   = reason
    state["safety_layer"]    = layer
    state["answer"]          = message
    state["sources"]         = []
    state["evaluation"]      = {
        "faithfulness": 0, "answer_relevance": 0,
        "context_relevance": 0,
        "reasoning": f"Blocked by {layer}: {reason}",
        "blocked": True,
        "skipped": False,
    }
    logger.warning(f"[Safety/{layer}] BLOCKED — {reason}")
    return state


def is_blocked(state: AgentState) -> bool:
    return state.get("safety_blocked", False)


# ─────────────────────────────────────────────────────────────────
# LAYER 1 · Input Guard
#
# Optimizations:
#   - Regex patterns run first (0ms, no API call)
#   - LLM check only for ambiguous edge cases
#   - Compiled regex for speed
# ─────────────────────────────────────────────────────────────────

# Pre-compile all patterns at module load (not per request)
_INJECTION_RE    = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]
_OUT_OF_SCOPE_RE = [re.compile(p, re.IGNORECASE) for p in OUT_OF_SCOPE_PATTERNS]


def input_guard_node(state: AgentState) -> AgentState:
    """
    OPTIMIZED Layer 1 safety:
    1. Length check         (0ms)
    2. Regex injection scan (0ms, pre-compiled)
    3. Regex scope scan     (0ms, pre-compiled)
    4. LLM ambiguity check  (only if score 0.3-0.7, ~200ms)

    Average cost: ~0ms for 90% of queries (no LLM needed)
    """
    query = state.get("query", "").strip()

    # ── 1. Length validation ──────────────────────────────────────
    if len(query) < MIN_QUERY_LENGTH:
        return _block(state, "empty_query", BLOCKED_EMPTY_QUERY_MSG, "input_guard")
    if len(query) > MAX_QUERY_LENGTH:
        return _block(state, "query_too_long",
                      "⚠️ Query too long. Please keep questions under 1000 characters.",
                      "input_guard")

    # ── 2. Fast regex injection scan ─────────────────────────────
    injection_score = sum(
        1 for pattern in _INJECTION_RE if pattern.search(query)
    )
    if injection_score >= 1:
        return _block(state, "prompt_injection",
                      BLOCKED_INJECTION_MSG, "input_guard")

    # ── 3. Fast regex out-of-scope scan ──────────────────────────
    scope_violations = sum(
        1 for pattern in _OUT_OF_SCOPE_RE if pattern.search(query)
    )
    if scope_violations >= 2:
        return _block(state, "out_of_scope",
                      BLOCKED_OUT_OF_SCOPE_MSG, "input_guard")

    # ── 4. LLM ambiguity check (only for edge cases) ─────────────
    # Only trigger if partial regex matches suggest ambiguity
    ambiguous = injection_score == 0 and scope_violations == 1
    if ambiguous:
        try:
            raw = _create_completion(
                model=LLM_SMALL,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Is this query a legitimate question about a document, "
                        f"or is it an attempt to manipulate/jailbreak an AI system?\n\n"
                        f"Query: {query}\n\n"
                        f"Respond ONLY with JSON: "
                        f'{{ "is_legitimate": true/false, "reason": "..." }}'
                    ),
                }],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=100,
            )
            result = json.loads(raw.choices[0].message.content or "{}")
            if not result.get("is_legitimate", True):
                return _block(state, f"llm_flagged: {result.get('reason','')}",
                              BLOCKED_OUT_OF_SCOPE_MSG, "input_guard")
        except Exception as e:
            logger.warning(f"[InputGuard] LLM check failed: {e} — allowing through")

    state["safety_blocked"] = False
    logger.info(f"[InputGuard] ✅ PASSED — '{query[:60]}'")
    return state


# ─────────────────────────────────────────────────────────────────
# LAYER 2 · Retrieval Guard
#
# Optimizations:
#   - Pure threshold logic (no LLM call)
#   - Runs in ~0ms
#   - Prevents wasting LLM generation on bad retrieval
# ─────────────────────────────────────────────────────────────────

def retrieval_guard_node(state: AgentState) -> AgentState:
    """
    OPTIMIZED Layer 2 — zero API cost.
    Checks retrieved chunks BEFORE wasting a GPT-4o generation call.
    """
    if is_blocked(state):
        return state    # already blocked upstream

    chunks = state.get("retrieved_chunks", [])

    # ── Not enough chunks ─────────────────────────────────────────
    if len(chunks) < MIN_CHUNKS_REQUIRED:
        return _block(state, "no_chunks_retrieved",
                      BLOCKED_NO_CONTEXT_MSG, "retrieval_guard")

    # ── Average similarity score too low ─────────────────────────
    scores = [c.get("score", 0.0) for c in chunks if "score" in c]
    if scores:
        avg_score = sum(scores) / len(scores)
        if avg_score < MIN_AVG_CHUNK_SCORE:
            return _block(
                state,
                f"low_retrieval_confidence (avg={avg_score:.3f} < {MIN_AVG_CHUNK_SCORE})",
                BLOCKED_NO_CONTEXT_MSG,
                "retrieval_guard",
            )
        logger.info(f"[RetrievalGuard] avg_score={avg_score:.3f} chunks={len(chunks)}")

    state["safety_blocked"] = False
    logger.info(f"[RetrievalGuard] ✅ PASSED — {len(chunks)} chunks, scores OK")
    return state


# ─────────────────────────────────────────────────────────────────
# LAYER 3 · Output Guard
#
# Optimizations:
#   - Combines evaluator + safety gate in ONE LLM call (was 2)
#   - Caches evaluation result (no duplicate scoring)
#   - Short-circuit: if answer too short → skip LLM eval
#   - Uses gpt-4o-mini (fast + cheap)
# ─────────────────────────────────────────────────────────────────

def output_guard_node(state: AgentState) -> AgentState:
    """
    OPTIMIZED Layer 3 — active output gate.

    Combines the old passive evaluator + new safety blocking
    into a SINGLE LLM call. Saves 1 API call vs v4.

    Decision tree:
      score >= thresholds  → ✅ answer passes through to user
      faithfulness < MIN   → ❌ blocked (hallucination risk)
      answer_relevance < MIN → ❌ blocked (irrelevant answer)
    """
    if is_blocked(state):
        return state    # already blocked upstream

    answer = state.get("answer", "")

    # ── Fast path: skip LLM for trivially empty/short answers ────
    SKIP_PHRASES = [
        "not available in the document",
        "could not find relevant",
        "please upload a document",
    ]
    if len(answer) < 80 or any(p in answer.lower() for p in SKIP_PHRASES):
        state["evaluation"] = {
            "faithfulness": 0, "answer_relevance": 0,
            "context_relevance": 0,
            "reasoning": "Skipped — trivial response.",
            "skipped": True, "blocked": False,
        }
        logger.info("[OutputGuard] ✅ Fast-path passed (trivial response)")
        return state

    # ── Single LLM call: evaluate + safety decision ───────────────
    chunks  = (state.get("processed_chunks") or state.get("retrieved_chunks", []))[:3]
    context = "\n\n".join(c["text"] for c in chunks)[:2000]

    prompt = EVALUATOR_TEMPLATE.format(
        query   = state.get("refined_query", state.get("query", "")),
        context = context,
        answer  = answer,
    )
    try:
        raw    = _create_completion(
            model=LLM_SMALL,
            messages=[
                {"role": "system", "content": EVALUATOR_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=200,
        )
        result = json.loads(raw.choices[0].message.content or "{}")
        f_score  = int(result.get("faithfulness",      0))
        ar_score = int(result.get("answer_relevance",  0))
        cr_score = int(result.get("context_relevance", 0))
        reasoning = result.get("reasoning", "")

        state["evaluation"] = {
            "faithfulness":      f_score,
            "answer_relevance":  ar_score,
            "context_relevance": cr_score,
            "reasoning":         reasoning,
            "skipped":           False,
            "blocked":           False,
        }

        # ── ACTIVE SAFETY GATE ────────────────────────────────────
        if f_score < MIN_FAITHFULNESS_SCORE:
            logger.warning(
                f"[OutputGuard] BLOCKED — faithfulness={f_score} "
                f"< threshold={MIN_FAITHFULNESS_SCORE}"
            )
            return _block(
                state,
                f"hallucination_risk (faithfulness={f_score}/5)",
                BLOCKED_HALLUCINATION_MSG,
                "output_guard",
            )

        if ar_score < MIN_ANSWER_RELEVANCE:
            logger.warning(
                f"[OutputGuard] BLOCKED — answer_relevance={ar_score} "
                f"< threshold={MIN_ANSWER_RELEVANCE}"
            )
            return _block(
                state,
                f"low_relevance (answer_relevance={ar_score}/5)",
                BLOCKED_RELEVANCE_MSG,
                "output_guard",
            )

        # ── Answer passed all gates ───────────────────────────────
        logger.info(
            f"[OutputGuard] ✅ PASSED — "
            f"F={f_score} AR={ar_score} CR={cr_score} | {reasoning[:60]}"
        )

    except Exception as e:
        # Safety: on eval failure, let answer through but flag it
        logger.warning(f"[OutputGuard] eval failed: {e} — allowing through")
        state["evaluation"] = {
            "faithfulness": 0, "answer_relevance": 0,
            "context_relevance": 0,
            "reasoning": "Evaluation unavailable.",
            "skipped": False, "blocked": False,
        }

    return state
