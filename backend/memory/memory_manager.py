"""
Memory Manager — 3-tier memory system.

Tier 1 · Short-term   : Last N messages verbatim (sliding window)
Tier 2 · Mid-term     : LLM-generated conversation summary (every K turns)
Tier 3 · Long-term    : Key facts extracted + stored as pgvector embeddings
                        → Semantically retrieved per query
"""
from __future__ import annotations
import json, logging
from typing import List

from openai import OpenAI

from backend.config import (
    OPENAI_API_KEY, LLM_SMALL,
    MEMORY_WINDOW, SUMMARY_THRESHOLD, MAX_FACTS_PER_SESSION,
)
from backend.ingestion.embedder import get_embedding
from backend.database.db import (
    save_memory_summary, get_latest_summary,
    save_memory_facts, search_memory_facts,
    count_messages,
)
from backend.llm_retry import with_llm_retry

logger  = logging.getLogger(__name__)
_client = OpenAI(api_key=OPENAI_API_KEY)


@with_llm_retry
def _create_completion(**kwargs):
    return _client.chat.completions.create(**kwargs)


def _llm_json(prompt: str) -> dict:
    resp = _create_completion(
        model=LLM_SMALL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=800,
    )
    return json.loads(resp.choices[0].message.content or "{}")


def _llm_text(prompt: str) -> str:
    resp = _client.chat.completions.create(
        model=LLM_SMALL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=500,
    )
    return resp.choices[0].message.content or ""


# ─────────────────────────────────────────────────────────────────
# TIER 2 · Conversation Summarisation
# ─────────────────────────────────────────────────────────────────

def maybe_summarise(session_id: str, full_history: List[dict]) -> None:
    """
    If conversation exceeds SUMMARY_THRESHOLD messages,
    summarise the older portion and store it.
    Called after every assistant turn.
    """
    if len(full_history) <= SUMMARY_THRESHOLD:
        return

    # Only summarise messages older than the recent window
    to_summarise = full_history[:-MEMORY_WINDOW]
    if not to_summarise:
        return

    # Check if we already have a summary covering this many messages
    existing = get_latest_summary(session_id)
    if existing and existing["covers_up_to"] >= len(to_summarise):
        return   # Already summarised, skip

    convo_text = "\n".join(
        f"{m['role'].upper()}: {m['content'][:300]}" for m in to_summarise
    )
    prompt = f"""Summarise this conversation segment into a concise paragraph.
Capture: key topics discussed, questions asked, answers given, user intent.
Be specific — mention document names, numbers, entities if present.

Conversation:
{convo_text}

Respond with just the summary paragraph."""

    try:
        summary = _llm_text(prompt)
        save_memory_summary(session_id, summary, covers_up_to=len(to_summarise))
        logger.info(f"[Memory] Saved summary covering {len(to_summarise)} messages.")
    except Exception as e:
        logger.warning(f"[Memory] Summarisation failed: {e}")


# ─────────────────────────────────────────────────────────────────
# TIER 3 · Key Fact Extraction + pgvector Storage
# ─────────────────────────────────────────────────────────────────

def extract_and_store_facts(
    session_id: str,
    query: str,
    answer: str,
) -> None:
    """
    After each turn, extract key facts and store them
    as embeddings in pgvector for semantic retrieval later.
    """
    prompt = f"""Extract 1-3 important facts from this Q&A exchange that would be
useful to remember for future questions in this conversation.

Question: {query}
Answer: {answer[:500]}

Respond ONLY with JSON:
{{
  "facts": [
    {{"fact": "<concise fact>", "relevance": "<why this matters>"}},
    ...
  ]
}}

Only extract genuinely important facts (entities, numbers, conclusions).
Return empty list if nothing significant."""

    try:
        result = _llm_json(prompt)
        facts  = result.get("facts", [])[:3]
        if not facts:
            return

        # Embed each fact for semantic retrieval
        embeddings = [get_embedding(f["fact"]) for f in facts]
        save_memory_facts(session_id, facts, embeddings)
        logger.info(f"[Memory] Stored {len(facts)} facts for session {session_id[:8]}")
    except Exception as e:
        logger.warning(f"[Memory] Fact extraction failed: {e}")


# ─────────────────────────────────────────────────────────────────
# Memory Context Builder — called by nodes.py
# ─────────────────────────────────────────────────────────────────

def build_memory_context(
    session_id: str,
    query: str,
    full_history: List[dict],
) -> dict:
    """
    Assembles all 3 memory tiers into a structured context dict.

    Returns:
    {
        recent_messages  : last MEMORY_WINDOW messages (verbatim)
        summary          : mid-term summary string (or "")
        relevant_facts   : list of semantically retrieved facts
        memory_prompt    : formatted string ready to inject into LLM prompt
    }
    """
    # Tier 1 — sliding window
    recent = full_history[-MEMORY_WINDOW:]

    # Tier 2 — conversation summary
    summary_record = get_latest_summary(session_id)
    summary        = summary_record["summary"] if summary_record else ""

    # Tier 3 — semantic fact retrieval via pgvector
    relevant_facts: List[str] = []
    try:
        query_emb      = get_embedding(query)
        relevant_facts = search_memory_facts(session_id, query_emb, top_k=5)
    except Exception as e:
        logger.warning(f"[Memory] Fact retrieval failed: {e}")

    # Build formatted memory prompt
    parts = []
    if summary:
        parts.append(f"📋 Conversation Summary (earlier context):\n{summary}")
    if relevant_facts:
        facts_text = "\n".join(f"  • {f}" for f in relevant_facts)
        parts.append(f"🧠 Relevant Facts from Memory:\n{facts_text}")

    memory_prompt = "\n\n".join(parts) if parts else ""

    return {
        "recent_messages": recent,
        "summary":         summary,
        "relevant_facts":  relevant_facts,
        "memory_prompt":   memory_prompt,
    }
