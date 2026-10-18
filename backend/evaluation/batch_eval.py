"""Batch Evaluation — post-prod metrics against golden test data."""
from __future__ import annotations
import logging
from statistics import mean

from backend.agent.graph import rag_agent
from backend.database.db import (
    get_golden_data, save_batch_eval,
    get_batch_eval_results, create_session,
)

logger = logging.getLogger(__name__)


def run_batch_eval(doc_id: str) -> dict:
    golden = get_golden_data(doc_id)
    if not golden:
        return {"error": "No golden test data found."}

    session_id = create_session(doc_id)
    results    = []

    for sample in golden:
        initial_state = {
            "query": sample["question"], "session_id": session_id,
            "doc_id": doc_id, "chat_history": [],
            "memory_context": {}, "refined_query": "", "sub_queries": [],
            "query_type": "", "relevant_doc_ids": [], "metadata_filters": {},
            "retrieved_chunks": [], "processed_chunks": [],
            "answer": "", "sources": [], "evaluation": {},
        }
        try:
            result  = rag_agent.invoke(initial_state)
            ev      = result.get("evaluation", {})
            f, ar, cr = ev.get("faithfulness",0), ev.get("answer_relevance",0), ev.get("context_relevance",0)
            avg    = mean([f, ar, cr]) if any([f, ar, cr]) else 0
            passed = avg >= 3.5
            save_batch_eval(doc_id, sample["question"], result["answer"], f, ar, cr, passed)
            results.append({"question": sample["question"], "faithfulness": f,
                            "answer_relevance": ar, "context_relevance": cr,
                            "avg_score": round(avg, 2), "passed": passed})
        except Exception as e:
            logger.error(f"Batch eval failed: {e}")

    if not results:
        return {"error": "All evaluations failed."}

    return {
        "total":            len(results),
        "passed":           sum(1 for r in results if r["passed"]),
        "pass_rate":        round(sum(1 for r in results if r["passed"]) / len(results), 2),
        "avg_faithfulness": round(mean(r["faithfulness"]     for r in results), 2),
        "avg_answer_rel":   round(mean(r["answer_relevance"] for r in results), 2),
        "avg_context_rel":  round(mean(r["context_relevance"] for r in results), 2),
        "avg_overall":      round(mean(r["avg_score"]         for r in results), 2),
        "details":          results,
    }


def get_eval_summary(doc_id: str) -> dict:
    rows = get_batch_eval_results(doc_id)
    if not rows: return {}
    return {
        "total_runs":       len(rows),
        "avg_faithfulness": round(mean(r["faithfulness"]     for r in rows), 2),
        "avg_answer_rel":   round(mean(r["answer_relevance"] for r in rows), 2),
        "avg_context_rel":  round(mean(r["context_relevance"] for r in rows), 2),
        "pass_rate":        round(sum(1 for r in rows if r["passed"]) / len(rows), 2),
    }
