"""
LangGraph RAG Agent v5 — 9-node pipeline with 3-layer safety.

Full topology:
  input_guard          ← Layer 1: blocks bad queries instantly
      ↓
  memory_loader
      ↓
  query_optimizer
      ↓
  source_identifier
      ↓
  retriever
      ↓
  retrieval_guard      ← Layer 2: blocks low-confidence retrieval
      ↓
  post_processor
      ↓
  answer_generator
      ↓
  output_guard         ← Layer 3: blocks hallucinated/irrelevant answers
      ↓
  END

Safety short-circuit: if any guard sets safety_blocked=True,
subsequent nodes detect this and skip processing.
"""
from langgraph.graph import StateGraph, END

from backend.agent.state import AgentState
from backend.agent.nodes import (
    memory_loader_node,
    query_optimizer_node,
    source_identifier_node,
    retriever_node,
    post_processor_node,
    answer_generator_node,
)
from backend.safety.safety_nodes import (
    input_guard_node,
    retrieval_guard_node,
    output_guard_node,
)


def _should_continue(state: AgentState) -> str:
    """
    Conditional edge: if safety blocked, jump straight to END.
    Prevents wasting API calls after a block.
    """
    if state.get("safety_blocked", False):
        return "blocked"
    return "continue"


def build_rag_graph():
    graph = StateGraph(AgentState)

    # ── Register all 9 nodes ──────────────────────────────────────
    graph.add_node("input_guard",       input_guard_node)       # Safety L1
    graph.add_node("memory_loader",     memory_loader_node)
    graph.add_node("query_optimizer",   query_optimizer_node)
    graph.add_node("source_identifier", source_identifier_node)
    graph.add_node("retriever",         retriever_node)
    graph.add_node("retrieval_guard",   retrieval_guard_node)   # Safety L2
    graph.add_node("post_processor",    post_processor_node)
    graph.add_node("answer_generator",  answer_generator_node)
    graph.add_node("output_guard",      output_guard_node)      # Safety L3

    # ── Entry ─────────────────────────────────────────────────────
    graph.set_entry_point("input_guard")

    # ── Conditional edge after input_guard ────────────────────────
    # If blocked → END immediately (save all downstream API calls)
    graph.add_conditional_edges(
        "input_guard",
        _should_continue,
        {"continue": "memory_loader", "blocked": END},
    )

    # ── Normal flow ───────────────────────────────────────────────
    graph.add_edge("memory_loader",     "query_optimizer")
    graph.add_edge("query_optimizer",   "source_identifier")
    graph.add_edge("source_identifier", "retriever")

    # ── Conditional edge after retriever ─────────────────────────
    graph.add_conditional_edges(
        "retrieval_guard",
        _should_continue,
        {"continue": "post_processor", "blocked": END},
    )
    graph.add_edge("retriever", "retrieval_guard")

    # ── Continue to generation ────────────────────────────────────
    graph.add_edge("post_processor",   "answer_generator")

    # ── Conditional edge after output_guard ──────────────────────
    graph.add_conditional_edges(
        "output_guard",
        _should_continue,
        {"continue": END, "blocked": END},   # both go to END, answer field differs
    )
    graph.add_edge("answer_generator", "output_guard")

    return graph.compile()


rag_agent = build_rag_graph()
