"""LangGraph state — v5 adds safety fields."""
from __future__ import annotations
from typing import TypedDict, List, Any


class AgentState(TypedDict):
    # Input
    query:        str
    session_id:   str
    doc_id:       str
    chat_history: List[dict]

    # Memory
    memory_context: dict

    # Pre-Process
    refined_query:     str
    sub_queries:       List[str]
    query_type:        str
    relevant_doc_ids:  List[str]
    metadata_filters:  dict

    # Retrieval
    retrieved_chunks:  List[dict]
    processed_chunks:  List[dict]

    # Generation
    answer:  str
    sources: List[str]

    # Evaluation
    evaluation: dict

    # Safety — v5
    safety_blocked: bool
    safety_reason:  str
    safety_layer:   str

    # Internal
    _tracker: Any
