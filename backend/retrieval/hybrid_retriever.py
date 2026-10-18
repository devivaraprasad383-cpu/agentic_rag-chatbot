"""
Hybrid Retriever — pgvector (dense) + BM25 (sparse) → RRF.
"""
from __future__ import annotations
from typing import List

from backend.retrieval.vector_store  import PGVectorStore
from backend.retrieval.feature_store import FeatureStore
from backend.config import TOP_K, RRF_K


def _rrf(dense: list, sparse: list, k: int = RRF_K) -> List[dict]:
    scores: dict[str, dict] = {}
    for rank, (chunk, _) in enumerate(dense):
        cid = chunk["chunk_id"]
        scores.setdefault(cid, {"chunk": chunk, "score": 0.0})
        scores[cid]["score"] += 1.0 / (k + rank + 1)
    for rank, (chunk, _) in enumerate(sparse):
        cid = chunk["chunk_id"]
        scores.setdefault(cid, {"chunk": chunk, "score": 0.0})
        scores[cid]["score"] += 1.0 / (k + rank + 1)
    return [v["chunk"] for v in sorted(scores.values(), key=lambda x: x["score"], reverse=True)]


class HybridRetriever:
    def __init__(self, doc_id: str) -> None:
        self.doc_id        = doc_id
        self.vector_store  = PGVectorStore()
        self.feature_store = FeatureStore(doc_id)

    def build(self, chunks: List[dict]) -> None:
        """BM25 is file-based; pgvector is already saved to DB by ingestion pipeline."""
        self.feature_store.build(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        metadata_filters: dict | None = None,
    ) -> List[dict]:
        dense  = self.vector_store.search(query, [self.doc_id], top_k, metadata_filters)
        sparse = self.feature_store.search(query, top_k, metadata_filters)
        return _rrf(dense, sparse)[:top_k]
