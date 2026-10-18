"""
Feature Store (BM25) — sparse retrieval, file-based.
pgvector handles dense; BM25 handles sparse — both fused via RRF.
"""
from __future__ import annotations
import os, pickle
from typing import List, Tuple

from rank_bm25 import BM25Okapi
from backend.config import BM25_INDEX_DIR, TOP_K


class FeatureStore:
    def __init__(self, doc_id: str) -> None:
        self.doc_id      = doc_id
        self._path       = os.path.join(BM25_INDEX_DIR, f"{doc_id}.pkl")
        self.bm25:  BM25Okapi | None = None
        self.chunks: List[dict]       = []
        if os.path.exists(self._path):
            self._load()

    def build(self, chunks: List[dict]) -> None:
        self.chunks = chunks
        self.bm25   = BM25Okapi([c["text"].lower().split() for c in chunks])
        self._save()

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        metadata_filters: dict | None = None,
    ) -> List[Tuple[dict, float]]:
        if not self.bm25 or not self.chunks:
            return []
        scores = self.bm25.get_scores(query.lower().split())
        ranked = scores.argsort()[::-1]
        results = []
        for idx in ranked:
            if scores[idx] <= 0:
                break
            chunk = self.chunks[idx]
            if metadata_filters and not _matches(chunk, metadata_filters):
                continue
            results.append((chunk, float(scores[idx])))
            if len(results) >= top_k:
                break
        return results

    def _save(self):
        with open(self._path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)

    def _load(self):
        with open(self._path, "rb") as f:
            d = pickle.load(f)
            self.bm25   = d["bm25"]
            self.chunks = d["chunks"]


def _matches(chunk: dict, filters: dict) -> bool:
    meta = chunk.get("metadata", {})
    if filters.get("content_type") and meta.get("content_type") != filters["content_type"]:
        return False
    if filters.get("has_tables") is not None and meta.get("has_tables") != filters["has_tables"]:
        return False
    if filters.get("keywords"):
        text = chunk["text"].lower()
        if not any(k.lower() in text for k in filters["keywords"]):
            return False
    return True
