"""
pgvector Vector Store — replaces FAISS entirely.

Why pgvector > FAISS for production:
  ✅ No separate .index files — embeddings live IN Postgres
  ✅ HNSW index for fast ANN — as fast as FAISS in practice
  ✅ Native SQL metadata filtering (WHERE clauses)
  ✅ Single DB for vectors + metadata + sessions
  ✅ ACID transactions — no partial index corruption
  ✅ Standard pg_dump backups
  ✅ Concurrent multi-user safe
"""
from __future__ import annotations
from typing import List, Tuple

from backend.database.db import vector_search
from backend.ingestion.embedder import get_embedding
from backend.config import TOP_K


class PGVectorStore:
    """
    Drop-in replacement for FAISSStore.
    Uses PostgreSQL + pgvector for all vector operations.
    """

    def search(
        self,
        query: str,
        doc_ids: list[str],
        top_k: int = TOP_K,
        metadata_filters: dict | None = None,
    ) -> List[Tuple[dict, float]]:
        """
        Cosine similarity search via pgvector.
        Single SQL query handles both ANN search AND metadata filtering.
        """
        query_embedding = get_embedding(query)
        results = vector_search(
            query_embedding=query_embedding,
            doc_ids=doc_ids,
            top_k=top_k,
            metadata_filters=metadata_filters,
        )
        return [(r, r["score"]) for r in results]
