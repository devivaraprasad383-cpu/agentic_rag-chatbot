"""
Table-Aware Chunker (diagram: bottom pipeline).

- Regular text → RecursiveCharacterTextSplitter
- Tables       → kept as whole chunks (with markdown + summary metadata)
- Each chunk carries rich metadata for downstream metadata filtering
"""
from __future__ import annotations
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter

from backend.config import CHUNK_SIZE, CHUNK_OVERLAP, TABLE_CHUNK_SIZE
from backend.ingestion.parser import ParsedDocument


def chunk_document(
    doc: ParsedDocument,
    doc_id:   str,
    filename: str,
    enriched_meta: dict,
) -> List[dict]:
    """
    Returns list of chunk dicts:
    {
        chunk_id, doc_id, filename, text,
        chunk_index, chunk_type,   ← "text" | "table"
        metadata: {topics, content_type, has_tables, ...}
    }
    """
    chunks: List[dict] = []
    idx = 0

    # ── 1. Table chunks (kept whole, table-aware) ─────────────────
    table_contents = {t["content"] for t in doc.tables}
    for i, table in enumerate(doc.tables):
        summary = ""
        summaries = enriched_meta.get("table_summaries", [])
        if i < len(summaries):
            summary = summaries[i]

        text = (
            f"[TABLE: {table.get('title', f'Table {i+1}')}]\n"
            f"{table.get('markdown', table['content'])}"
            + (f"\n\nTable Summary: {summary}" if summary else "")
        )
        chunks.append(_make_chunk(
            chunk_id    = f"{doc_id}_table_{i}",
            doc_id      = doc_id,
            filename    = filename,
            text        = text,
            chunk_index = idx,
            chunk_type  = "table",
            enriched    = enriched_meta,
        ))
        idx += 1

    # ── 2. Text chunks (table content excluded to avoid duplication) ─
    # Remove table content from the raw text before splitting
    clean_text = doc.raw_text
    for content in table_contents:
        clean_text = clean_text.replace(content, "")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators    = ["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        keep_separator= True,
    )
    for raw_chunk in splitter.split_text(clean_text):
        if not raw_chunk.strip():
            continue
        chunks.append(_make_chunk(
            chunk_id    = f"{doc_id}_chunk_{idx}",
            doc_id      = doc_id,
            filename    = filename,
            text        = raw_chunk,
            chunk_index = idx,
            chunk_type  = "text",
            enriched    = enriched_meta,
        ))
        idx += 1

    return chunks


def _make_chunk(
    chunk_id: str, doc_id: str, filename: str,
    text: str, chunk_index: int, chunk_type: str,
    enriched: dict,
) -> dict:
    return {
        "chunk_id":    chunk_id,
        "doc_id":      doc_id,
        "filename":    filename,
        "text":        text,
        "chunk_index": chunk_index,
        "chunk_type":  chunk_type,
        "metadata": {
            "content_type":  enriched.get("content_type", "general"),
            "topics":        enriched.get("custom_metadata", {}).get("topics", []),
            "key_entities":  enriched.get("key_entities", []),
            "has_tables":    enriched.get("has_tables", False),
            "numerical_data":enriched.get("custom_metadata", {}).get("numerical_data", False),
            "is_table_chunk":chunk_type == "table",
        },
    }
