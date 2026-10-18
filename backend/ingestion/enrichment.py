"""
Content Enrichment (LLM large) — diagram: bottom pipeline.

Steps:
1. Table enrichment  — understand table contents, generate summaries
2. Custom metadata   — topics, entities, content_type, numerical flags
3. Returns enriched metadata for the whole document
"""
from __future__ import annotations
import json, logging

from openai import OpenAI
from backend.config import OPENAI_API_KEY, LLM_LARGE
from backend.ingestion.parser import ParsedDocument
from backend.prompts.prompt_configs import (
    TABLE_ENRICHMENT_SYSTEM,
    TABLE_ENRICHMENT_TEMPLATE,
)

logger = logging.getLogger(__name__)
_client = OpenAI(api_key=OPENAI_API_KEY)


def enrich_document(doc: ParsedDocument, sample_chars: int = 3000) -> dict:
    """
    Run LLM-large enrichment on document content.
    Returns custom_metadata dict to be stored alongside the document.
    """
    sample = doc.raw_text[:sample_chars]

    # If tables exist, include a table sample too
    if doc.tables:
        table_sample = "\n\n".join(
            f"[{t.get('title','Table')}]\n{t['content'][:400]}"
            for t in doc.tables[:3]
        )
        sample += f"\n\n--- Tables Found ---\n{table_sample}"

    prompt = TABLE_ENRICHMENT_TEMPLATE.format(content=sample)

    try:
        resp = _client.chat.completions.create(
            model=LLM_LARGE,
            messages=[
                {"role": "system", "content": TABLE_ENRICHMENT_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=800,
        )
        result = json.loads(resp.choices[0].message.content or "{}")
        return {
            "has_tables":      result.get("has_tables", doc.has_tables),
            "table_summaries": result.get("table_summaries", []),
            "key_entities":    result.get("key_entities", []),
            "content_type":    result.get("content_type", "general"),
            "custom_metadata": result.get("custom_metadata", {}),
        }
    except Exception as e:
        logger.warning(f"Content enrichment failed: {e}; using defaults.")
        return {
            "has_tables":      doc.has_tables,
            "table_summaries": [],
            "key_entities":    [],
            "content_type":    "general",
            "custom_metadata": {},
        }
