"""
Optional API-key auth.

Wired onto the cost-incurring routes only (/upload, /chat, /chat/stream,
/batch-eval) — read-only routes (/documents, /history, /stats/*) stay
open since they don't spend OpenAI budget. This is single-shared-key
auth, not per-user accounts: enough to keep a public demo link from
being hit by bots/scrapers, not a real authz system. With API_KEY
unset (the default), this is a no-op so local/dev usage is unaffected.
If you need real per-user auth, that's a bigger design decision —
ask and we can add it.
"""
from __future__ import annotations

from fastapi import Header, HTTPException

from backend.config import API_KEY


async def require_api_key(authorization: str | None = Header(default=None)) -> None:
    if not API_KEY:
        return  # auth disabled — explicit opt-in via API_KEY env var
    expected = f"Bearer {API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")
