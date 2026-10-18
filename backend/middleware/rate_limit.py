"""
Per-IP rate limiting via slowapi.

Every /chat/stream or /upload call costs real OpenAI money and compute;
with no limit, one script or one bad actor can run up an unbounded bill.
Limits are per-endpoint and configurable via env (see config.py) so a
portfolio demo can keep them low without touching code.
"""
from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
