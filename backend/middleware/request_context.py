"""
Request-ID middleware.

Without this, a busy log stream interleaves lines from many concurrent
requests with no way to tell which 9-node pipeline run a given log
line belongs to. This assigns one short ID per request, injects it
into every log record for the duration of the request (via a
contextvar-based logging.Filter), and echoes it back as X-Request-ID
so a user can quote it when reporting an error.
"""
from __future__ import annotations
import contextvars
import logging
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

_request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_ctx.get()
        return True


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        token = _request_id_ctx.set(rid)
        try:
            response = await call_next(request)
        finally:
            _request_id_ctx.reset(token)
        response.headers["X-Request-ID"] = rid
        return response


def current_request_id() -> str:
    return _request_id_ctx.get()
