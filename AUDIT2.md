# Production audit — agentic_rag_v5

The pipeline design itself is good: 9-node LangGraph, 3 real safety gates with correct
short-circuiting, streaming with an `answer_replace` fallback when Output Guard blocks
*after* tokens have already streamed (confirmed in `streaming.py` — this is NOT the bug
I flagged from the diagram alone; the diagram just doesn't show it), 2-tier embedding
cache, prompt caching via static-system-prompt-first ordering, and parameterized SQL
throughout `vector_search` (no injection risk found).

What's missing is everything that keeps a service alive under real traffic and lets you
*reason* about it after a bad deploy. Fixed below, in priority order.

## Fixed in this pass

1. **No connection pooling** (`database/db.py`) — every call opened a brand-new
   `psycopg2.connect()`. Under concurrent chat requests this exhausts Postgres'
   `max_connections` fast and adds TCP+auth latency to every query. Replaced with a
   `ThreadedConnectionPool`, reused across requests, closed cleanly on shutdown.
2. **No readiness check** — `/health` always returned `{"status": "ok"}` even if
   Postgres or the OpenAI key was broken, so Kubernetes/Compose would route traffic to a
   backend that can't actually serve a request. Added `/ready` that pings the DB pool and
   checks `OPENAI_API_KEY` is set, and kept `/health` as a pure liveness check.
3. **`init_db()` at import time with no retry** — on `docker-compose up`, if the app
   container started a beat before Postgres finished its own init scripts (beyond the
   `pg_isready` healthcheck window), the whole process crashed on import. Added a
   bounded retry with backoff around the connection pool's first connection.
4. **No rate limiting** — `/chat/stream` and `/upload` were open to unlimited calls from
   any IP, each one costing real OpenAI money. Added `slowapi`-based per-IP limits
   (configurable via env), with a 429 response before the pipeline even starts.
5. **No retry/backoff on OpenAI calls** — a single transient `RateLimitError` or
   `APITimeoutError` from OpenAI anywhere in the 9-node pipeline failed the entire user
   request. Added a shared `tenacity` retry decorator (3 attempts, exponential backoff,
   only on retryable OpenAI exceptions) and applied it to every direct
   `client.chat.completions.create` / `embeddings.create` call site.
6. **No upload size limit** — `/upload` streamed any size file to a temp file before
   validating anything. Added a `MAX_UPLOAD_MB` env-configurable cap, checked via
   `Content-Length` and enforced again while streaming to disk.
7. **CORS wide open (`allow_origins=["*"]`)** — fine for local dev, not for a deployed
   demo. Made it an env-configured allow-list (`CORS_ORIGINS`), defaulting to `*` only
   when unset, with a startup log warning if so.
8. **No request correlation / structured logs** — logs had no request ID, so you
   couldn't trace one user's query across the 9 nodes in a busy log stream. Added
   middleware that generates a request ID, binds it into the logger via a
   `contextvars`-based filter, and returns it as `X-Request-ID`.
9. **No global exception handler** — an uncaught exception anywhere returned FastAPI's
   default plaintext 500 with a full stack trace exposed to the client. Added a handler
   that logs the full trace server-side and returns a generic JSON error with the
   request ID, so the user can report it without leaking internals.
10. **Dockerfile ran as root, no HEALTHCHECK, `--reload` baked into the "prod" compose
    command.** Added a non-root user, a `HEALTHCHECK` hitting `/health`, split the
    compose command so `--reload` only applies when `ENVIRONMENT=development`, and
    switched the prod command to multiple uvicorn workers.
11. **Unpinned dependencies (`>=` everywhere)** — a `pip install` next month can pull in
    a breaking LangGraph/OpenAI SDK release with no warning. Pinned exact versions from
    what's compatible with this codebase and added `tenacity`, `slowapi`.
12. **Default DB password committed in `.env.example` and `docker-compose.yml`**
    (`ragpassword`) — fine for a laptop demo, worth flagging loudly so it's never copied
    into a real deployment. Added a comment and a startup warning if the default is
    still in use outside `ENVIRONMENT=development`.
13. **No `.dockerignore`** — `.git`, `data/`, `__pycache__` and `.env` were all being
    sent to the Docker build context. Added one.

## Also fixed: optional API-key auth

Added single-shared-key auth (`backend/middleware/auth.py`), applied to the
cost-incurring routes only (`/upload`, `/chat`, `/chat/stream`, `/batch-eval/{doc_id}`) —
read-only stats/history routes stay open. It's a no-op when `API_KEY` is unset (the
default), so local dev is unaffected; set `API_KEY` in `.env` and callers must send
`Authorization: Bearer <key>`. This is enough to keep a public demo link from being
scraped/hit by bots — it is **not** per-user auth. If you want real accounts
(JWT, OAuth, per-user rate limits/usage), that's a bigger design decision — say the word.

## Deliberately NOT done in this pass (flagging, not fixing)

- **BM25 index is still file-based** (`data/bm25_indexes/`), not in Postgres. That's a
  real architectural inconsistency (one retrieval path is ACID, the other isn't) but
  fixing it means a schema change and a reindex path, not a quick patch.
- **Observability (tracing/metrics dashboards).** I added request IDs and structured
  logs, which get you 80% of the debugging value for free. Full OpenTelemetry/Langfuse
  wiring is a separate, larger piece — say the word and I'll add it.
- **Load testing.** Nothing here has been verified under actual concurrent load; the
  pooling and rate-limit changes are the standard fix for the failure modes that would
  show up, but I haven't run a k6/locust test against it.
- **Everything here was verified by compiling every `.py` file in the repo
  (`python -m py_compile`) and running the eval harness end to end against a mock API**
  (`tests/mock_api.py`) — both pass. I have no network access in this environment, so
  none of it has run against a real OpenAI key or a real Postgres instance. Run
  `docker compose up --build` yourself and watch `/ready` before trusting it in prod.

## Where things stand now

- `benchmarks/`, `tests/`, `.github/workflows/ci.yml` from the eval-kit round are now
  merged into this repo (same endpoints the harness expects — `/session`,
  `/chat/stream`, `/stats/usage`, `/stats/cache` — already match `backend/main.py`).
- `benchmarks/golden.example.json` is still a 5-question placeholder. The real
  50-question golden set (next step) still needs the source document you benchmark
  against.
