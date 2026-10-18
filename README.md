# Agentic RAG Document Chatbot (v5)

Document Q&A over PDF / DOCX / TXT / Markdown, built as a 9-node LangGraph pipeline with a
3-layer safety system, hybrid retrieval, persistent memory and token streaming.

> **Status:** code complete; **benchmark results are pending** (see [Evaluation](#evaluation)).
> No quality or cost numbers are claimed in this README until they have been measured
> with `benchmarks/run_benchmarks.py`.

## Architecture

```
input_guard ──► memory_loader ──► query_optimizer ──► source_identifier ──► retriever
 (Safety L1)                                                                   │
                                                                               ▼
   END ◄── output_guard ◄── answer_generator ◄── post_processor ◄── retrieval_guard
        (Safety L3)                                                  (Safety L2)
```

Any guard that blocks sets `safety_blocked`, and the graph jumps straight to `END`
so no further LLM calls are spent.

| Layer | Where | What it does |
|---|---|---|
| L1 Input guard | before any LLM work | length checks, regex prompt-injection and out-of-scope patterns, LLM fallback for ambiguous queries |
| L2 Retrieval guard | after retrieval | blocks when too few chunks or average similarity is below threshold |
| L3 Output guard | after generation | LLM-as-judge scores faithfulness and answer relevance; blocks answers below threshold |

Thresholds and patterns live in `backend/safety/safety_config.py`. Answers stream token by
token; if L3 blocks an answer after it has streamed, the server sends an `answer_replace`
SSE event so the UI swaps the text for a safe message.

### Retrieval
Hybrid search: pgvector (cosine) + BM25, fused with Reciprocal Rank Fusion (`RRF_K=60`),
top-10 retrieved then post-processed down to 5. Chunking is table-aware.

### Memory (3 tiers)
Recent message window, rolling conversation summary, and a semantic fact store in pgvector.

### Models and caching
`gpt-4o` for answer generation and table enrichment; `gpt-4o-mini` for query optimisation,
source identification, post-processing and memory; `text-embedding-3-small` for embeddings.
Embedding cache (in-process + Postgres), response cache, and per-query token/cost tracking.

## Quickstart

```bash
cp .env.example .env        # set OPENAI_API_KEY and PG_PASSWORD (required, no defaults)
docker compose up --build
```

- API: http://localhost:8000 (docs at `/docs`)
- UI (Gradio): http://localhost:7860
- Readiness: `curl localhost:8000/ready`

On Windows without Docker or Linux, see [`WINDOWS.md`](WINDOWS.md).

Set `ENVIRONMENT=production` for multi-worker uvicorn and config warnings.

## API

| Endpoint | Purpose |
|---|---|
| `POST /upload` | ingest a document, returns `doc_id` |
| `POST /session` | create a chat session for a `doc_id` |
| `POST /chat/stream` | SSE streaming chat (node progress, tokens, sources, evaluation) |
| `POST /chat` | non-streaming chat |
| `GET /health`, `GET /ready` | liveness / readiness (DB + config) |
| `GET /stats/usage`, `GET /stats/cache` | cost and cache statistics |
| `POST /golden`, `POST /batch-eval/{doc_id}`, `GET /eval-summary/{doc_id}` | in-app golden set and batch evaluation |

`/upload`, `/chat`, `/chat/stream` and `/batch-eval` are rate-limited per IP and can be
protected with a shared key (`API_KEY`, sent as `Authorization: Bearer ...`).

## Configuration

See `.env.example`. Key settings: `OPENAI_API_KEY`, `PG_*`, `DB_POOL_MIN/MAX`,
`ENVIRONMENT`, `CORS_ORIGINS`, `MAX_UPLOAD_MB`, `RATE_LIMIT_CHAT`, `RATE_LIMIT_UPLOAD`, `API_KEY`.

## Evaluation

`benchmarks/golden.json` holds 58 questions built from NVIDIA's FY2025 10-K
(factoid, table lookup, multi-hop, unanswerable, prompt-injection, out-of-scope, and
benign look-alikes for false-positive testing). See `benchmarks/README_GOLDEN.md`.

```bash
# upload the 10-K through /upload first to get a doc_id
python benchmarks/run_benchmarks.py --doc_id <ID> --questions benchmarks/golden.json
```

Reports retrieval metrics (Recall@10, Precision@5, MRR, NDCG@10) computed from retrieved
chunks against labelled gold snippets, judge scores, safety attack-recall and
false-positive rate, latency percentiles and cost, each with 95% bootstrap confidence
intervals, plus git SHA and dataset hash.

### Results
_Pending. Paste the output of a real run here._

## Production notes

Connection pooling, retries with backoff on OpenAI calls, request-ID logging, a global
error handler, upload size cap, non-root container with healthcheck, pinned dependencies.
Details and known gaps are in [`AUDIT2.md`](AUDIT2.md).

## Known limitations
- Not yet verified end to end against a live Postgres and OpenAI key after the hardening changes.
- BM25 indexes are stored as files under `data/`, not in Postgres.
- Single shared API key, not per-user auth. No distributed tracing. No load testing yet.
- Regex injection patterns can false-positive on legitimate questions (for example
  "act as a ..."); the golden set includes look-alike cases to measure this.

## Layout
```
backend/{agent,safety,retrieval,ingestion,memory,database,evaluation,middleware,prompts}
frontend/app.py     benchmarks/     tests/     docker-compose.yml     Dockerfile
```