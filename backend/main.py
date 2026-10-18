"""FastAPI v5 — safety-aware async backend."""
import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
import asyncio

from fastapi import Depends, FastAPI, Request, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from slowapi.errors import RateLimitExceeded

from backend.config import (
    CORS_ORIGINS, ENVIRONMENT, MAX_UPLOAD_MB, OPENAI_API_KEY,
    RATE_LIMIT_CHAT, RATE_LIMIT_UPLOAD,
)
from backend.database.db import (
    init_db, init_pool, close_pool, pool_ready,
    save_document, update_document_after_indexing,
    create_session, save_message, get_chat_history, get_all_documents,
    save_chunks, save_golden_sample, get_golden_data,
    search_memory_facts, get_latest_summary,
    get_usage_stats, get_cache_stats,
)
from backend.ingestion.parser     import parse_document, SUPPORTED
from backend.ingestion.enrichment import enrich_document
from backend.ingestion.chunker    import chunk_document
from backend.ingestion.embedder   import get_embeddings, get_embedding
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.agent.graph          import rag_agent
from backend.agent.streaming      import stream_rag_pipeline
from backend.evaluation.batch_eval import run_batch_eval, get_eval_summary
from backend.middleware.request_context import RequestIdMiddleware, RequestIdFilter, current_request_id
from backend.middleware.rate_limit import limiter
from backend.middleware.auth import require_api_key

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | [%(request_id)s] | %(message)s",
)
for _h in logging.getLogger().handlers:
    _h.addFilter(RequestIdFilter())
logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_pool()   # retries internally — see db.py — so a slow-starting
                  # Postgres container doesn't crash the app on import.
    init_db()
    if CORS_ORIGINS == ["*"] and ENVIRONMENT == "production":
        logger.warning("[Startup] ENVIRONMENT=production with CORS_ORIGINS=* — "
                        "set CORS_ORIGINS to your actual frontend origin(s).")
    yield
    close_pool()


app = FastAPI(
    title="Agentic RAG v5 — Safety + Optimized",
    description="9-node pipeline: 3-layer safety + pgvector + memory + streaming + caching",
    version="5.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS,
                   allow_methods=["*"], allow_headers=["*"])
app.add_middleware(RequestIdMiddleware)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={
        "error": "rate_limited",
        "detail": "Too many requests — please slow down.",
        "request_id": current_request_id(),
    })


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Full trace goes to the server log (with request ID for correlation);
    # the client gets a generic message so internals never leak.
    rid = current_request_id()
    logger.exception(f"[Unhandled] {request.method} {request.url.path}")
    return JSONResponse(status_code=500, content={
        "error": "internal_error",
        "detail": "Something went wrong. If this persists, include the request ID below.",
        "request_id": rid,
    })


class SessionRequest(BaseModel):
    doc_id: str
class ChatRequest(BaseModel):
    query: str; session_id: str; doc_id: str
class GoldenSampleRequest(BaseModel):
    doc_id: str; question: str; expected: str


@app.get("/health")
async def health():
    """Liveness only — is the process up. Does NOT touch the DB or
    OpenAI, so an orchestrator can't mistake a slow dependency for a
    dead process. Use /ready to check whether it can serve traffic."""
    return {"status": "ok", "version": "5.0.0"}


@app.get("/ready")
async def ready():
    """Readiness — can this instance actually serve a request right
    now. Checked separately from /health so a load balancer can keep
    routing to healthy-but-still-starting instances during a DB
    hiccup without killing the process."""
    db_ok = await asyncio.to_thread(pool_ready)
    checks = {"database": db_ok, "openai_key_configured": bool(OPENAI_API_KEY)}
    status_code = 200 if all(checks.values()) else 503
    return JSONResponse(status_code=status_code, content={
        "ready": all(checks.values()), "checks": checks,
    })


@app.post("/upload", dependencies=[Depends(require_api_key)])
@limiter.limit(RATE_LIMIT_UPLOAD)
async def upload_document(request: Request, file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED:
        raise HTTPException(400, f"Unsupported type '{ext}'.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp_path = tmp.name
        written = 0
        while chunk := file.file.read(1024 * 1024):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                tmp.close()
                os.unlink(tmp_path)
                raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB}MB limit.")
            tmp.write(chunk)
    try:
        parsed    = await asyncio.to_thread(parse_document,  tmp_path)
        enriched  = await asyncio.to_thread(enrich_document, parsed)
        doc_id    = save_document(file.filename, ext, enriched["has_tables"],
                                  enriched["content_type"], enriched["custom_metadata"])
        chunks    = await asyncio.to_thread(chunk_document, parsed, doc_id, file.filename, enriched)
        embeddings = await asyncio.to_thread(get_embeddings, [c["text"] for c in chunks])
        save_chunks(chunks, embeddings)
        retriever  = HybridRetriever(doc_id)
        await asyncio.to_thread(retriever.build, chunks)
        update_document_after_indexing(doc_id, len(chunks), enriched["has_tables"],
                                       enriched["content_type"], enriched["custom_metadata"])
        return {"doc_id": doc_id, "filename": file.filename,
                "chunk_count": len(chunks), "has_tables": enriched["has_tables"],
                "content_type": enriched["content_type"],
                "key_entities": enriched.get("key_entities", [])}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(500, str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/session")
async def create_chat_session(req: SessionRequest):
    return {"session_id": create_session(req.doc_id)}


@app.post("/chat/stream", dependencies=[Depends(require_api_key)])
@limiter.limit(RATE_LIMIT_CHAT)
async def chat_stream(request: Request, req: ChatRequest):
    return StreamingResponse(
        stream_rag_pipeline(req.query, req.session_id, req.doc_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/chat", dependencies=[Depends(require_api_key)])
@limiter.limit(RATE_LIMIT_CHAT)
async def chat(request: Request, req: ChatRequest):
    history = get_chat_history(req.session_id, limit=20)
    state   = {
        "query": req.query, "session_id": req.session_id, "doc_id": req.doc_id,
        "chat_history": history, "memory_context": {}, "refined_query": "",
        "sub_queries": [], "query_type": "", "relevant_doc_ids": [],
        "metadata_filters": {}, "retrieved_chunks": [], "processed_chunks": [],
        "answer": "", "sources": [], "evaluation": {},
        "safety_blocked": False, "safety_reason": "", "safety_layer": "",
    }
    try:
        result = await asyncio.to_thread(rag_agent.invoke, state)
    except Exception as e:
        raise HTTPException(500, str(e))
    save_message(req.session_id, "user", req.query)
    save_message(req.session_id, "assistant", result["answer"],
                 result["sources"], result["evaluation"], result["query_type"])
    return {
        "answer": result["answer"], "sources": result["sources"],
        "evaluation": result["evaluation"], "query_type": result["query_type"],
        "safety_blocked": result.get("safety_blocked", False),
        "safety_layer":   result.get("safety_layer", ""),
    }


@app.get("/documents")
async def list_documents(): return get_all_documents()

@app.get("/history/{session_id}")
async def history(session_id: str): return get_chat_history(session_id)

@app.get("/memory/summary/{session_id}")
async def get_memory_summary(session_id: str):
    return get_latest_summary(session_id) or {"summary": "No summary yet."}

@app.get("/memory/facts/{session_id}")
async def get_memory_facts(session_id: str, query: str = ""):
    if not query: return {"facts": []}
    emb = await asyncio.to_thread(get_embedding, query)
    return {"facts": search_memory_facts(session_id, emb, top_k=10)}

@app.get("/stats/usage")
async def usage_stats(): return get_usage_stats()

@app.get("/stats/cache")
async def cache_stats(): return get_cache_stats()

@app.post("/golden")
async def add_golden_sample(req: GoldenSampleRequest):
    return {"id": save_golden_sample(req.doc_id, req.question, req.expected)}

@app.get("/golden/{doc_id}")
async def list_golden(doc_id: str): return get_golden_data(doc_id)

@app.post("/batch-eval/{doc_id}", dependencies=[Depends(require_api_key)])
async def batch_evaluate(doc_id: str):
    return await asyncio.to_thread(run_batch_eval, doc_id)

@app.get("/eval-summary/{doc_id}")
async def eval_summary(doc_id: str): return get_eval_summary(doc_id)
