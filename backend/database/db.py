"""
PostgreSQL persistence — v4 adds:
  - embedding_cache    (persistent embedding cache)
  - response_cache     (repeated query cache)
  - api_usage          (cost tracking per query)
"""
from __future__ import annotations
import json, logging, time, uuid
from contextlib import contextmanager
from datetime import datetime

import psycopg2
import psycopg2.extras
from psycopg2 import pool as _pgpool
from psycopg2.extras import RealDictCursor

from backend.config import PG_DSN, EMBEDDING_DIM, DB_POOL_MIN, DB_POOL_MAX

logger = logging.getLogger(__name__)

# ── Connection pool ───────────────────────────────────────────────
# A fresh TCP+auth connection per call (the old behaviour) exhausts
# Postgres' max_connections under any real concurrency. One pool is
# created at process start and reused for the life of the process.
_pool: _pgpool.ThreadedConnectionPool | None = None


def init_pool(retries: int = 10, delay_seconds: float = 2.0) -> None:
    """Create the connection pool, retrying while Postgres is still
    starting up (relevant on `docker-compose up` even with a
    healthcheck gate, since the gate only proves the server accepts
    TCP connections, not that this app's user/db are ready)."""
    global _pool
    if _pool is not None:
        return
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            _pool = _pgpool.ThreadedConnectionPool(DB_POOL_MIN, DB_POOL_MAX, PG_DSN)
            logger.info(f"[DB] Connection pool ready ({DB_POOL_MIN}-{DB_POOL_MAX}) on attempt {attempt}")
            return
        except Exception as e:
            last_err = e
            logger.warning(f"[DB] pool init attempt {attempt}/{retries} failed: {e}")
            time.sleep(delay_seconds)
    raise RuntimeError(f"Could not establish DB connection pool after {retries} attempts") from last_err


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None
        logger.info("[DB] Connection pool closed")


def pool_ready() -> bool:
    """Cheap check used by /ready — borrows and returns a connection."""
    if _pool is None:
        return False
    conn = None
    try:
        conn = _pool.getconn()
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
        return True
    except Exception as e:
        logger.warning(f"[DB] readiness check failed: {e}")
        return False
    finally:
        if conn is not None:
            _pool.putconn(conn)


@contextmanager
def get_db():
    if _pool is None:
        # Fallback for scripts/tests that import db.py without going
        # through the FastAPI startup event.
        init_pool()
    conn = _pool.getconn()
    conn.autocommit = False
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)


def init_db() -> None:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS documents (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            filename        TEXT NOT NULL,
            file_type       TEXT,
            chunk_count     INTEGER DEFAULT 0,
            has_tables      BOOLEAN DEFAULT FALSE,
            content_type    TEXT DEFAULT 'general',
            custom_metadata JSONB DEFAULT '{{}}',
            created_at      TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            chunk_id    TEXT UNIQUE NOT NULL,
            doc_id      UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            filename    TEXT,
            text        TEXT NOT NULL,
            chunk_index INTEGER,
            chunk_type  TEXT DEFAULT 'text',
            metadata    JSONB DEFAULT '{{}}',
            embedding   vector({EMBEDDING_DIM})
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            doc_id     UUID NOT NULL REFERENCES documents(id),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS messages (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id  UUID NOT NULL REFERENCES sessions(id),
            role        TEXT NOT NULL,
            content     TEXT NOT NULL,
            sources     JSONB DEFAULT '[]',
            evaluation  JSONB DEFAULT '{{}}',
            query_type  TEXT DEFAULT '',
            created_at  TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS memory_summaries (
            id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id   UUID NOT NULL REFERENCES sessions(id),
            summary      TEXT NOT NULL,
            covers_up_to INTEGER NOT NULL,
            created_at   TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS memory_facts (
            id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id UUID NOT NULL REFERENCES sessions(id),
            fact       TEXT NOT NULL,
            relevance  TEXT DEFAULT '',
            embedding  vector({EMBEDDING_DIM}),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS golden_test_data (
            id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            doc_id     UUID NOT NULL REFERENCES documents(id),
            question   TEXT NOT NULL,
            expected   TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS batch_eval_results (
            id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            doc_id            UUID NOT NULL,
            question          TEXT NOT NULL,
            answer            TEXT NOT NULL,
            faithfulness      REAL,
            answer_relevance  REAL,
            context_relevance REAL,
            passed            BOOLEAN DEFAULT FALSE,
            run_at            TIMESTAMPTZ DEFAULT NOW()
        );

        -- ── OPTIMIZATION 1: Embedding cache ──────────────────────
        CREATE TABLE IF NOT EXISTS embedding_cache (
            text_hash    TEXT PRIMARY KEY,
            text_preview TEXT,
            embedding    TEXT NOT NULL,
            hit_count    INTEGER DEFAULT 0,
            created_at   TIMESTAMPTZ DEFAULT NOW(),
            last_used_at TIMESTAMPTZ DEFAULT NOW()
        );

        -- ── OPTIMIZATION 6: Response cache ───────────────────────
        CREATE TABLE IF NOT EXISTS response_cache (
            id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            query_hash   TEXT NOT NULL,
            doc_id       TEXT NOT NULL,
            answer       TEXT NOT NULL,
            sources      JSONB DEFAULT '[]',
            evaluation   JSONB DEFAULT '{{}}',
            hit_count    INTEGER DEFAULT 0,
            created_at   TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE (query_hash, doc_id)
        );

        -- ── Cost tracking ─────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS api_usage (
            id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id       TEXT,
            query_hash       TEXT,
            embedding_calls  INTEGER DEFAULT 0,
            embedding_cached INTEGER DEFAULT 0,
            llm_calls        INTEGER DEFAULT 0,
            prompt_tokens    INTEGER DEFAULT 0,
            cached_tokens    INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0,
            estimated_cost   REAL DEFAULT 0,
            cache_hit        BOOLEAN DEFAULT FALSE,
            created_at       TIMESTAMPTZ DEFAULT NOW()
        );
        """)

        # pgvector HNSW indexes
        cur.execute("""
        CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
        ON chunks USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """)
        cur.execute("""
        CREATE INDEX IF NOT EXISTS memory_facts_embedding_hnsw_idx
        ON memory_facts USING hnsw (embedding vector_cosine_ops)
        WITH (m = 8, ef_construction = 32);
        """)
        cur.execute("""
        CREATE INDEX IF NOT EXISTS response_cache_lookup_idx
        ON response_cache (query_hash, doc_id);
        """)
        cur.execute("""
        CREATE INDEX IF NOT EXISTS embedding_cache_hash_idx
        ON embedding_cache (text_hash);
        """)
        conn.commit()
        print("✅ DB schema v4 initialised.")


# ── Documents ─────────────────────────────────────────────────────

def save_document(filename, file_type, has_tables=False,
                  content_type="general", custom_metadata=None) -> str:
    doc_id = str(uuid.uuid4())
    with get_db() as conn:
        conn.cursor().execute(
            """INSERT INTO documents (id, filename, file_type, has_tables, content_type, custom_metadata)
               VALUES (%s,%s,%s,%s,%s,%s)""",
            (doc_id, filename, file_type, has_tables, content_type,
             json.dumps(custom_metadata or {}))
        )
    return doc_id


def update_document_after_indexing(doc_id, chunk_count, has_tables,
                                   content_type, custom_metadata) -> None:
    with get_db() as conn:
        conn.cursor().execute(
            """UPDATE documents SET chunk_count=%s, has_tables=%s,
               content_type=%s, custom_metadata=%s WHERE id=%s""",
            (chunk_count, has_tables, content_type,
             json.dumps(custom_metadata), doc_id)
        )


def get_all_documents() -> list:
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM documents ORDER BY created_at DESC")
        return [dict(r) for r in cur.fetchall()]


def get_document(doc_id: str) -> dict | None:
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM documents WHERE id=%s", (doc_id,))
        row = cur.fetchone()
    return dict(row) if row else None


# ── Chunks ────────────────────────────────────────────────────────

def save_chunks(chunks: list[dict], embeddings: list[list[float]]) -> None:
    records = []
    for chunk, embedding in zip(chunks, embeddings):
        records.append((
            str(uuid.uuid4()), chunk["chunk_id"], chunk["doc_id"],
            chunk["filename"], chunk["text"], chunk["chunk_index"],
            chunk["chunk_type"], json.dumps(chunk.get("metadata", {})),
            embedding,
        ))
    with get_db() as conn:
        psycopg2.extras.execute_values(
            conn.cursor(),
            """INSERT INTO chunks
               (id, chunk_id, doc_id, filename, text, chunk_index, chunk_type, metadata, embedding)
               VALUES %s ON CONFLICT (chunk_id) DO NOTHING""",
            records,
            template="(%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)",
        )


def vector_search(query_embedding, doc_ids, top_k=10,
                  metadata_filters=None) -> list[dict]:
    filters = ["doc_id = ANY(%s)"]
    params: list = [doc_ids]
    if metadata_filters:
        if metadata_filters.get("content_type"):
            filters.append("metadata->>'content_type' = %s")
            params.append(metadata_filters["content_type"])
        if metadata_filters.get("has_tables") is not None:
            filters.append("(metadata->>'has_tables')::boolean = %s")
            params.append(metadata_filters["has_tables"])
        if metadata_filters.get("keywords"):
            kw_conds = " OR ".join(["text ILIKE %s"] * len(metadata_filters["keywords"]))
            filters.append(f"({kw_conds})")
            for kw in metadata_filters["keywords"]:
                params.append(f"%{kw}%")
    where_clause   = " AND ".join(filters)
    filter_params  = params
    final_params   = filter_params + [query_embedding, query_embedding, top_k]
    sql = f"""
        SELECT chunk_id, doc_id, filename, text, chunk_index, chunk_type, metadata,
               1 - (embedding <=> %s::vector) AS score
        FROM chunks
        WHERE {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(sql, final_params)
        rows = cur.fetchall()
    return [
        {
            "chunk_id":    r["chunk_id"],
            "doc_id":      str(r["doc_id"]),
            "filename":    r["filename"],
            "text":        r["text"],
            "chunk_index": r["chunk_index"],
            "chunk_type":  r["chunk_type"],
            "metadata":    r["metadata"] if isinstance(r["metadata"], dict) else json.loads(r["metadata"]),
            "score":       float(r["score"]),
        }
        for r in rows
    ]


# ── Sessions & Messages ───────────────────────────────────────────

def create_session(doc_id: str) -> str:
    session_id = str(uuid.uuid4())
    with get_db() as conn:
        conn.cursor().execute(
            "INSERT INTO sessions (id, doc_id) VALUES (%s,%s)",
            (session_id, doc_id)
        )
    return session_id


def save_message(session_id, role, content,
                 sources=None, evaluation=None, query_type="") -> str:
    msg_id = str(uuid.uuid4())
    with get_db() as conn:
        conn.cursor().execute(
            """INSERT INTO messages (id, session_id, role, content, sources, evaluation, query_type)
               VALUES (%s,%s,%s,%s,%s,%s,%s)""",
            (msg_id, session_id, role, content,
             json.dumps(sources or []),
             json.dumps(evaluation or {}),
             query_type)
        )
    return msg_id


def get_chat_history(session_id: str, limit: int = 20) -> list[dict]:
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """SELECT role, content FROM messages
               WHERE session_id=%s ORDER BY created_at DESC LIMIT %s""",
            (session_id, limit)
        )
        rows = cur.fetchall()
    return list(reversed([{"role": r["role"], "content": r["content"]} for r in rows]))


def count_messages(session_id: str) -> int:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM messages WHERE session_id=%s", (session_id,))
        return cur.fetchone()[0]


# ── Memory ────────────────────────────────────────────────────────

def save_memory_summary(session_id, summary, covers_up_to) -> None:
    with get_db() as conn:
        conn.cursor().execute(
            "INSERT INTO memory_summaries (session_id, summary, covers_up_to) VALUES (%s,%s,%s)",
            (session_id, summary, covers_up_to)
        )


def get_latest_summary(session_id) -> dict | None:
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT * FROM memory_summaries WHERE session_id=%s ORDER BY created_at DESC LIMIT 1",
            (session_id,)
        )
        row = cur.fetchone()
    return dict(row) if row else None


def save_memory_facts(session_id, facts, embeddings) -> None:
    records = [
        (str(uuid.uuid4()), session_id, f["fact"], f.get("relevance", ""), emb)
        for f, emb in zip(facts, embeddings)
    ]
    with get_db() as conn:
        psycopg2.extras.execute_values(
            conn.cursor(),
            "INSERT INTO memory_facts (id, session_id, fact, relevance, embedding) VALUES %s",
            records,
            template="(%s,%s,%s,%s,%s::vector)",
        )


def search_memory_facts(session_id, query_embedding, top_k=5) -> list[str]:
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """SELECT fact FROM memory_facts WHERE session_id=%s
               ORDER BY embedding <=> %s::vector LIMIT %s""",
            (session_id, query_embedding, top_k)
        )
        return [r["fact"] for r in cur.fetchall()]


# ── OPTIMIZATION 1: Embedding Cache ──────────────────────────────

def get_cached_embedding(text_hash: str) -> list[float] | None:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT embedding FROM embedding_cache WHERE text_hash=%s",
            (text_hash,)
        )
        row = cur.fetchone()
        if row:
            # Update hit count + last used
            conn.cursor().execute(
                """UPDATE embedding_cache
                   SET hit_count=hit_count+1, last_used_at=NOW()
                   WHERE text_hash=%s""",
                (text_hash,)
            )
            return json.loads(row[0])
    return None


def set_cached_embedding(text_hash: str, text_preview: str,
                         embedding: list[float]) -> None:
    with get_db() as conn:
        conn.cursor().execute(
            """INSERT INTO embedding_cache (text_hash, text_preview, embedding)
               VALUES (%s,%s,%s) ON CONFLICT (text_hash) DO NOTHING""",
            (text_hash, text_preview[:100], json.dumps(embedding))
        )


def bulk_get_cached_embeddings(text_hashes: list[str]) -> dict[str, list[float]]:
    """Single DB query to fetch multiple cached embeddings at once."""
    if not text_hashes:
        return {}
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT text_hash, embedding FROM embedding_cache WHERE text_hash = ANY(%s)",
            (text_hashes,)
        )
        result = {row[0]: json.loads(row[1]) for row in cur.fetchall()}
        if result:
            conn.cursor().execute(
                """UPDATE embedding_cache SET hit_count=hit_count+1, last_used_at=NOW()
                   WHERE text_hash = ANY(%s)""",
                (list(result.keys()),)
            )
    return result


def bulk_set_cached_embeddings(items: list[tuple[str, str, list[float]]]) -> None:
    """Bulk insert embeddings into cache."""
    if not items:
        return
    records = [(h, preview[:100], json.dumps(emb)) for h, preview, emb in items]
    with get_db() as conn:
        psycopg2.extras.execute_values(
            conn.cursor(),
            """INSERT INTO embedding_cache (text_hash, text_preview, embedding)
               VALUES %s ON CONFLICT (text_hash) DO NOTHING""",
            records,
        )


# ── OPTIMIZATION 6: Response Cache ───────────────────────────────

def get_cached_response(query_hash: str, doc_id: str,
                         ttl_hours: int = 24) -> dict | None:
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """SELECT answer, sources, evaluation FROM response_cache
               WHERE query_hash=%s AND doc_id=%s
               AND created_at > NOW() - INTERVAL '%s hours'""",
            (query_hash, doc_id, ttl_hours)
        )
        row = cur.fetchone()
        if row:
            conn.cursor().execute(
                "UPDATE response_cache SET hit_count=hit_count+1 WHERE query_hash=%s AND doc_id=%s",
                (query_hash, doc_id)
            )
            return dict(row)
    return None


def set_cached_response(query_hash: str, doc_id: str, answer: str,
                         sources: list, evaluation: dict) -> None:
    with get_db() as conn:
        conn.cursor().execute(
            """INSERT INTO response_cache (query_hash, doc_id, answer, sources, evaluation)
               VALUES (%s,%s,%s,%s,%s)
               ON CONFLICT (query_hash, doc_id) DO UPDATE
               SET answer=%s, sources=%s, evaluation=%s, created_at=NOW()""",
            (query_hash, doc_id, answer, json.dumps(sources), json.dumps(evaluation),
             answer, json.dumps(sources), json.dumps(evaluation))
        )


# ── Cost Tracking ─────────────────────────────────────────────────

def log_api_usage(session_id: str, query_hash: str, embedding_calls: int,
                  embedding_cached: int, llm_calls: int, prompt_tokens: int,
                  cached_tokens: int, completion_tokens: int,
                  estimated_cost: float, cache_hit: bool) -> None:
    with get_db() as conn:
        conn.cursor().execute(
            """INSERT INTO api_usage
               (session_id, query_hash, embedding_calls, embedding_cached,
                llm_calls, prompt_tokens, cached_tokens, completion_tokens,
                estimated_cost, cache_hit)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (session_id, query_hash, embedding_calls, embedding_cached,
             llm_calls, prompt_tokens, cached_tokens, completion_tokens,
             estimated_cost, cache_hit)
        )


def get_usage_stats() -> dict:
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT
                COUNT(*)                           AS total_queries,
                SUM(embedding_calls)               AS total_embedding_calls,
                SUM(embedding_cached)              AS total_embedding_cached,
                SUM(llm_calls)                     AS total_llm_calls,
                SUM(prompt_tokens)                 AS total_prompt_tokens,
                SUM(cached_tokens)                 AS total_cached_tokens,
                SUM(completion_tokens)             AS total_completion_tokens,
                ROUND(SUM(estimated_cost)::numeric, 4) AS total_cost,
                ROUND(AVG(estimated_cost)::numeric, 6) AS avg_cost_per_query,
                SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) AS response_cache_hits
            FROM api_usage
        """)
        row = cur.fetchone()
    return dict(row) if row else {}


def get_cache_stats() -> dict:
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT
                COUNT(*)       AS total_entries,
                SUM(hit_count) AS total_hits
            FROM embedding_cache
        """)
        emb = dict(cur.fetchone() or {})
        cur.execute("""
            SELECT
                COUNT(*)       AS total_entries,
                SUM(hit_count) AS total_hits
            FROM response_cache
        """)
        resp = dict(cur.fetchone() or {})
    return {"embedding_cache": emb, "response_cache": resp}


# ── Golden & Eval ─────────────────────────────────────────────────

def save_golden_sample(doc_id, question, expected) -> str:
    gid = str(uuid.uuid4())
    with get_db() as conn:
        conn.cursor().execute(
            "INSERT INTO golden_test_data (id, doc_id, question, expected) VALUES (%s,%s,%s,%s)",
            (gid, doc_id, question, expected)
        )
    return gid


def get_golden_data(doc_id: str) -> list[dict]:
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM golden_test_data WHERE doc_id=%s", (doc_id,))
        return [dict(r) for r in cur.fetchall()]


def save_batch_eval(doc_id, question, answer, faithfulness,
                    answer_relevance, context_relevance, passed) -> None:
    with get_db() as conn:
        conn.cursor().execute(
            """INSERT INTO batch_eval_results
               (id, doc_id, question, answer, faithfulness, answer_relevance, context_relevance, passed)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
            (str(uuid.uuid4()), doc_id, question, answer,
             faithfulness, answer_relevance, context_relevance, passed)
        )


def get_batch_eval_results(doc_id: str) -> list[dict]:
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT * FROM batch_eval_results WHERE doc_id=%s ORDER BY run_at DESC",
            (doc_id,)
        )
        return [dict(r) for r in cur.fetchall()]
