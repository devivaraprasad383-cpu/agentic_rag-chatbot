"""
Central configuration — pgvector + PostgreSQL + Memory-Aware RAG.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI ────────────────────────────────────────────────────────
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
LLM_LARGE       = "gpt-4o"           # answer generator, table enrichment
LLM_SMALL       = "gpt-4o-mini"      # query optimizer, source id, post-processor, memory
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM   = 1536

# ── PostgreSQL + pgvector ─────────────────────────────────────────
PG_HOST     = os.getenv("PG_HOST",     "localhost")
PG_PORT     = int(os.getenv("PG_PORT", "5432"))
PG_DB       = os.getenv("PG_DB",       "ragdb")
PG_USER     = os.getenv("PG_USER",     "raguser")
PG_PASSWORD = os.getenv("PG_PASSWORD", "ragpassword")
PG_DSN      = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
DB_POOL_MIN = int(os.getenv("DB_POOL_MIN", "2"))
DB_POOL_MAX = int(os.getenv("DB_POOL_MAX", "20"))

# ── Deployment / safety-net settings ──────────────────────────────
ENVIRONMENT      = os.getenv("ENVIRONMENT", "development")  # development | production
CORS_ORIGINS     = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
MAX_UPLOAD_MB    = int(os.getenv("MAX_UPLOAD_MB", "25"))
RATE_LIMIT_CHAT  = os.getenv("RATE_LIMIT_CHAT",   "20/minute")   # per-IP, slowapi syntax
RATE_LIMIT_UPLOAD = os.getenv("RATE_LIMIT_UPLOAD", "5/minute")
API_KEY          = os.getenv("API_KEY", "")  # if set, required as `Authorization: Bearer <key>`

if ENVIRONMENT == "production" and PG_PASSWORD == "ragpassword":
    import logging
    logging.getLogger(__name__).warning(
        "[Config] ENVIRONMENT=production but PG_PASSWORD is still the docker-compose "
        "default ('ragpassword'). Set a real password before exposing this publicly."
    )

# ── Chunking ──────────────────────────────────────────────────────
CHUNK_SIZE       = 512
CHUNK_OVERLAP    = 64
TABLE_CHUNK_SIZE = 1024

# ── Retrieval ─────────────────────────────────────────────────────
TOP_K              = 10
POST_PROCESS_TOP_K = 5
RRF_K              = 60

# ── Memory ────────────────────────────────────────────────────────
MEMORY_WINDOW       = 6     # last N messages always included verbatim
SUMMARY_THRESHOLD   = 10    # summarise older messages when history > this
MAX_FACTS_PER_SESSION = 20  # max key facts stored per session

# ── Storage ───────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "data")
BM25_INDEX_DIR = os.path.join(DATA_DIR, "bm25_indexes")   # BM25 still file-based

for _d in [DATA_DIR, BM25_INDEX_DIR]:
    os.makedirs(_d, exist_ok=True)
