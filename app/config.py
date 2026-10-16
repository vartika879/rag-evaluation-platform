
import os
from pathlib import Path
 
from dotenv import load_dotenv
 
load_dotenv()
 
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def _env_str(name, default):
    value = os.getenv(name)
    return value if value not in (None, "") else default
 
 
def _env_int(name, default):
    value = os.getenv(name)
    return int(value) if value not in (None, "") else default
 
 
def _env_float(name, default):
    value = os.getenv(name)
    return float(value) if value not in (None, "") else default
 
 
def _env_bool(name, default):
    value = os.getenv(name)
 
    if value in (None, ""):
        return default
 
    return value.strip().lower() in ("1", "true", "yes", "on")
 
 
# ---------------------------------------------------------------------------
# Paths  (app/config.py -> project root is one level above app/)
# ---------------------------------------------------------------------------
 
PROJECT_ROOT = Path(__file__).resolve().parents[1]
 
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
TRACES_DIR = Path(_env_str("TRACES_DIR", str(DATA_DIR / "traces")))
 
# Qdrant local database. Its internal meta.json lives inside this folder:
# never edit that file by hand.
QDRANT_PATH = Path(_env_str("QDRANT_PATH", str(DATA_DIR / "qdrant")))

# Cloud Qdrant (if set, overrides local path)
QDRANT_URL = _env_str("QDRANT_URL", "")
QDRANT_API_KEY = _env_str("QDRANT_API_KEY", "")
# Our own collection registry (strategy -> collection, chunk counts, ...)
REGISTRY_PATH = Path(_env_str("REGISTRY_PATH", str(DATA_DIR / "registry.json")))
 
# Evaluation data (separate from app/evaluation, which holds the code)
EVAL_DIR = PROJECT_ROOT / "evaluation"
DATASET_PATH = Path(_env_str("DATASET_PATH", str(EVAL_DIR / "dataset.json")))
MANUAL_LABELS_PATH = EVAL_DIR / "manual_labels.json"
RUNS_DIR = EVAL_DIR / "runs"
 
 
# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
 
EMBEDDING_MODEL = _env_str("EMBEDDING_MODEL", "embed-english-v3.0")
EMBEDDING_DIM = _env_int("EMBEDDING_DIM", 1024)   # must match EMBEDDING_MODEL
EMBEDDING_BATCH_SIZE = 96                         # Cohere limit per call
 
RERANK_MODEL = _env_str("RERANK_MODEL", "rerank-v3.5")


LLM_MODEL = _env_str("GROQ_MODEL", "openai/gpt-oss-20b")
LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.0)

# Judge provider is independent of the answer-generation provider.
JUDGE_PROVIDER = _env_str("JUDGE_PROVIDER", "groq").strip().lower()
JUDGE_MODEL = _env_str("JUDGE_MODEL", LLM_MODEL)

# ---------------------------------------------------------------------------
# Ingestion / chunking defaults
# ---------------------------------------------------------------------------
 
DEFAULT_CHUNK_STRATEGY = _env_str("DEFAULT_CHUNK_STRATEGY", "recursive")
DEFAULT_CHUNK_SIZE = _env_int("DEFAULT_CHUNK_SIZE", 1000)
DEFAULT_CHUNK_OVERLAP = _env_int("DEFAULT_CHUNK_OVERLAP", 150)
 
# Per-strategy defaults: (chunk_size, chunk_overlap). Semantic has no size.
CHUNK_DEFAULTS = {
    "recursive": (1000, 150),
    "fixed": (1000, 150),
    "structure_aware": (1500, 150),
    "semantic": (None, None),
}
 
 
# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
 
# During the transition the old collection name is kept. Later the registry
# alias "default" decides which collection is used.
DEFAULT_COLLECTION = _env_str("DEFAULT_COLLECTION", "rag_documents")
UPSERT_BATCH_SIZE = 100
 
 
# ---------------------------------------------------------------------------
# Retrieval defaults
# ---------------------------------------------------------------------------
 
DEFAULT_TOP_K = _env_int("DEFAULT_TOP_K", 5)
MAX_TOP_K = _env_int("MAX_TOP_K", 50)
 
# How many candidates to fetch from each retriever before fusing / reranking.
# Must be larger than top_k, otherwise fusion has nothing to choose from.
CANDIDATE_POOL = _env_int("CANDIDATE_POOL", 30)
 
HYBRID_VECTOR_WEIGHT = _env_float("HYBRID_VECTOR_WEIGHT", 0.5)
RRF_K = 60                                     # Reciprocal Rank Fusion constant
 
USE_RERANKER = _env_bool("USE_RERANKER", False)
RERANK_TOP_K = _env_int("RERANK_TOP_K", 3)
 
# Chunks scoring below this are dropped before the LLM (None = disabled).
# Calibrate on the golden set before enabling.
MIN_SCORE_THRESHOLD = None
 
 
# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
 
REFUSAL_MESSAGE = "I don't have enough information in the provided documents."
MAX_CONTEXT_CHARS = _env_int("MAX_CONTEXT_CHARS", 6000)
 
 
# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
 
EVAL_TOP_K = _env_int("EVAL_TOP_K", 5)
EVAL_K_VALUES = (1, 3, 5, 10)                  # Hit@k / Recall@k reported
 
# A single answer counts as "faithful" / "correct" at or above these scores.
FAITHFULNESS_PASS = _env_float("FAITHFULNESS_PASS", 0.7)
CORRECTNESS_PASS = _env_float("CORRECTNESS_PASS", 0.7)
 
# Pause between LLM-judge calls to stay under free-tier rate limits.
JUDGE_DELAY_SECONDS = _env_float("JUDGE_DELAY_SECONDS", 0.0)
 
 
# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------
 
TRACE_ENABLED = _env_bool("TRACE_ENABLED", True)
TRACE_MAX_CONTEXT_CHARS = 20000                # cap stored prompt size
 
# LangSmith is optional. It reads these variables itself; they are listed
# here only so validate_env() can warn when tracing is on without a key.
LANGSMITH_TRACING = _env_bool("LANGSMITH_TRACING", False)
 
 
# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
 
API_HOST = _env_str("API_HOST", "127.0.0.1")
API_PORT = _env_int("API_PORT", 8000)
 
 
# ---------------------------------------------------------------------------
# Environment checks
# ---------------------------------------------------------------------------
 
REQUIRED_KEYS = ("COHERE_API_KEY", "GROQ_API_KEY")
 
 
def get_api_key(name):
    value = os.getenv(name)
 
    if not value:
        raise ValueError(f"{name} not found in environment configuration (.env).")
 
    return value
 
 
def validate_env(required=REQUIRED_KEYS):
    """Raise one clear error listing every missing key, instead of failing
    later inside a library call."""
    missing = [key for key in required if not os.getenv(key)]
 
    if missing:
        raise ValueError(
            "Missing environment variables: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill them in."
        )
 
    if LANGSMITH_TRACING and not os.getenv("LANGSMITH_API_KEY"):
        raise ValueError(
            "LANGSMITH_TRACING is on but LANGSMITH_API_KEY is not set."
        )
 
    if not 0 <= HYBRID_VECTOR_WEIGHT <= 1:
        raise ValueError("HYBRID_VECTOR_WEIGHT must be between 0 and 1.")
 
    if CANDIDATE_POOL < DEFAULT_TOP_K:
        raise ValueError("CANDIDATE_POOL must be >= DEFAULT_TOP_K.")
 
 
def ensure_dirs():
    for directory in (DOCUMENTS_DIR, TRACES_DIR, RUNS_DIR):
        Path(directory).mkdir(parents=True, exist_ok=True)
 