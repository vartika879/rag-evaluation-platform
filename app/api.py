import logging
import hmac
import os
from typing import Optional

from fastapi import FastAPI, HTTPException, APIRouter, Depends, Header
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from app.config import ensure_dirs, validate_env
from app.schemas import AskRequest, AskResponse
from app.services.rag_service import answer_question
from pydantic import BaseModel, Field
from typing import List as ListType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Evaluation & Debugging Platform",
    version="1.0.0",
    description="Document-grounded question answering and debugging API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



def require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    """Require a configured shared secret for application endpoints.

    /health remains public for hosting-platform health checks.
    Set API_ACCESS_TOKEN in the API environment and RAG_API_TOKEN in the UI
    environment to the same long, random value.
    """
    expected = os.getenv("API_ACCESS_TOKEN", "").strip()
    if not expected:
        raise HTTPException(503, "API authentication is not configured.")
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(401, "Missing or invalid API key.")


@app.on_event("startup")
def _startup():
    validate_env()
    ensure_dirs()


# ===========================================================================
# Core endpoints
# ===========================================================================

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
async def ask_question(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    logger.info("RAG request received; top_k=%s", request.top_k)

    try:
        result = await run_in_threadpool(
            answer_question,
            request.question,
            request.top_k,
            None,        
            None,        
            None,        
            request.preset, 
        )
        logger.info(
            "RAG request completed; retrieved=%s refused=%s",
            result["retrieved_count"],
            result["refused"],
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception("RAG request failed")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the question.",
        )


# ===========================================================================
# /api/* — read-only data for the dashboard
# ===========================================================================

api = APIRouter(prefix="/api", tags=["dashboard"], dependencies=[Depends(require_api_key)])


@api.get("/stats/overview")
def stats_overview():
    """Dashboard cards: runs, traces, refusal rate, latency."""
    from app.evaluation.runner import list_runs
    from app.tracing.trace_store import get_store

    store = get_store()
    stats = store.stats()
    runs = list_runs()

    traces = stats.get("traces", 0)
    refused = stats.get("refused", 0)

    return {
        "total_runs": len(runs),
        "total_traces": traces,
        "refusal_rate": (refused / traces) if traces else 0.0,
        "avg_latency_ms": stats.get("avg_latency_ms", 0),
        "p50_latency_ms": stats.get("p50_latency_ms", 0),
        "p95_latency_ms": stats.get("p95_latency_ms", 0),
        "avg_tokens": stats.get("avg_tokens", 0),
        "with_errors": stats.get("with_errors", 0),
        "by_action": stats.get("by_action", {}),
    }


@api.get("/collections")
def list_collections():
    """Qdrant collections with registry metadata."""
    from app.vectorstore.qdrant_store import get_qdrant_client, list_collection_names
    from app.vectorstore.registry import list_registered

    try:
        client = get_qdrant_client()
        names = list_collection_names(client)
        registered = list_registered()
    except Exception as exc:
        logger.exception("Failed to read collections")
        raise HTTPException(500, f"Cannot read collections: {exc}") from exc

    collections = []
    for name in names:
        try:
            info = client.get_collection(name)
            points = info.points_count
        except Exception:
            points = None

        reg = registered.get(name, {})

        collections.append({
            "name": name,
            "points_count": points,
            "strategy": reg.get("strategy"),
            "chunk_size": reg.get("chunk_size"),
            "chunk_overlap": reg.get("chunk_overlap"),
            "embedding_model": reg.get("embedding_model"),
            "created_at": reg.get("created_at"),
            "documents": list((reg.get("documents") or {}).keys()),
        })

    return {"collections": collections}


@api.get("/runs")
def list_runs_endpoint(limit: int = 20):
    """Recent evaluation runs (newest first)."""
    from app.evaluation.runner import list_runs

    try:
        runs = list_runs()
    except Exception as exc:
        logger.exception("Failed to list runs")
        raise HTTPException(500, f"Cannot list runs: {exc}") from exc

    return {"runs": runs[:limit], "total": len(runs)}


@api.get("/runs/{run_id}")
def get_run_detail(run_id: str):
    """Full run file (rows + summary)."""
    from app.evaluation.runner import load_run

    try:
        return load_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    methods: ListType[str] = Field(default_factory=lambda: ["vector", "bm25", "hybrid"], max_length=3)
    top_k: int = Field(default=5, ge=1, le=20)
    collection_name: Optional[str] = Field(default=None, min_length=1, max_length=128)


def _serialize_chunk(rank, c):
    return {
        "rank": rank,
        "chunk_id": c.chunk_id,
        "page": c.page_number,
        "source": c.source,
        "score": c.score,
        "method": c.method,
        "vector_score": c.vector_score,
        "bm25_score": c.bm25_score,
        "hybrid_score": c.hybrid_score,
        "text": (c.text or "")[:600],
    }

class ChunkingRequest(BaseModel):
    num_pages: int = Field(default=10, ge=1, le=30)
    strategies: ListType[str] = Field(default_factory=lambda: ["recursive", "fixed", "structure_aware"], min_length=1, max_length=4)


@api.post("/chunking/compare")
def api_chunking_compare(payload: ChunkingRequest):
    """Run several chunking strategies on the same PDF pages and compare."""
    from app.ingestion.loader import load_pdf
    from app.ingestion.chunker import create_chunks
    from app.config import DOCUMENTS_DIR

    pdf_path = DOCUMENTS_DIR / "rag_survey.pdf"
    if not pdf_path.exists():
        raise HTTPException(404, f"PDF not found at {pdf_path}")

    try:
        documents = load_pdf(str(pdf_path))
    except Exception as exc:
        raise HTTPException(500, f"Failed to load PDF: {exc}") from exc

    num_pages = max(1, min(payload.num_pages, len(documents)))
    subset = documents[:num_pages]

    results = {}
    for strategy in payload.strategies:
        try:
            chunks = create_chunks(subset, strategy)
            lengths = [len(c.page_content) for c in chunks]

            samples = []
            for i, c in enumerate(chunks[:3]):
                samples.append({
                    "index": i,
                    "length": len(c.page_content),
                    "page": c.metadata.get("page_number"),
                    "text": c.page_content[:800],
                })

            results[strategy] = {
                "count": len(chunks),
                "avg_chars": round(sum(lengths) / len(lengths)) if lengths else 0,
                "min_chars": min(lengths) if lengths else 0,
                "max_chars": max(lengths) if lengths else 0,
                "samples": samples,
            }
        except Exception as exc:
            logger.exception("Chunking failed for %s", strategy)
            results[strategy] = {"error": f"{type(exc).__name__}: {exc}"}

    return {
        "pdf": pdf_path.name,
        "num_pages_used": num_pages,
        "total_pages": len(documents),
        "results": results,

    }

class EvalRunRequest(BaseModel):
    preset: str = Field(default="vector", min_length=1, max_length=64)
    limit: Optional[int] = Field(default=None, ge=1, le=60)
    use_judge: bool = True


@api.post("/evaluation/run")
def api_run_evaluation(payload: EvalRunRequest):
    """Run an evaluation synchronously and return the run summary."""
    from app.evaluation.runner import run_evaluation
    from app.evaluation.dataset import load_dataset
    from app.pipeline_config import get_preset

    try:
        config = get_preset(payload.preset)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    dataset = load_dataset()
    if payload.limit is not None:
        # Limit the dataset deterministically for controlled, smaller runs.
        dataset = dataset.filter(ids=[q.id for q in dataset][:payload.limit])

    try:
        run = run_evaluation(
            config=config,
            dataset=dataset,
            use_judge=payload.use_judge,
            check_collection=True,
        )

        summary = run.get("summary") or {}
        e2e = summary.get("end_to_end") or {}
        retrieval = summary.get("retrieval", {}).get("all", {})

        return {
            "status": "ok",
            "run_id": run["run_id"],
            "label": run.get("label"),
            "questions": len(run.get("rows", [])),
            "hit@5": retrieval.get("hit@5"),
            "mrr": retrieval.get("mrr"),
            "answerable_ok_rate": e2e.get("answerable_ok_rate"),
            "unanswerable_ok_rate": e2e.get("unanswerable_ok_rate"),
            "diagnosis": {
                k: v["count"]
                for k, v in (e2e.get("diagnosis") or {}).items()
                if v["count"] > 0
            },
        }
    except Exception as exc:
        logger.exception("Evaluation failed")
        raise HTTPException(500, f"Evaluation failed: {exc}") from exc

    

class IngestRequest(BaseModel):
    strategy: str = Field(default="recursive", min_length=1, max_length=32)
    chunk_size: int = Field(default=1000, ge=100, le=3000)
    chunk_overlap: int = Field(default=150, ge=0, le=500)
    recreate: bool = False


@api.post("/ingest")
def api_ingest(payload: IngestRequest):
    """Ingest the bundled PDF with the chosen chunking strategy."""
    from app.ingestion.ingest import ingest_document
    from app.config import DOCUMENTS_DIR

    pdf_path = DOCUMENTS_DIR / "rag_survey.pdf"
    if not pdf_path.exists():
        raise HTTPException(404, f"PDF not found at {pdf_path}")

    valid = ("recursive", "fixed", "structure_aware", "semantic")
    if payload.strategy not in valid:
        raise HTTPException(400, f"Strategy must be one of {valid}")

    try:
        summary = ingest_document(
            pdf_path,
            strategy=payload.strategy,
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
            recreate=payload.recreate,
        )
        return {"status": "ok", "summary": summary}
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(500, f"Ingestion failed: {exc}") from exc

@api.get("/collections/{name}/chunks")
def api_collection_chunks(name: str, limit: int = 20, offset: int = 0):
    """Browse chunks in a collection."""
    from app.vectorstore.qdrant_store import get_qdrant_client

    try:
        client = get_qdrant_client()
        info = client.get_collection(name)

        points, _ = client.scroll(
            collection_name=name,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        chunks = []
        for p in points:
            payload = p.payload or {}
            chunks.append({
                "chunk_id": payload.get("chunk_id"),
                "page": payload.get("page_number") or payload.get("page"),
                "source": payload.get("source"),
                "section": payload.get("section"),
                "strategy": payload.get("chunk_strategy"),
                "text": (payload.get("text") or "")[:800],
            })

        return {
            "collection": name,
            "total_points": info.points_count,
            "chunks": chunks,
            "limit": limit,
            "offset": offset,
        }
    except Exception as exc:
        logger.exception("Failed to fetch chunks")
        raise HTTPException(500, f"Cannot fetch chunks: {exc}") from exc

    
@api.post("/retrieve")
def api_retrieve(payload: RetrieveRequest):
    """Run one or more retrieval methods on the same query."""
    from app.retrieval.retriever import retrieve_chunks
    from app.retrieval.bm25 import retrieve_bm25
    from app.retrieval.hybrid import retrieve_hybrid

    query = (payload.query or "").strip()
    if not query:
        raise HTTPException(400, "Query cannot be empty.")

    results = {}

    for method in payload.methods:
        try:
            if method == "vector":
                chunks = retrieve_chunks(
                    query, top_k=payload.top_k,
                    collection_name=payload.collection_name,
                )
            elif method == "bm25":
                chunks = retrieve_bm25(
                    query, top_k=payload.top_k,
                    collection_name=payload.collection_name,
                )
            elif method == "hybrid":
                chunks = retrieve_hybrid(
                    query, top_k=payload.top_k,
                    collection_name=payload.collection_name,
                )
            else:
                results[method] = {"error": f"Unknown method: {method}"}
                continue

            results[method] = [
                _serialize_chunk(i + 1, c) for i, c in enumerate(chunks)
            ]
        except Exception as exc:
            logger.exception("Retrieval failed for %s", method)
            results[method] = {"error": f"{type(exc).__name__}: {exc}"}

    return {
        "query": query,
        "top_k": payload.top_k,
        "results": results,
    }

@api.get("/traces")
def list_traces(
    limit: int = 50,
    offset: int = 0,
    action: Optional[str] = None,
    refused: Optional[bool] = None,
    question_contains: Optional[str] = None,
):
    """Recent traces with optional filters."""
    from app.tracing.trace_store import get_store

    store = get_store()

    try:
        traces = store.list(
            limit=limit,
            offset=offset,
            action=action,
            refused=refused,
            question_contains=question_contains,
        )
        total = store.count(
            action=action,
            refused=refused,
            question_contains=question_contains,
        )
    except Exception as exc:
        logger.exception("Failed to list traces")
        raise HTTPException(500, f"Cannot list traces: {exc}") from exc

    return {"traces": traces, "total": total, "limit": limit, "offset": offset}


@api.get("/traces/{trace_id}")
def get_trace_detail(trace_id: str):
    """One trace with full stage details."""
    from app.tracing.trace_store import get_store

    trace = get_store().get(trace_id)
    if trace is None:
        raise HTTPException(404, f"Trace {trace_id} not found")

    return trace.to_dict()


@api.get("/traces/stats/actions")
def trace_action_counts():
    """Counts per action (for a chart on the dashboard)."""
    from app.tracing.trace_store import get_store

    stats = get_store().stats()
    return stats.get("by_action", {})


app.include_router(api)