
import logging
import time
 
from qdrant_client.models import FieldCondition, Filter, MatchValue, Range
 
from app.config import (
    DEFAULT_COLLECTION,
    DEFAULT_TOP_K,
    MAX_CONTEXT_CHARS,
    MAX_TOP_K,
    EMBEDDING_DIM, 
)
from app.embeddings.service import embed_query_text
from app.schemas import RetrievedChunk, assign_ranks
from app.vectorstore.qdrant_store import (
    collection_exists,
    get_qdrant_client,
    list_collection_names,
    release_qdrant_client,
)
from app.vectorstore.registry import resolve_collection
 
logger = logging.getLogger(__name__)
 
# kept for old imports (from app.retrieval.retriever import COLLECTION_NAME)
COLLECTION_NAME = DEFAULT_COLLECTION
 
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def release_client():
    """Close the shared Qdrant client. Local Qdrant allows only one process
    to open the folder, so call this before running ingest from another
    terminal while an API/notebook is still alive."""
    release_qdrant_client()
 
 
def _validate_top_k(top_k):
    if not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be an integer between 1 and {MAX_TOP_K}.")
 
 
def _build_filter(page_min=None, page_max=None, doc_id=None):
    """page_min / page_max are 1-indexed (what users see), matched against
    the payload field `page_number`."""
    conditions = []
 
    if page_min is not None or page_max is not None:
        conditions.append(
            FieldCondition(
                key="page_number",
                range=Range(gte=page_min, lte=page_max),
            )
        )
 
    if doc_id is not None:
        conditions.append(
            FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
        )
 
    return Filter(must=conditions) if conditions else None
 
 
# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def search_by_vector(
    query_vector,
    top_k=DEFAULT_TOP_K,
    collection_name=None,
    page_min=None,
    page_max=None,
    doc_id=None,
    score_threshold=None,
):
    """Search Qdrant with an already computed vector (also used by HyDE)."""
    _validate_top_k(top_k)

    if len(query_vector) != EMBEDDING_DIM:
        raise ValueError(
            f"Expected {EMBEDDING_DIM}-dim vector, got {len(query_vector)}."
        )
 
    name = resolve_collection(collection_name or DEFAULT_COLLECTION)
    client = get_qdrant_client()
 
    if not collection_exists(client, name):
        raise ValueError(
            f"Collection '{name}' does not exist. "
            f"Available: {list_collection_names(client)}. "
            "Run the ingest script first."
        )
 
    started = time.perf_counter()
 
    response = client.query_points(
        collection_name=name,
        query=list(query_vector),
        query_filter=_build_filter(page_min, page_max, doc_id),
        limit=top_k,
        with_payload=True,
        score_threshold=score_threshold,
    )
 
    chunks = [RetrievedChunk.from_qdrant_point(p) for p in response.points]
    assign_ranks(chunks, "vector")
 
    logger.info(
        "Vector search in '%s': %d results in %.0f ms",
        name,
        len(chunks),
        (time.perf_counter() - started) * 1000,
    )
 
    return chunks
 
 
def retrieve_chunks(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    collection_name=None,
    page_min=None,
    page_max=None,
    doc_id=None,
    score_threshold=None,
):
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")
 
    
 
    query_vector = embed_query_text(query)
 
    return search_by_vector(
        query_vector,
        top_k=top_k,
        collection_name=collection_name,
        page_min=page_min,
        page_max=page_max,
        doc_id=doc_id,
        score_threshold=score_threshold,
    )
 
 
def retrieve_with_query_vector(
    query_vector,
    top_k=DEFAULT_TOP_K,
    collection_name=None,
    page_min=None,
    page_max=None,
    doc_id=None,
):
    return search_by_vector(
        query_vector,
        top_k=top_k,
        collection_name=collection_name,
        page_min=page_min,
        page_max=page_max,
        doc_id=doc_id,
    )
 
 
# ---------------------------------------------------------------------------
# Context + sources (work with RetrievedChunk and with old Qdrant points)
# ---------------------------------------------------------------------------
 
def _page_number(payload):
    page_number = payload.get("page_number")
 
    if page_number is not None:
        return page_number
 
    page = payload.get("page")
 
    return page + 1 if page is not None else None
 
 
def build_context(results, max_chars=MAX_CONTEXT_CHARS):
    context_parts = []
    total = 0
 
    for i, result in enumerate(results, start=1):
        payload = result.payload or {}
        page_number = _page_number(payload)
 
        part = (
            f"[Source {i}]\n"
            f"File: {payload.get('source', '')}\n"
            f"Page: {page_number if page_number is not None else 'Unknown'}\n"
            f"Content:\n{payload.get('text', '')}"
        )
 
        if max_chars and context_parts and total + len(part) > max_chars:
            logger.info("Context cut at %d of %d chunks (max_chars=%d)",
                        len(context_parts), len(results), max_chars)
            break
 
        context_parts.append(part)
        total += len(part)
 
    return "\n\n".join(context_parts)
 
 
def build_sources(results):
    """One entry per (file, page). Keeps the best score on that page and lists every chunk_id found there."""
    sources = {}
 
    for result in results:
        payload = result.payload or {}
        key = (payload.get("source", ""), _page_number(payload))
 
        entry = sources.setdefault(
            key,
            {
                "source": key[0],
                "page": key[1],
                "score": result.score,
                "chunk_ids": [],
            },
        )
 
        entry["chunk_ids"].append(payload.get("chunk_id"))
 
        if result.score is not None and (
            entry["score"] is None or result.score > entry["score"]
        ):
            entry["score"] = result.score
 
    return list(sources.values())
 