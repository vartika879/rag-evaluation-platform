
import logging
import time
from dataclasses import replace
from functools import lru_cache
 
from app.config import (
    HYBRID_VECTOR_WEIGHT,
    RERANK_MODEL,
    RERANK_TOP_K,
    get_api_key,
)
from app.schemas import assign_ranks
 
logger = logging.getLogger(__name__)
 
DEFAULT_RERANK_CANDIDATES = 20
MAX_RETRIES = 3
 

# ---------------------------------------------------------------------------
# Client + retry
# ---------------------------------------------------------------------------
 
@lru_cache(maxsize=1)
def _get_client():
    import cohere  # imported here so the module loads without the package
 
    return cohere.Client(api_key=get_api_key("COHERE_API_KEY"), timeout=60)
 
 
def _call_with_retry(func):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func()
        except Exception as error:
            message = str(error).lower()
            retryable = (
                "429" in message
                or "rate limit" in message
                or "too many requests" in message
                or "timeout" in message
            )
 
            if not retryable or attempt == MAX_RETRIES:
                raise
 
            wait_seconds = 2 ** attempt
            logger.warning(
                "Cohere rerank failed (attempt %d/%d), retrying in %ds: %s",
                attempt,
                MAX_RETRIES,
                wait_seconds,
                error,
            )
            time.sleep(wait_seconds)
 
 
# ---------------------------------------------------------------------------
# Rerank
# ---------------------------------------------------------------------------
 
def rerank_results(query, results, top_k=RERANK_TOP_K, min_score=None):
    """Return the best `top_k` chunks, re-ordered by relevance to `query`.
 
    results    list[RetrievedChunk] (e.g. output of hybrid search)
    min_score  optional: drop chunks whose rerank score is below this value
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")
 
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be an integer >= 1.")
 
    if not results:
        return []
 
    # Cohere rejects empty documents; remember where each one came from.
    usable = [chunk for chunk in results if chunk.text and chunk.text.strip()]
 
    if not usable:
        return []
 
    top_n = min(top_k, len(usable))
    started = time.perf_counter()
 
    response = _call_with_retry(
        lambda: _get_client().rerank(
            model=RERANK_MODEL,
            query=query,
            documents=[chunk.text for chunk in usable],
            top_n=top_n,
        )
    )
 
    reranked = []
 
    for item in response.results:
        score = float(item.relevance_score)
 
        if min_score is not None and score < min_score:
            continue
 
        reranked.append(
            replace(usable[item.index], method="rerank", rerank_score=score)
        )
 
    reranked.sort(key=lambda chunk: -chunk.rerank_score)
    assign_ranks(reranked, "rerank")
 
    logger.info(
        "Reranked %d candidates -> %d results in %.0f ms",
        len(usable),
        len(reranked),
        (time.perf_counter() - started) * 1000,
    )
 
    return reranked
 
 
def rank_changes(results, from_stage="hybrid"):
    """How did each final chunk move? positive change = moved up."""
    return [
        {
            "chunk_id": chunk.chunk_id,
            "page_number": chunk.page_number,
            "before": getattr(chunk, f"{from_stage}_rank"),
            "after": chunk.rerank_rank,
            "change": chunk.rank_change(from_stage, "rerank"),
        }
        for chunk in results
    ]
 
 
# ---------------------------------------------------------------------------
# End-to-end: hybrid candidates -> rerank
# ---------------------------------------------------------------------------
 
def retrieve_reranked(
    query,
    top_k=RERANK_TOP_K,
    collection_name=None,
    page_min=None,
    page_max=None,
    doc_id=None,
    candidates=DEFAULT_RERANK_CANDIDATES,
    vector_weight=HYBRID_VECTOR_WEIGHT,
    method="rrf",
    min_score=None,
):
    from app.retrieval.hybrid import retrieve_hybrid
 
    candidates = max(candidates, top_k)
 
    hybrid_results = retrieve_hybrid(
        query,
        top_k=candidates,
        collection_name=collection_name,
        page_min=page_min,
        page_max=page_max,
        doc_id=doc_id,
        vector_weight=vector_weight,
        method=method,
    )
 
    return rerank_results(query, hybrid_results, top_k=top_k, min_score=min_score)
 
