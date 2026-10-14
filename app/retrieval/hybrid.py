
import logging
from dataclasses import replace
 
from app.config import (
    CANDIDATE_POOL,
    DEFAULT_TOP_K,
    HYBRID_VECTOR_WEIGHT,
    MAX_TOP_K,
    RRF_K,
)
from app.schemas import RetrievedChunk, assign_ranks
 
logger = logging.getLogger(__name__)
 
FUSION_METHODS = ("rrf", "weighted")


def min_max_normalize(scores):
    """Scale scores to 0..1.
 
    Edge cases (the old version got the last one wrong):
      - empty list              -> []
      - all scores equal, > 0   -> all 1.0  (all tied for best)
      - all scores equal, <= 0  -> all 0.0  (nothing matched, do NOT reward)
    """
    if not scores:
        return []

    low=min(scores)

    high=max(scores)

    if high == low:
        value = 1.0 if high > 0 else 0.0
        return [value] * len(scores)
 
    return [(score - low) / (high - low) for score in scores]


def _validate(top_k, vector_weight, method):
    if not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}.")
 
    if not 0 <= vector_weight <= 1:
        raise ValueError("vector_weight must be between 0 and 1.")
 
    if method not in FUSION_METHODS:
        raise ValueError(f"method must be one of {FUSION_METHODS}.")
 
 
def _merge(vector_chunk, bm25_chunk):
    """One RetrievedChunk carrying the info of both retrievers."""
    base = vector_chunk if vector_chunk is not None else bm25_chunk
    merged = replace(base, method="hybrid")
 
    if vector_chunk is not None:
        merged.vector_score = vector_chunk.vector_score
        merged.vector_rank = vector_chunk.vector_rank
 
    if bm25_chunk is not None:
        merged.bm25_score = bm25_chunk.bm25_score
        merged.bm25_rank = bm25_chunk.bm25_rank
 
    return merged



def hybrid_search(vector_results,
    bm25_results,
    top_k=DEFAULT_TOP_K,
    vector_weight=HYBRID_VECTOR_WEIGHT,
    method="rrf",
    rrf_k=RRF_K
    ):
    _validate(top_k, vector_weight, method)

    bm25_weight= 1-vector_weight

    vector_by_key = {chunk.key: chunk for chunk in vector_results}
    bm25_by_key = {chunk.key: chunk for chunk in bm25_results}
 
    if method == "weighted":
        vector_norm = dict(zip(
            vector_by_key,
            min_max_normalize([c.vector_score or 0.0 for c in vector_by_key.values()]),
        ))
        bm25_norm = dict(zip(
            bm25_by_key,
            min_max_normalize([c.bm25_score or 0.0 for c in bm25_by_key.values()]),
        ))
 
    fused = []
 
    for key in set(vector_by_key) | set(bm25_by_key):
        vector_chunk = vector_by_key.get(key)
        bm25_chunk = bm25_by_key.get(key)
 
        if method == "rrf":
            # scaled by (rrf_k + 1) so rank 1 in both lists scores exactly 1.0
            score = (rrf_k + 1) * (
                vector_weight
                * (1 / (rrf_k + vector_chunk.vector_rank) if vector_chunk else 0.0)
                + bm25_weight
                * (1 / (rrf_k + bm25_chunk.bm25_rank) if bm25_chunk else 0.0)
            )
        else:
            score = (
                vector_weight * vector_norm.get(key, 0.0)
                + bm25_weight * bm25_norm.get(key, 0.0)
            )
 
        merged = _merge(vector_chunk, bm25_chunk)
        merged.hybrid_score = float(score)
        fused.append(merged)
 
    # deterministic: best score, then better vector rank, then better bm25 rank
    fused.sort(
        key=lambda c: (
            -c.hybrid_score,
            c.vector_rank if c.vector_rank is not None else 10**9,
            c.bm25_rank if c.bm25_rank is not None else 10**9,
            c.chunk_id if c.chunk_id is not None else 10**9,
        )
    )
 
    fused = fused[:top_k]
    assign_ranks(fused, "hybrid")
 
    return fused
 

def overlap_stats(vector_results, bm25_results):
    """How much do the two retrievers agree? Useful for debugging/dashboard."""
    vector_keys = {chunk.key for chunk in vector_results}
    bm25_keys = {chunk.key for chunk in bm25_results}
 
    return {
        "vector_only": len(vector_keys - bm25_keys),
        "bm25_only": len(bm25_keys - vector_keys),
        "both": len(vector_keys & bm25_keys),
    }
 

 
# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------
 
def retrieve_hybrid(
    query,
    top_k=DEFAULT_TOP_K,
    collection_name=None,
    page_min=None,
    page_max=None,
    doc_id=None,
    vector_weight=HYBRID_VECTOR_WEIGHT,
    method="rrf",
    candidate_pool=CANDIDATE_POOL,
):
    # imported here so hybrid_search() can be tested without API keys / Qdrant
    from app.retrieval.bm25 import retrieve_bm25
    from app.retrieval.retriever import retrieve_chunks
 
    _validate(top_k, vector_weight, method)
 
    # Fusion needs more candidates than the final top_k, otherwise there is
    # nothing to choose from.
    pool = min(max(candidate_pool, top_k), MAX_TOP_K)
 
    filters = dict(
        collection_name=collection_name,
        page_min=page_min,
        page_max=page_max,
        doc_id=doc_id,
    )
 
    vector_results = retrieve_chunks(query, top_k=pool, **filters)
    bm25_results = retrieve_bm25(query, top_k=pool, **filters)
 
    stats = overlap_stats(vector_results, bm25_results)
    logger.info("Hybrid candidates: %s (pool=%d, method=%s)", stats, pool, method)
 
    return hybrid_search(
        vector_results,
        bm25_results,
        top_k=top_k,
        vector_weight=vector_weight,
        method=method,
    )
 