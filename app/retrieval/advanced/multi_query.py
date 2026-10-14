
import logging
import re
from dataclasses import replace
 
from app.config import CANDIDATE_POOL, DEFAULT_TOP_K, RRF_K
from app.generation.llm import complete_text
from app.schemas import assign_ranks
 
logger = logging.getLogger(__name__)
 
MAX_VARIANTS = 10
MAX_QUERY_CHARS = 300
 
_LIST_MARK = re.compile(r"^\s*(?:\d+[\.\)]|[-*\u2022])\s*")
 
 
def _norm(text):
    return " ".join(re.findall(r"\w+", (text or "").lower()))
 
 
# ---------------------------------------------------------------------------
# Generating the variants
# ---------------------------------------------------------------------------
 
def build_variants_prompt(query, num_queries):
    return f"""
Generate {num_queries} different search queries that could find the same information as the question below.
 
Rules:
1. Each query should look at the question from a different angle (different wording, synonyms, or a more specific or more general phrasing).
2. Keep the technical terms from the question.
3. Do NOT answer the question.
4. Return ONLY the queries, one per line, with no numbering and no extra text.
 
Question:
{query}
"""
 
 
def parse_queries(text, original_query, num_queries):
    """Clean LLM output: drop numbering, bullets, quotes, intro lines ending
    with ':', duplicates, and anything equal to the original question."""
    seen = {_norm(original_query)}
    queries = []
 
    for line in (text or "").splitlines():
        line = _LIST_MARK.sub("", line.strip()).strip(" \"'`*_")
 
        if not line or line.endswith(":") or len(line) > MAX_QUERY_CHARS:
            continue
 
        key = _norm(line)
 
        if not key or key in seen:
            continue
 
        seen.add(key)
        queries.append(line)
 
        if len(queries) == num_queries:
            break
 
    return queries
 
 
def generate_query_variants(query, num_queries=3, include_original=True):
    """num_queries = how many EXTRA wordings to ask the LLM for. The original
    question is always kept first (include_original=True), so the final list
    has at most num_queries + 1 entries.
 
    Returns a dict: original_query, queries (use these for search), variants
    (only the new ones), llm_called, fallback_reason, latency_ms, usage."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")
 
    if not isinstance(num_queries, int) or not 1 <= num_queries <= MAX_VARIANTS:
        raise ValueError(f"num_queries must be an integer between 1 and {MAX_VARIANTS}.")
 
    original = query.strip()
 
    result = {
        "original_query": original,
        "queries": [original],
        "variants": [],
        "llm_called": True,
        "fallback_reason": None,
        "latency_ms": 0,
        "usage": None,
    }
 
    try:
        response = complete_text(build_variants_prompt(original, num_queries))
    except Exception as error:  # fail open: search with the original only
        logger.warning("Multi-query generation failed, using original only: %s", error)
        result["fallback_reason"] = f"llm_error: {error}"
        return result
 
    result["latency_ms"] = response["latency_ms"]
    result["usage"] = response["usage"]
 
    variants = parse_queries(response["text"], original, num_queries)
 
    if not variants:
        result["fallback_reason"] = "no_usable_queries"
 
    result["variants"] = variants
    result["queries"] = ([original] if include_original else []) + variants
 
    if not result["queries"]:
        result["queries"] = [original]
 
    return result
 
 
def generate_queries(query, num_queries=3):
    """Old-style helper: returns a list of query strings (original first)."""
    return generate_query_variants(query, num_queries)["queries"]
 
 
# ---------------------------------------------------------------------------
# Fusing the results of several searches
# ---------------------------------------------------------------------------
 
def fuse_ranked_lists(ranked_lists, top_k=DEFAULT_TOP_K, rrf_k=RRF_K):
    """Reciprocal Rank Fusion across any number of ranked chunk lists.
 
    score = average over lists of 1 / (rrf_k + position), scaled by
    (rrf_k + 1) so "position 1 in every list" scores exactly 1.0.
 
    The fused score and rank are written to hybrid_score / hybrid_rank (they
    now mean "fused across query variants"). Everything else (vector_rank,
    bm25_rank...) is taken from the occurrence where the chunk ranked best.
    """
    lists = [chunks for chunks in ranked_lists if chunks]
 
    if not lists:
        return []
 
    totals = {}
    best = {}
 
    for chunks in lists:
        for position, chunk in enumerate(chunks, start=1):
            key = chunk.key
            totals[key] = totals.get(key, 0.0) + 1.0 / (rrf_k + position)
 
            if key not in best or position < best[key][0]:
                best[key] = (position, chunk)
 
    fused = []
 
    for key, total in totals.items():
        position, chunk = best[key]
        score = (rrf_k + 1) * total / len(lists)
        fused.append(replace(chunk, method="multi_query", hybrid_score=score))
 
    fused.sort(
        key=lambda c: (
            -c.hybrid_score,
            best[c.key][0],
            c.chunk_id if c.chunk_id is not None else 10**9,
        )
    )
 
    fused = fused[:top_k]
    assign_ranks(fused, "hybrid")
 
    return fused
 
 
def retrieve_multi_query(
    query,
    top_k=DEFAULT_TOP_K,
    num_queries=3,
    variants=None,
    retrieve_fn=None,
    per_query_k=min(CANDIDATE_POOL, 20),
    collection_name=None,
    page_min=None,
    page_max=None,
    doc_id=None,
):
    """Search with every wording and fuse.
 
    variants     the query list to use (e.g. info["queries"]); if None, they
                 are generated here.
    retrieve_fn  function(query, top_k=..., **filters) -> list[RetrievedChunk];
                 defaults to hybrid search.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")
 
    if retrieve_fn is None:
        from app.retrieval.hybrid import retrieve_hybrid
 
        retrieve_fn = retrieve_hybrid
 
    queries = variants or generate_query_variants(query, num_queries)["queries"]
 
    filters = dict(
        collection_name=collection_name,
        page_min=page_min,
        page_max=page_max,
        doc_id=doc_id,
    )
 
    ranked_lists = [
        retrieve_fn(text, top_k=per_query_k, **filters) for text in queries
    ]
 
    logger.info(
        "Multi-query: %d queries, %s results each",
        len(queries),
        [len(chunks) for chunks in ranked_lists],
    )
 
    return fuse_ranked_lists(ranked_lists, top_k=top_k)