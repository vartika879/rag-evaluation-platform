
import logging
import re
 
from app.config import DEFAULT_TOP_K, MAX_TOP_K
from app.generation.llm import complete_text
from app.retrieval.advanced.multi_query import (
    fuse_ranked_lists,
    generate_query_variants,
)
from app.retrieval.advanced.query_expansion import expand_query_detailed
 
logger = logging.getLogger(__name__)
 
VERDICTS = ("GOOD", "PARTIAL", "BAD")
STRATEGIES = ("expand", "multi_query")
MAX_GRADE_CHARS = 6000
MAX_CHARS_PER_CHUNK = 800
 
_VERDICT_LINE = re.compile(r"\bverdict\s*[:\-]?\W*(good|partial|bad)\b", re.IGNORECASE)
_ONLY_WORD = re.compile(r"^\W*(good|partial|bad)\W*$", re.IGNORECASE)
_REASON = re.compile(r"\breason\s*[:\-]\s*(.+)", re.IGNORECASE)
 
 
# ---------------------------------------------------------------------------
# 1. Grading
# ---------------------------------------------------------------------------
 
def _format_chunks(chunks, max_chars=MAX_GRADE_CHARS):
    parts = []
    total = 0
 
    for number, chunk in enumerate(chunks, start=1):
        text = " ".join(chunk.text.split())[:MAX_CHARS_PER_CHUNK]
        part = f"[{number}] {text}"
 
        if parts and total + len(part) > max_chars:
            break
 
        parts.append(part)
        total += len(part)
 
    return "\n\n".join(parts)
 
 
def build_grade_prompt(query, context):
    return f"""
You are checking retrieved text for a question-answering system.
 
Question:
{query}
 
Retrieved text:
{context}
 
Decide whether the retrieved text contains enough information to answer the question.
VERDICT must be one of:
GOOD    - the text clearly contains the answer
PARTIAL - the text is related and has some of the answer, but it is incomplete
BAD     - the text does not contain the answer
 
Reply in exactly this format:
VERDICT: <GOOD or PARTIAL or BAD>
REASON: <one short sentence>
"""
 
 
def parse_verdict(text):
    """Return (verdict, reason). verdict is GOOD / PARTIAL / BAD / UNKNOWN.
 
    Only the labelled "VERDICT: x" or a first line that is just the word is
    trusted, so a sentence like "the text is not bad" is never read as BAD."""
    text = (text or "").strip()
    reason_match = _REASON.search(text)
    reason = reason_match.group(1).strip() if reason_match else ""
 
    labelled = _VERDICT_LINE.search(text)
 
    if labelled:
        return labelled.group(1).upper(), reason
 
    first_line = text.splitlines()[0] if text else ""
    only = _ONLY_WORD.match(first_line)
 
    if only:
        return only.group(1).upper(), reason
 
    return "UNKNOWN", reason
 
 
def grade_context(query, chunks):
    """Return a dict (good for tracing): verdict, reason, llm_called,
    fallback_reason, latency_ms, usage."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")
 
    result = {
        "verdict": "BAD",
        "reason": "",
        "llm_called": False,
        "fallback_reason": None,
        "latency_ms": 0,
        "usage": None,
    }
 
    if not chunks:
        result["reason"] = "no chunks retrieved"
        return result
 
    result["llm_called"] = True
 
    try:
        response = complete_text(
            build_grade_prompt(query.strip(), _format_chunks(chunks))
        )
    except Exception as error:
        logger.warning("Context grading failed: %s", error)
        result["verdict"] = "UNKNOWN"
        result["fallback_reason"] = f"llm_error: {error}"
        return result
 
    result["latency_ms"] = response["latency_ms"]
    result["usage"] = response["usage"]
 
    verdict, reason = parse_verdict(response["text"])
    result["verdict"] = verdict
    result["reason"] = reason
 
    if verdict == "UNKNOWN":
        result["fallback_reason"] = "unparseable_output"
 
    return result
 
 
def evaluate_context(query, results):
    """Old-style helper: returns just the verdict string."""
    return grade_context(query, results)["verdict"]
 
 
# ---------------------------------------------------------------------------
# 2. Cheap score-based check (no LLM)
# ---------------------------------------------------------------------------
 
def has_supporting_evidence(results, threshold=0.30):
    """True if the best ABSOLUTE score reaches `threshold`.
 
    Uses the rerank score if present, otherwise the vector (cosine) score.
    Hybrid / RRF / BM25 scores have no fixed scale, so when only those exist
    this returns True (no opinion). The 0.30 default is a guess: calibrate it
    on the golden dataset before relying on it.
    """
    if not results:
        return False
 
    best = None
 
    for chunk in results:
        value = getattr(chunk, "rerank_score", None)
 
        if value is None:
            value = getattr(chunk, "vector_score", None)
 
        if value is None and not hasattr(chunk, "hybrid_score"):
            value = getattr(chunk, "score", None)  # raw Qdrant point
 
        if value is not None:
            best = value if best is None else max(best, value)
 
    return True if best is None else best >= threshold
 
 
# ---------------------------------------------------------------------------
# 3. Corrective retrieval
# ---------------------------------------------------------------------------
 
def corrective_retrieval_detailed(
    query,
    top_k=DEFAULT_TOP_K,
    strategy="expand",
    retrieve_fn=None,
    **filters,
):
    """Search again, differently. Returns {chunks, strategy, queries, query_info}.
 
    strategy "expand"       one search with the query + related keywords
    strategy "multi_query"  several rewordings, results fused (RRF)
    The net is wider than top_k (2x) so the grader has more to look at.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}.")
 
    if retrieve_fn is None:
        from app.retrieval.hybrid import retrieve_hybrid
 
        retrieve_fn = retrieve_hybrid
 
    pool = min(max(top_k * 2, 10), MAX_TOP_K)
 
    if strategy == "expand":
        info = expand_query_detailed(query)
        queries = [info["query"]]
        chunks = retrieve_fn(info["query"], top_k=pool, **filters)
    else:
        info = generate_query_variants(query)
        queries = info["queries"]
        lists = [retrieve_fn(text, top_k=pool, **filters) for text in queries]
        chunks = fuse_ranked_lists(lists, top_k=pool)
 
    return {
        "chunks": chunks,
        "strategy": strategy,
        "queries": queries,
        "query_info": info,
    }
 
 
def corrective_retrieval(query, top_k=DEFAULT_TOP_K, **kwargs):
    """Old-style helper: returns just the list of chunks."""
    return corrective_retrieval_detailed(query, top_k, **kwargs)["chunks"]
 
 
# ---------------------------------------------------------------------------
# 4. The whole CRAG step
# ---------------------------------------------------------------------------
 
def run_crag(
    query,
    chunks,
    top_k=DEFAULT_TOP_K,
    strategy="expand",
    retrieve_fn=None,
    **filters,
):
    """Returns a dict:
        action   "use_initial" | "use_corrected" | "no_evidence"
        chunks   the chunks to give to the LLM (for "no_evidence" the
                 last chunks tried, kept for debugging; do not answer from them)
        grades   list of grade dicts (1 or 2 entries)
        corrective  None, or {strategy, queries}
        llm_calls   total LLM calls made by this step
        usages      token-usage dict of every LLM call that answered
    """
    first = grade_context(query, chunks)
    grades = [first]
 
    if first["verdict"] == "GOOD":
        return {
            "action": "use_initial",
            "chunks": chunks,
            "grades": grades,
            "corrective": None,
            "llm_calls": int(first["llm_called"]),
            "usages": [first["usage"]] if first["usage"] else [],
        }
 
    corrective = corrective_retrieval_detailed(
        query, top_k, strategy, retrieve_fn, **filters
    )
    new_chunks = corrective["chunks"]
 
    if first["verdict"] == "BAD" or not chunks:
        pool = new_chunks[:top_k]  # the first chunks were noise
    else:
        # PARTIAL / UNKNOWN: keep what we had and add what we found
        pool = fuse_ranked_lists([chunks, new_chunks], top_k=top_k)
 
    second = grade_context(query, pool)
    grades.append(second)
 
    action = "no_evidence" if second["verdict"] == "BAD" else "use_corrected"
 
    extra_calls = int(corrective["query_info"].get("llm_called", False))
 
    usages = [g["usage"] for g in grades if g["usage"]]
 
    if corrective["query_info"].get("usage"):
        usages.append(corrective["query_info"]["usage"])
 
    return {
        "action": action,
        "chunks": pool,
        "grades": grades,
        "corrective": {
            "strategy": corrective["strategy"],
            "queries": corrective["queries"],
        },
        "llm_calls": sum(int(g["llm_called"]) for g in grades) + extra_calls,
        "usages": usages,
    }
 