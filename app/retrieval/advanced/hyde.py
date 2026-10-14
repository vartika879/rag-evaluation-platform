
import logging
import math
import re
 
from app.config import DEFAULT_TOP_K
from app.generation.llm import complete_text
from dataclasses import replace
 
logger = logging.getLogger(__name__)
 
MAX_PASSAGE_CHARS = 1000
 
_LABEL = re.compile(
    r"^\s*(?:hypothetical\s+)?(?:answer|passage|document)\s*:\s*",
    re.IGNORECASE,
)
 
# If the model apologises instead of writing a passage, do not embed that.
_REFUSAL_STARTS = (
    "i'm sorry", "i am sorry", "sorry", "i cannot", "i can't", "i can not",
    "i don't know", "i do not know", "as an ai", "i'm unable", "i am unable",
)
 
 
# ---------------------------------------------------------------------------
# Writing the hypothetical passage
# ---------------------------------------------------------------------------
 
def build_hyde_prompt(query):
    return f"""
Write a short passage (2-4 sentences) that directly answers the question below, in the style of a technical document or research paper.
 
Rules:
1. Write it as if it were an excerpt from the document that contains the answer.
2. Use precise technical terms. Do not hedge, and do not say you don't know.
3. Return ONLY the passage, with no title, label or introduction.
 
Question:
{query}
"""
 
 
def clean_passage(text):
    text = " ".join((text or "").replace("*", " ").replace("`", " ").split())
    text = _LABEL.sub("", text).strip()
 
    if len(text) > MAX_PASSAGE_CHARS:
        text = text[:MAX_PASSAGE_CHARS].rsplit(" ", 1)[0].rstrip()
 
    return text
 
 
def _looks_like_refusal(text):
    return text.lower().lstrip(" \"'").startswith(_REFUSAL_STARTS)
 
 
# ---------------------------------------------------------------------------
# Building the search vector
# ---------------------------------------------------------------------------
 
def _blend(hyde_vector, query_vector, query_weight):
    """(1 - w) * hyde + w * query, scaled to length 1."""
    if query_weight <= 0:
        return list(hyde_vector)
 
    mixed = [
        (1 - query_weight) * h + query_weight * q
        for h, q in zip(hyde_vector, query_vector)
    ]
    length = math.sqrt(sum(x * x for x in mixed)) or 1.0
 
    return [x / length for x in mixed]
 
 
def create_hyde_vector(query, embed_as="document", query_weight=0.0):
    """Return a dict (good for tracing): original_query, hypothetical_document,
    vector (use this for search), vector_dim, embed_as, query_weight,
    llm_called, fallback_reason (None if HyDE was really used), latency_ms,
    usage.
 
    embed_as      "document" (default) embeds the passage like a stored chunk,
                  which suits Cohere's asymmetric model; "query" embeds it like
                  a question. Compare both with the evaluation.
    query_weight  0.0 = pure HyDE, 1.0 = ignore HyDE; e.g. 0.3 keeps some of
                  the real question in the vector.
    """
    # imported here so this module loads (and is testable) without API keys
    from app.embeddings.service import embed_as_document, embed_query_text
 
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")
 
    if embed_as not in ("document", "query"):
        raise ValueError('embed_as must be "document" or "query".')
 
    if not 0 <= query_weight <= 1:
        raise ValueError("query_weight must be between 0 and 1.")
 
    original = query.strip()
 
    result = {
        "original_query": original,
        "hypothetical_document": "",
        "vector": None,
        "vector_dim": None,
        "embed_as": embed_as,
        "query_weight": query_weight,
        "llm_called": True,
        "fallback_reason": None,
        "latency_ms": 0,
        "usage": None,
    }
 
    passage = ""
 
    try:
        response = complete_text(build_hyde_prompt(original))
        result["latency_ms"] = response["latency_ms"]
        result["usage"] = response["usage"]
        passage = clean_passage(response["text"])
    except Exception as error:  # fail open: use a normal query embedding
        logger.warning("HyDE generation failed, using plain query: %s", error)
        result["fallback_reason"] = f"llm_error: {error}"
 
    if not result["fallback_reason"]:
        if not passage:
            result["fallback_reason"] = "empty_output"
        elif _looks_like_refusal(passage):
            result["fallback_reason"] = "model_refused"
 
    result["hypothetical_document"] = passage
 
    if result["fallback_reason"]:
        vector = embed_query_text(original)
    else:
        embed = embed_as_document if embed_as == "document" else embed_query_text
        hyde_vector = embed(passage)
        vector = (
            _blend(hyde_vector, embed_query_text(original), query_weight)
            if query_weight > 0
            else list(hyde_vector)
        )
 
    result["vector"] = vector
    result["vector_dim"] = len(vector)
 
    return result
 
 
# ---------------------------------------------------------------------------
# Old-style helpers
# ---------------------------------------------------------------------------
 
def generate_hypothetical_document(query):
    """Returns just the (cleaned) hypothetical passage as a string."""
    response = complete_text(build_hyde_prompt(query.strip()))
 
    return clean_passage(response["text"])
 
 
def create_hyde_embedding(query):
    """Returns (hypothetical_document, embedding), like the old function."""
    info = create_hyde_vector(query)
 
    return info["hypothetical_document"], info["vector"]
 
 
# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
 
def retrieve_hyde(
    query,
    top_k=DEFAULT_TOP_K,
    collection_name=None,
    page_min=None,
    page_max=None,
    doc_id=None,
    hyde=None,
    embed_as="document",
    query_weight=0.0,
):
    """Vector search with the HyDE vector. Pass `hyde=` (the dict from
    create_hyde_vector) to reuse it and avoid a second LLM call. The signature
    also fits retrieve_multi_query(retrieve_fn=...)."""
    from app.retrieval.retriever import search_by_vector
 
    if hyde is None:
        hyde = create_hyde_vector(query, embed_as, query_weight)
 
    chunks = search_by_vector(
        hyde["vector"],
        top_k=top_k,
        collection_name=collection_name,
        page_min=page_min,
        page_max=page_max,
        doc_id=doc_id,
    )
 
    return [replace(chunk, method="hyde") for chunk in chunks]
 