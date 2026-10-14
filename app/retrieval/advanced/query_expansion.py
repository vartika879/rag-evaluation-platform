
import logging
import re
 
from app.generation.llm import complete_text
 
logger = logging.getLogger(__name__)
 
MAX_TERMS = 8
MAX_TERM_CHARS = 40
MAX_TERM_WORDS = 4
 
# ignored when deciding whether a term adds anything new to the query
_FILLER = frozenset({"a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "with"})
 
_LIST_MARK = re.compile(r"^\s*(?:\d+[\.\)]|[-*\u2022])\s*")
_LABEL = re.compile(
    r"^\s*(?:here\s+are|related|keywords?|terms?|expanded)[^:\n]*:\s*",
    re.IGNORECASE,
)
 
 
def _words(text):
    return re.findall(r"\w+", (text or "").lower())
 
 
def build_expansion_prompt(query, max_terms=MAX_TERMS):
    return f"""
List up to {max_terms} extra search keywords or short phrases that are closely related to the question below, such as synonyms, abbreviations or key technical terms.
Do not repeat words that are already in the question, and do not answer the question.
Return ONLY the terms, separated by commas.
 
Question:
{query}
"""
 
 
def parse_terms(text, original_query, max_terms=MAX_TERMS):
    """Turn messy LLM output into a clean list of NEW terms."""
    text = _LABEL.sub("", text or "")
    original_words = set(_words(original_query))
 
    terms = []
    seen = set()
 
    for raw in re.split(r"[,\n;]", text):
        term = _LIST_MARK.sub("", raw.strip()).strip(" \"'`*_.")
 
        if not 2 <= len(term) <= MAX_TERM_CHARS:
            continue
 
        if len(term.split()) > MAX_TERM_WORDS:
            continue
 
        words = _words(term)
        meaningful = [word for word in words if word not in _FILLER]
 
        # nothing new if every meaningful word is already in the original query
        if not meaningful or all(word in original_words for word in meaningful):
            continue
 
        key = " ".join(words)
 
        if key in seen:
            continue
 
        seen.add(key)
        terms.append(term)
 
        if len(terms) == max_terms:
            break
 
    return terms
 
 
def expand_query_detailed(query, max_terms=MAX_TERMS):
    """Return a dict (good for tracing): original_query, query (the expanded
    one), added_terms, changed, llm_called, fallback_reason, latency_ms, usage."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")
 
    if not isinstance(max_terms, int) or not 1 <= max_terms <= 20:
        raise ValueError("max_terms must be an integer between 1 and 20.")
 
    original = query.strip()
 
    result = {
        "original_query": original,
        "query": original,
        "added_terms": [],
        "changed": False,
        "llm_called": False,
        "fallback_reason": None,
        "latency_ms": 0,
        "usage": None,
    }
 
    try:
        response = complete_text(build_expansion_prompt(original, max_terms))
    except Exception as error:  # fail open: search with the original query
        logger.warning("Query expansion failed, using original query: %s", error)
        result["fallback_reason"] = f"llm_error: {error}"
        return result
 
    result["latency_ms"] = response["latency_ms"]
    result["usage"] = response["usage"]
 
    terms = parse_terms(response["text"], original, max_terms)
 
    if not terms:
        result["fallback_reason"] = "no_usable_terms"
        return result
 
    result["added_terms"] = terms
    result["query"] = f"{original} {' '.join(terms)}"
    result["changed"] = True
 
    return result
 
 
def expand_query(query):
    """Old-style helper: returns just the expanded query string."""
    return expand_query_detailed(query)["query"]
 