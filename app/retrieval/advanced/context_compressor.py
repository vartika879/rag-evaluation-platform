
import logging
import re
 
from app.generation.llm import complete_text
 
logger = logging.getLogger(__name__)
 
MIN_CHARS_TO_COMPRESS = 1500
 
_SOURCE_MARK = re.compile(r"^\s*\[Source\s+(\d+)\]\s*$", re.IGNORECASE | re.MULTILINE)
_LIST_MARK = re.compile(r"^\s*(?:[-*\u2022]|\d+[\.\)])\s+")
 
 
# ---------------------------------------------------------------------------
# Formatting (same layout as retriever.build_context, so generate_answer
# works unchanged)
# ---------------------------------------------------------------------------
 
def _format_chunk(number, chunk, text):
    page = chunk.page_number if chunk.page_number is not None else "Unknown"
 
    return (
        f"[Source {number}]\n"
        f"File: {chunk.source or ''}\n"
        f"Page: {page}\n"
        f"Content:\n{text}"
    )
 
 
def _full_context(chunks):
    return "\n\n".join(
        _format_chunk(number, chunk, chunk.text)
        for number, chunk in enumerate(chunks, start=1)
    )
 
 
# ---------------------------------------------------------------------------
# Prompt + parsing + verification
# ---------------------------------------------------------------------------
 
def build_compress_prompt(query, chunks):
    sources = "\n\n".join(
        f"[Source {number}]\n{chunk.text}"
        for number, chunk in enumerate(chunks, start=1)
    )
 
    return f"""
You are shortening retrieved text for a question-answering system.
 
For each source below, copy ONLY the sentences that help answer the question.
Copy them EXACTLY, word for word. Do not paraphrase, summarize or add anything.
If a source has nothing relevant, write NONE for it.
 
Return every source, in the same order, in exactly this format:
[Source 1]
<copied sentences, one per line, or NONE>
[Source 2]
<copied sentences, one per line, or NONE>
 
Question:
{query}
 
Sources:
{sources}
"""
 
 
def parse_sources(text):
    """'[Source 2]\\nline\\nline ...' -> {2: ['line', 'line'], 3: []}.
    NONE becomes an empty list."""
    pieces = _SOURCE_MARK.split(text or "")
    parsed = {}
 
    # pieces = [before, "1", body1, "2", body2, ...]
    for index in range(1, len(pieces) - 1, 2):
        number = int(pieces[index])
        lines = []
 
        for raw in pieces[index + 1].splitlines():
            line = _LIST_MARK.sub("", raw.strip()).strip(" \"'")
 
            if line and line.upper().rstrip(".") != "NONE":
                lines.append(line)
 
        parsed[number] = lines
 
    return parsed
 
 
def _normalize(text):
    return " ".join((text or "").lower().split())
 
 
def _is_verbatim(line, original_normalized):
    """True if the copied line (or each part of it, when the LLM joined
    pieces with '...') appears in the original chunk."""
    parts = re.split(r"\.\.\.|\u2026", line)
    parts = [_normalize(part).strip(" .") for part in parts]
    parts = [part for part in parts if part]
 
    return bool(parts) and all(part in original_normalized for part in parts)
 
 
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
 
def compress_context_detailed(query, chunks, min_chars=MIN_CHARS_TO_COMPRESS):
    """Return a dict (good for tracing): context (give this to the LLM),
    original_chars, compressed_chars, ratio, per_chunk (status per source),
    unverified_lines, llm_called, fallback_reason, latency_ms, usage."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")
 
    full = _full_context(chunks)
 
    result = {
        "context": full,
        "original_chars": len(full),
        "compressed_chars": len(full),
        "ratio": 1.0,
        "per_chunk": [],
        "unverified_lines": 0,
        "llm_called": False,
        "fallback_reason": None,
        "latency_ms": 0,
        "usage": None,
    }
 
    if not chunks:
        result["fallback_reason"] = "no_chunks"
        return result
 
    if len(full) < min_chars:
        result["fallback_reason"] = "already_short"
        return result
 
    result["llm_called"] = True
 
    try:
        response = complete_text(build_compress_prompt(query.strip(), chunks))
    except Exception as error:  # fail open: keep the full context
        logger.warning("Context compression failed, keeping full context: %s", error)
        result["fallback_reason"] = f"llm_error: {error}"
        return result
 
    result["latency_ms"] = response["latency_ms"]
    result["usage"] = response["usage"]
 
    parsed = parse_sources(response["text"])
 
    if not parsed:
        result["fallback_reason"] = "unparseable_output"
        return result
 
    parts = []
    per_chunk = []
    unverified_total = 0
 
    for number, chunk in enumerate(chunks, start=1):
        original_normalized = _normalize(chunk.text)
        lines = parsed.get(number)
 
        entry = {
            "source_index": number,
            "chunk_id": chunk.chunk_id,
            "original_chars": len(chunk.text),
            "kept_chars": 0,
            "status": "",
        }
 
        if lines is None:
            # the LLM skipped this source: keep it whole, do not lose evidence
            entry["status"] = "kept_original_missing"
            text = chunk.text
        elif not lines:
            entry["status"] = "dropped_none"
            text = None
        else:
            verified = [line for line in lines if _is_verbatim(line, original_normalized)]
            unverified_total += len(lines) - len(verified)
 
            if verified:
                entry["status"] = "extracted"
                text = "\n".join(verified)
            else:
                entry["status"] = "kept_original_unverified"
                text = chunk.text
 
        if text is not None:
            entry["kept_chars"] = len(text)
            parts.append(_format_chunk(number, chunk, text))
 
        per_chunk.append(entry)
 
    result["per_chunk"] = per_chunk
    result["unverified_lines"] = unverified_total
 
    if not parts:
        # every source was NONE: do not trust that blindly, let the answer step decide
        result["fallback_reason"] = "all_none"
        return result
 
    compressed = "\n\n".join(parts)
 
    result["context"] = compressed
    result["compressed_chars"] = len(compressed)
    result["ratio"] = round(len(compressed) / len(full), 3)
 
    return result
 
 
def compress_context(query, results):
    """Old-style helper: returns just the context string."""
    return compress_context_detailed(query, results)["context"]
 