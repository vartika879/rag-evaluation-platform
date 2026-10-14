 
import logging
import os
 
from app.config import REFUSAL_MESSAGE
from app.generation.llm import create_llm
from app.pipeline import resolve_config, run_pipeline
 
logger = logging.getLogger(__name__)
 
REFUSAL = REFUSAL_MESSAGE  # old name, kept for compatibility
DEFAULT_PRESET = "vector"
CONTEXT_PREVIEW_CHARS = 5000
SOURCE_PREVIEW_CHARS = 250
 
 
def get_llm():
    """Old helper, kept for compatibility (create_llm() is cached itself)."""
    return create_llm()
 
 
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
 
def _default_config():
    preset = os.getenv("RAG_PRESET") or DEFAULT_PRESET
 
    try:
        return resolve_config(preset)
    except ValueError as error:
        raise ValueError(f"RAG_PRESET is invalid: {error}") from error
 
 
def _with_top_k(config, top_k):
    """Change top_k and keep dependent settings valid (a bigger top_k needs a
    candidate pool / rerank list at least as big)."""
    if top_k is None:
        return config
 
    changes = {"top_k": top_k}
 
    if config.use_reranker and config.rerank_candidates < top_k:
        changes["rerank_candidates"] = top_k
 
    if config.retrieval_method == "hybrid" and config.candidate_pool < top_k:
        changes["candidate_pool"] = top_k
 
    return config.with_changes(**changes)
 
 
# ---------------------------------------------------------------------------
# Trace -> response
# ---------------------------------------------------------------------------
 
def _chunk_id_text(value):
    return None if value is None else str(value)
 
 
def build_sources(final_chunks):
    """One entry per (file, page). Keeps the best score and the preview of the
    best chunk on that page, and lists every chunk id found there.
    (The old version zipped deduplicated sources with un-deduplicated results,
    which attached the wrong chunk id / preview to a page.)"""
    sources = {}
 
    for chunk in final_chunks:
        key = (chunk.get("source"), chunk.get("page_number"))
        score = chunk.get("score")
        entry = sources.get(key)
 
        if entry is None:
            sources[key] = {
                "source": key[0],
                "page": key[1],
                "score": score,
                "chunk_id": _chunk_id_text(chunk.get("chunk_id")),
                "chunk_ids": [chunk.get("chunk_id")],
                "preview": " ".join(chunk.get("text", "").split())[:SOURCE_PREVIEW_CHARS],
            }
            continue
 
        entry["chunk_ids"].append(chunk.get("chunk_id"))
 
        if score is not None and (entry["score"] is None or score > entry["score"]):
            entry["score"] = score
            entry["chunk_id"] = _chunk_id_text(chunk.get("chunk_id"))
            entry["preview"] = " ".join(chunk.get("text", "").split())[:SOURCE_PREVIEW_CHARS]
 
    return list(sources.values())
 
 
def trace_to_response(trace):
    """Turn a finished Trace into the dict the API returns."""
    rejected = trace.action == "refused_no_evidence"
 
    # chunks the checker rejected are debugging data, not sources for the user
    sources = [] if rejected else build_sources(trace.final_chunks)
 
    return {
        "question": trace.question,
        "answer": trace.answer,
        "refused": trace.refused,
        "sources": sources,
        "retrieved_count": len(trace.final_chunks),
        "context_preview": trace.context[:CONTEXT_PREVIEW_CHARS],
        # added by the new pipeline
        "trace_id": trace.trace_id,
        "action": trace.action,
        "config_label": trace.config_label,
        "latency_ms": trace.total_latency_ms,
        "usage": trace.usage,
        "chunks": [
            {
                "chunk_id": chunk.get("chunk_id"),
                "page": chunk.get("page_number"),
                "score": chunk.get("score"),
                "rank": chunk.get("rank"),
                "text": chunk.get("text", ""),
            }
            for chunk in ([] if rejected else trace.final_chunks)
        ],
    }
 
 
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
 
def answer_question(
    question: str,
    top_k=None,
    config=None,
    chat_history=None,
    run_id=None,
) -> dict:
    """Answer one question with the configured pipeline.
 
    top_k         how many chunks the LLM gets (None = the config's own value)
    config        preset name, dict or PipelineConfig (None = RAG_PRESET)
    chat_history  [{"role": "user"|"assistant", "content": "..."}] for
                  follow-up questions (used when the config has rewrite on)
    """
    question = (question or "").strip()
 
    if not question:
        raise ValueError("Question cannot be empty.")
 
    config = _with_top_k(
        resolve_config(config) if config is not None else _default_config(),
        top_k,
    )
 
    trace = run_pipeline(
        question,
        config,
        chat_history=chat_history,
        run_id=run_id,
    )
 
    return trace_to_response(trace)
 