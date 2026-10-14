"""
The RAG pipeline: one question in, one complete Trace out.

Place this file at: app/pipeline.py

    run_pipeline(question, config) -> Trace

The PipelineConfig decides which stages run. Stages, in order:

    rewrite          standalone-question rewrite from chat history     (optional)
    query_transform  expand / hyde / multi_query                       (optional)
    retrieve         vector | bm25 | hybrid
    rerank           Cohere rerank                                     (optional)
    crag             grade the chunks, search again if needed          (optional)
    compress         keep only the useful sentences                    (optional)
    context          build the text given to the LLM
    generate         the grounded answer

Every stage is timed and recorded in the Trace (skipped stages too, with the
reason). The Trace is saved to the TraceStore unless save=False.

Failure policy
    - a stage that fails raises: the Trace is saved with that stage marked
      "error", then the exception is re-raised (raise_errors=True, for the
      API) or the Trace is returned with action "error" (raise_errors=False,
      for evaluation runs that must continue with the next question).
    - helper stages that have a safe fallback (rewrite, expansion, HyDE,
      multi-query, compression) never fail the run: they fall back and the
      stage is marked "fallback".
"""

import logging

from app.config import MAX_TOP_K, REFUSAL_MESSAGE
from app.generation.llm import generate_answer
from app.pipeline_config import PipelineConfig, get_preset
from app.retrieval.advanced.context_compressor import compress_context_detailed
from app.retrieval.advanced.crag import run_crag
from app.retrieval.advanced.hyde import create_hyde_vector, retrieve_hyde
from app.retrieval.advanced.multi_query import (
    generate_query_variants,
    retrieve_multi_query,
)
from app.retrieval.advanced.query_expansion import expand_query_detailed
from app.retrieval.advanced.query_transformer import rewrite_query
from app.retrieval.bm25 import retrieve_bm25
from app.retrieval.hybrid import retrieve_hybrid
from app.retrieval.reranker import rank_changes, rerank_results
from app.retrieval.retriever import build_context, retrieve_chunks
from app.tracing.trace import Trace
from app.tracing.trace_store import save_trace

logger = logging.getLogger(__name__)

# fallback reasons that are normal, not a problem
_BENIGN_REASONS = {None, "no_history", "already_short"}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def resolve_config(config):
    """Accept a PipelineConfig, a preset name, a dict, or None (baseline)."""
    if config is None:
        return PipelineConfig()

    if isinstance(config, PipelineConfig):
        return config

    if isinstance(config, str):
        return get_preset(config)

    if isinstance(config, dict):
        return PipelineConfig.from_dict(config)

    raise TypeError("config must be a PipelineConfig, preset name, dict or None.")


def _filters(config):
    return {
        "collection_name": config.collection_name,
        "page_min": config.page_min,
        "page_max": config.page_max,
        "doc_id": config.doc_id,
    }


def _make_retrieve_fn(config):
    """One function(text, top_k, **filters) for the configured retrieval
    method. Also handed to multi-query and CRAG, which search more than once."""

    def retrieve(text, top_k, collection_name=None, page_min=None,
                 page_max=None, doc_id=None):
        filters = dict(
            collection_name=collection_name,
            page_min=page_min,
            page_max=page_max,
            doc_id=doc_id,
        )

        if config.retrieval_method == "vector":
            return retrieve_chunks(text, top_k=top_k, **filters)

        if config.retrieval_method == "bm25":
            return retrieve_bm25(text, top_k=top_k, **filters)

        return retrieve_hybrid(
            text,
            top_k=top_k,
            vector_weight=config.vector_weight,
            method=config.fusion_method,
            candidate_pool=config.candidate_pool,
            **filters,
        )

    return retrieve


def _rank_stage(config):
    """Which stage's rank to compare the reranker against."""
    if config.query_mode == "multi_query" or config.retrieval_method == "hybrid":
        return "hybrid"

    return "bm25" if config.retrieval_method == "bm25" else "vector"


def _summaries(chunks):
    # "preview" keeps up to 1500 characters (a whole chunk for the default
    # chunk sizes): evaluation needs the text to match evidence snippets, and
    # the dashboard can show the full chunk. UIs should shorten it themselves.
    return [c.to_dict(include_text=False, preview_chars=1500) for c in chunks]


def _status(fallback_reason):
    return "ok" if fallback_reason in _BENIGN_REASONS else "fallback"


def _add_usage(trace, usage):
    if usage:
        trace.add_usage(usage)


def _finalize(trace, save, run_id):
    trace.finish()

    if save:
        save_trace(trace, run_id)

    return trace


# ---------------------------------------------------------------------------
# The stages
# ---------------------------------------------------------------------------

def _run(trace, question, config, chat_history, llm):
    filters = _filters(config)
    retrieve_fn = _make_retrieve_fn(config)
    search_query = question

    # ---- rewrite ---------------------------------------------------------
    if config.rewrite_with_history and chat_history:
        with trace.stage("rewrite") as stage:
            info = rewrite_query(question, chat_history)
            _add_usage(trace, info["usage"])
            search_query = info["query"]
            stage.status = _status(info["fallback_reason"])
            stage.details.update(
                original=info["original_query"],
                rewritten=info["query"],
                changed=info["changed"],
                fallback_reason=info["fallback_reason"],
                history_turns=info["history_turns"],
            )
    else:
        reason = "no_history" if config.rewrite_with_history else "disabled"
        trace.skip("rewrite", reason)

    # ---- query transform -------------------------------------------------
    retrieval_text = search_query
    hyde_info = None
    multi_queries = None

    if config.query_mode == "expand":
        with trace.stage("query_transform", mode="expand") as stage:
            info = expand_query_detailed(search_query)
            _add_usage(trace, info["usage"])
            retrieval_text = info["query"]
            stage.status = _status(info["fallback_reason"])
            stage.details.update(
                query=info["query"],
                added_terms=info["added_terms"],
                fallback_reason=info["fallback_reason"],
            )

    elif config.query_mode == "hyde":
        with trace.stage("query_transform", mode="hyde") as stage:
            hyde_info = create_hyde_vector(
                search_query, config.hyde_embed_as, config.hyde_query_weight
            )
            _add_usage(trace, hyde_info["usage"])
            stage.status = _status(hyde_info["fallback_reason"])
            stage.details.update(
                hypothetical_document=hyde_info["hypothetical_document"],
                embed_as=hyde_info["embed_as"],
                query_weight=hyde_info["query_weight"],
                vector_dim=hyde_info["vector_dim"],
                fallback_reason=hyde_info["fallback_reason"],
            )

    elif config.query_mode == "multi_query":
        with trace.stage("query_transform", mode="multi_query") as stage:
            info = generate_query_variants(search_query, config.num_query_variants)
            _add_usage(trace, info["usage"])
            multi_queries = info["queries"]
            stage.status = _status(info["fallback_reason"])
            stage.details.update(
                queries=info["queries"],
                fallback_reason=info["fallback_reason"],
            )
    else:
        trace.skip("query_transform", "plain")

    # ---- retrieve --------------------------------------------------------
    n_fetch = max(config.top_k, config.retrieval_depth)
    if config.use_reranker:
        n_fetch = max(n_fetch, config.rerank_candidates)

    with trace.stage(
        "retrieve",
        method=config.retrieval_method,
        query_mode=config.query_mode,
        requested=n_fetch,
    ) as stage:
        if config.query_mode == "hyde":
            chunks = retrieve_hyde(search_query, top_k=n_fetch, hyde=hyde_info, **filters)
        elif config.query_mode == "multi_query":
            chunks = retrieve_multi_query(
                search_query,
                top_k=n_fetch,
                variants=multi_queries,
                retrieve_fn=retrieve_fn,
                per_query_k=min(MAX_TOP_K, max(n_fetch, 20)),
                **filters,
            )
        else:
            chunks = retrieve_fn(retrieval_text, n_fetch, **filters)

        stage.details.update(
            query=retrieval_text if config.query_mode != "hyde" else "(hypothetical passage)",
            returned=len(chunks),
            results=_summaries(chunks),
        )

    # ---- rerank ----------------------------------------------------------
    if config.use_reranker and chunks:
        with trace.stage("rerank", candidates=len(chunks), top_k=config.top_k) as stage:
            chunks = rerank_results(
                search_query,
                chunks,
                top_k=config.top_k,
                min_score=config.rerank_min_score,
            )
            stage.details.update(
                kept=len(chunks),
                changes=rank_changes(chunks, from_stage=_rank_stage(config)),
            )
    elif config.use_reranker:
        trace.skip("rerank", "no_candidates")
    else:
        trace.skip("rerank", "disabled")
        chunks = chunks[: config.top_k]

    # ---- crag ------------------------------------------------------------
    if config.use_crag:
        with trace.stage("crag", strategy=config.crag_strategy) as stage:
            out = run_crag(
                search_query,
                chunks,
                top_k=config.top_k,
                strategy=config.crag_strategy,
                retrieve_fn=retrieve_fn,
                **filters,
            )

            for usage in out["usages"]:
                _add_usage(trace, usage)

            if any(g["fallback_reason"] for g in out["grades"]):
                stage.status = "fallback"

            stage.details.update(
                action=out["action"],
                grades=[
                    {
                        "verdict": g["verdict"],
                        "reason": g["reason"],
                        "fallback_reason": g["fallback_reason"],
                    }
                    for g in out["grades"]
                ],
                corrective=out["corrective"],
                results=_summaries(out["chunks"]),
            )

            chunks = out["chunks"]
            crag_action = out["action"]

        if crag_action == "no_evidence":
            # final_chunks holds what CRAG rejected (for debugging); the LLM
            # is never called, so it cannot guess.
            trace.set_chunks(chunks)
            trace.skip("compress", "no_evidence")
            trace.skip("context", "no_evidence")
            trace.skip("generate", "no_evidence")
            trace.answer = REFUSAL_MESSAGE
            trace.refused = True
            trace.action = "refused_no_evidence"
            return
    else:
        trace.skip("crag", "disabled")

    trace.set_chunks(chunks)

    # ---- compress + context ---------------------------------------------
    context = None

    if config.use_compression and chunks:
        with trace.stage("compress") as stage:
            info = compress_context_detailed(search_query, chunks)
            _add_usage(trace, info["usage"])
            context = info["context"]
            stage.status = _status(info["fallback_reason"])
            stage.details.update(
                original_chars=info["original_chars"],
                compressed_chars=info["compressed_chars"],
                ratio=info["ratio"],
                per_chunk=info["per_chunk"],
                unverified_lines=info["unverified_lines"],
                fallback_reason=info["fallback_reason"],
            )
    else:
        trace.skip("compress", "disabled" if not config.use_compression else "no_chunks")

    with trace.stage("context") as stage:
        if context is None:
            context = build_context(chunks)

        stage.details.update(chunks=len(chunks), chars=len(context))

    trace.set_texts(context=context)

    # ---- generate --------------------------------------------------------
    with trace.stage("generate") as stage:
        result = generate_answer(
            llm, search_query, context, cite_sources=config.cite_sources
        )

        if result["llm_called"]:
            _add_usage(trace, result["usage"])

        stage.details.update(
            model=result["model"],
            llm_called=result["llm_called"],
            llm_latency_ms=result["latency_ms"],
            refused=result["refused"],
        )

        if result["empty"]:
            stage.status = "error"
            stage.error = "LLM returned an empty answer"

    trace.set_texts(prompt=result["prompt"])
    trace.answer = result["answer"]
    trace.refused = result["refused"]

    if result["empty"]:
        trace.action = "error"
        trace.error = "LLM returned an empty answer"
    elif result["refused"]:
        trace.action = "refused_by_llm" if result["llm_called"] else "refused_no_context"
    else:
        trace.action = "answered"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(
    question,
    config=None,
    chat_history=None,
    run_id=None,
    save=True,
    raise_errors=True,
    llm=None,
):
    """Run one question through the configured pipeline and return its Trace.

    config        PipelineConfig, preset name ("hybrid_rerank"), dict or None
    chat_history  list of {"role", "content"} (used when rewrite_with_history)
    run_id        groups the trace with an evaluation run
    save          store the trace in the TraceStore
    raise_errors  True: re-raise after saving the failed trace (API);
                  False: return the Trace with action "error" (eval runs)
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    config = resolve_config(config)
    question = question.strip()
    trace = Trace.new(question, config)

    try:
        _run(trace, question, config, chat_history, llm)
    except Exception as error:
        logger.error("Pipeline failed: %s", error)
        trace.error = f"{type(error).__name__}: {error}"
        trace.action = "error"
        _finalize(trace, save, run_id)

        if raise_errors:
            raise

        return trace

    return _finalize(trace, save, run_id)