"""
Evaluation runner: one pipeline configuration, the whole golden dataset, one run file.

Place this file at: app/evaluation/runner.py

    run = run_evaluation("hybrid_rerank")           # preset name, dict or PipelineConfig
    run["summary"]["retrieval"]["all"]["hit@5"]

What one run does, per question
    1. runs the pipeline (the trace is saved in the TraceStore under this run_id)
    2. retrieval metrics on what the RETRIEVE stage returned  (up to retrieval_depth
       chunks, so Hit@10 is measurable even when the LLM only gets top_k)
    3. the same metrics on the chunks the LLM actually RECEIVED ("context")
    4. the LLM judges (faithfulness, correctness, refusal)         [optional]
    5. a DIAGNOSIS: why did this question succeed or fail?

Diagnosis (the point of the whole project)
    ok                answerable, judged CORRECT
    partial           answerable, judged PARTIAL
    false_refusal     answerable, but the system refused
    retrieval_miss    answer wrong, the evidence was never retrieved
    lost_before_llm   answer wrong, the evidence WAS retrieved but did not reach the
                      LLM (cut by top_k, rerank or CRAG)
    generation_error  answer wrong although the evidence was in the LLM's context
    correct_refusal   unanswerable, the system refused
    hallucination     unanswerable, the system answered anyway
    pipeline_error    the pipeline crashed (counted as a failure, never dropped)
    unjudged          no judge verdict available

No silent inflation: a question whose pipeline crashed before retrieval counts as
a retrieval miss (all zeros), not as "missing data"; end-to-end rates divide by
ALL answerable questions, not only the ones that were judged.

The run file (evaluation/runs/run_NNN.json) is saved after EVERY question, so a
crash or Ctrl-C loses nothing: run_evaluation(..., run_id="run_003", resume=True)
continues where it stopped.
"""

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from app.config import EVAL_DIR, EVAL_K_VALUES, MAX_TOP_K, RUNS_DIR
from app.evaluation import metrics as M
from app.evaluation.dataset import load_dataset
from app.evaluation.judges import Judge, JudgeCache, aggregate_judgments
from app.pipeline import resolve_config, run_pipeline
from app.tracing.trace_store import get_store

logger = logging.getLogger(__name__)

RUN_FORMAT_VERSION = 1
DEFAULT_MATCH_THRESHOLD = 0.8
DEFAULT_LOW_RANK = 3
JUDGE_CACHE_PATH = EVAL_DIR / "judge_cache.jsonl"


DIAGNOSES = (
    "ok",
    "partial",
    "false_refusal",
    "retrieval_miss",
    "lost_before_llm",
    "generation_error",
    "correct_refusal",
    "hallucination",
    "pipeline_error",
    "provider_rate_limit",
    "unjudged",
)

class EvaluationSetupError(RuntimeError):
    """The evaluation cannot give trustworthy numbers (wrong PDF, run id clash...)."""


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def run_path(run_id, runs_dir=RUNS_DIR):
    return Path(runs_dir) / f"{run_id}.json"


def next_run_id(runs_dir=RUNS_DIR):
    """run_001, run_002 ... (one more than the highest existing number)."""
    highest = 0
    directory = Path(runs_dir)

    if directory.exists():
        for path in directory.glob("run_*.json"):
            match = re.fullmatch(r"run_(\d+)", path.stem)

            if match:
                highest = max(highest, int(match.group(1)))

    return f"run_{highest + 1:03d}"


def load_run(run, runs_dir=RUNS_DIR):
    """Load a run by id ("run_003"), by path, or pass an already loaded dict."""
    if isinstance(run, dict):
        return run

    path = Path(run)

    if not path.suffix:
        path = run_path(str(run), runs_dir)

    if not path.exists():
        raise FileNotFoundError(f"Run file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if data.get("format_version") != RUN_FORMAT_VERSION:
        raise ValueError(
            f"{path.name}: unsupported run format {data.get('format_version')!r} "
            f"(this code reads version {RUN_FORMAT_VERSION})."
        )

    return data


def list_runs(runs_dir=RUNS_DIR):
    """Short rows for every run file, newest first (dashboard / CLI listing)."""
    rows = []
    directory = Path(runs_dir)

    if not directory.exists():
        return rows

    for path in directory.glob("run_*.json"):
        try:
            run = load_run(path)
        except (ValueError, json.JSONDecodeError, OSError):
            continue

        summary = run.get("summary") or {}
        retrieval = (summary.get("retrieval") or {}).get("all") or {}
        end_to_end = summary.get("end_to_end") or {}

        rows.append({
            "run_id": run["run_id"],
            "label": run.get("label"),
            "status": run.get("status"),
            "created_at": run.get("created_at"),
            "questions": len(run.get("rows", [])),
            "hit@5": retrieval.get("hit@5"),
            "mrr": retrieval.get("mrr"),
            "answerable_ok_rate": end_to_end.get("answerable_ok_rate"),
            "unanswerable_ok_rate": end_to_end.get("unanswerable_ok_rate"),
        })

    return sorted(rows, key=lambda r: (r["created_at"] or "", r["run_id"]), reverse=True)


# ---------------------------------------------------------------------------
# Per-question evaluation
# ---------------------------------------------------------------------------

def _slim(result):
    if not result:
        return None

    return {
        key: result.get(key)
        for key in ("verdict", "score", "reason", "passed", "claims", "supported", "fallback_reason")
        if key in result
    }


def compact_judgment(judged):
    """The judge_trace() result without usage/latency noise, still readable by
    aggregate_judgments()."""
    return {
        "skipped": judged.get("skipped"),
        "expected": judged["expected"],
        "refused": judged["refused"],
        "faithfulness": _slim(judged.get("faithfulness")),
        "correctness": _slim(judged.get("correctness")),
        "llm_calls": judged.get("llm_calls", 0),
    }


def diagnose_row(row):
    """Classify an evaluation row without mislabeling provider failures."""

    error_text = str(row.get("error") or "").lower()
    action = str(row.get("action") or "").lower()

    # Provider quota/rate-limit failures are not retrieval failures.
    rate_limit_indicators = (
        "ratelimiterror",
        "rate limit",
        "rate_limit",
        "tokens per day",
        "token limit",
        "too many requests",
        "http 429",
        "status code 429",
    )

    if any(term in error_text for term in rate_limit_indicators):
        return "provider_rate_limit"

    if row.get("error") or action == "error":
        return "pipeline_error"

    judge = row.get("judge")
    refused = row["refused"]

    if judge and judge.get("refused") is not None:
        refused = judge["refused"]

    if row["expected"] == "unanswerable":
        return "correct_refusal" if refused else "hallucination"

    if refused:
        return "false_refusal"

    verdict = ((judge or {}).get("correctness") or {}).get("verdict")

    if verdict == "CORRECT":
        return "ok"

    if verdict == "PARTIAL":
        return "partial"

    if verdict != "INCORRECT":
        return "unjudged"

    def evidence_found(result):
        return bool(result) and all(
            rank is not None
            for rank in result["group_ranks"].values()
        )

    if evidence_found(row.get("context")):
        return "generation_error"

    return (
        "lost_before_llm"
        if evidence_found(row.get("retrieval"))
        else "retrieval_miss"
    )


def _evaluate_question(question, config, run_id, store, judge, ks, threshold):
    trace = run_pipeline(
        question.question, config, run_id=run_id, save=False, raise_errors=False
    )
    store.save(trace, run_id)  # strict: a trace that cannot be saved must not go unnoticed

    row = {
        "question_id": question.id,
        "question": question.question,
        "type": question.type,
        "tags": list(question.tags),
        "expected": question.expected,
        "trace_id": trace.trace_id,
        "action": trace.action,
        "refused": bool(trace.refused),
        "answer": trace.answer,
        "error": trace.error,
        "latency_ms": trace.total_latency_ms,
        "tokens": trace.usage.get("total_tokens", 0),
        "llm_calls": trace.usage.get("llm_calls", 0),
        "retrieval": None,
        "retrieval_unavailable": False,
        "context": None,
        "judge": None,
        "judge_error": None,
    }

    if question.answerable:
        try:
            candidates = M.texts_from_trace(trace, "retrieve")
        except ValueError:  # the pipeline failed before/inside retrieval
            candidates = []
            row["retrieval_unavailable"] = True

        row["retrieval"] = M.evaluate_retrieval(question, candidates, ks, threshold)
        row["context"] = M.evaluate_retrieval(
            question, M.texts_from_trace(trace, "final"), ks, threshold
        )

    if judge is not None:
        try:
            row["judge"] = compact_judgment(judge.judge_trace(trace, question))
        except Exception as error:  # a broken judge must not stop the run
            logger.warning("Judge failed for question %s: %s", question.id, error)
            row["judge_error"] = f"{type(error).__name__}: {error}"

    row["diagnosis"] = diagnose_row(row)

    return row


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _by_tag(rows, key):
    groups = {}

    for row in rows:
        if row[key] is None:
            continue

        for tag in row["tags"]:
            groups.setdefault(tag, []).append(row[key])

    return {tag: M.aggregate(items) for tag, items in sorted(groups.items())}

def summarize(
    rows,
    ks=EVAL_K_VALUES,
    low_rank=DEFAULT_LOW_RANK,
    unreachable_ids=(),
    judge_enabled=None,
):
    """All numbers of a run, computed from the per-question rows."""

    if judge_enabled is None:
        judge_enabled = any(r.get("judge") is not None for r in rows)

    answerable = [r for r in rows if r["expected"] == "answerable"]
    unanswerable = [r for r in rows if r["expected"] == "unanswerable"]
    retrieval_rows = [r["retrieval"] for r in rows if r["retrieval"]]
    context_rows = [r["context"] for r in rows if r["context"]]
    judged = [r["judge"] for r in rows if r["judge"]]
    unreachable = set(unreachable_ids)
    
    by_action = {}
    for r in rows:
        by_action[r["action"] or "unknown"] = by_action.get(r["action"] or "unknown", 0) + 1

    diagnosis = {name: {"count": 0, "ids": []} for name in DIAGNOSES}
    for r in rows:
        diagnosis[r["diagnosis"]]["count"] += 1
        diagnosis[r["diagnosis"]]["ids"].append(r["question_id"])

    outcome_ids = {"miss": [], "partial": [], "low_rank": []}
    for r in rows:
        label = M.classify_retrieval(r["retrieval"], low_rank)
        if label in outcome_ids:
            outcome_ids[label].append(r["question_id"])

    ok = sum(1 for r in answerable if r["diagnosis"] == "ok")
    refused_ok = sum(1 for r in unanswerable if r["diagnosis"] == "correct_refusal")

    ci = {}
    for k in ks:
        ci[f"retrieval.hit@{k}"] = M.mean_ci([r[f"hit@{k}"] for r in retrieval_rows])
        ci[f"retrieval.recall@{k}"] = M.mean_ci([r[f"recall@{k}"] for r in retrieval_rows])
    ci["retrieval.mrr"] = M.mean_ci([r["mrr"] for r in retrieval_rows])
    ci["end_to_end.answerable_ok"] = (
        M.mean_ci([1.0 if r["diagnosis"] == "ok" else 0.0 for r in answerable]
    )
        if judge_enabled
        else M.mean_ci([])
)
    ci["end_to_end.unanswerable_ok"] = M.mean_ci(
        [1.0 if r["diagnosis"] == "correct_refusal" else 0.0 for r in unanswerable]
    )
    if judged:
        ci["answer.correctness"] = M.mean_ci(
            [j["correctness"]["score"] for j in judged if j.get("correctness") and j["correctness"].get("score") is not None]
        )
        ci["answer.faithfulness"] = M.mean_ci(
            [j["faithfulness"]["score"] for j in judged if j.get("faithfulness") and j["faithfulness"].get("score") is not None]
        )

    shims = [
        SimpleNamespace(
            total_latency_ms=r["latency_ms"],
            usage={"total_tokens": r["tokens"], "llm_calls": r["llm_calls"]},
            action=r["action"],
        )
        for r in rows
    ]

    return {
        "counts": {
            "questions": len(rows),
            "answerable": len(answerable),
            "unanswerable": len(unanswerable),
            "pipeline_errors": diagnosis["pipeline_error"]["count"],
            "provider_rate_limits": diagnosis["provider_rate_limit"]["count"],
            "retrieval_unavailable": sum(1 for r in rows if r["retrieval_unavailable"]),
            "by_action": by_action,
        },
        "retrieval": {
            "all": M.aggregate(retrieval_rows),
            "by_type": M.aggregate_by_type(retrieval_rows),
            "by_tag": _by_tag(rows, "retrieval"),
            "outcomes": M.outcome_counts(retrieval_rows, low_rank),
            "outcome_ids": outcome_ids,
            "reachable": M.aggregate([r["retrieval"] for r in rows if r["retrieval"] and r["question_id"] not in unreachable]),
        },
        "context": {
            "all": M.aggregate(context_rows),
            "outcomes": M.outcome_counts(context_rows, low_rank),
        },
        "answer": aggregate_judgments(judged) if judged else None,
        "end_to_end": {
            "answerable_ok_rate": ok / len(answerable) if judge_enabled and answerable else None,
            "unanswerable_ok_rate": refused_ok / len(unanswerable) if unanswerable else None,
            "diagnosis": diagnosis,
        },
        "refusal_flag": M.refusal_metrics([{"expected": r["expected"], "refused": r["refused"]} for r in rows]),
        "efficiency": M.efficiency_metrics(shims),
        "judge_llm_calls": sum(j.get("llm_calls", 0) for j in judged),
        "ci": ci,
        "unreachable_ids": sorted(unreachable),
    }


# ---------------------------------------------------------------------------
# Setup checks
# ---------------------------------------------------------------------------

def _check_collection(config, dataset, threshold):
    """Is the dataset really about the PDF in this collection? Which questions
    can no chunk of this chunking ever answer?"""
    from app.retrieval.bm25 import load_chunks_from_collection

    try:
        chunks = load_chunks_from_collection(config.collection_name)
    except ValueError as error:
        raise EvaluationSetupError(str(error)) from error

    doc_ids = {c.doc_id for c in chunks if c.doc_id}

    if dataset.doc_id and doc_ids and dataset.doc_id not in doc_ids:
        raise EvaluationSetupError(
            f"The dataset was written for document {dataset.doc_id}, but this collection holds "
            f"{sorted(doc_ids)}. Metrics would be meaningless. Ingest the right PDF, or pass "
            "check_collection=False if you know what you are doing."
        )

    unreachable = M.unreachable_questions(dataset, [c.text for c in chunks], threshold)

    return unreachable, {"doc_ids": sorted(doc_ids), "chunks": len(chunks)}


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def run_evaluation(
    config=None,
    dataset=None,
    run_id=None,
    judge=None,
    use_judge=True,
    ks=EVAL_K_VALUES,
    match_threshold=DEFAULT_MATCH_THRESHOLD,
    low_rank=DEFAULT_LOW_RANK,
    runs_dir=RUNS_DIR,
    store=None,
    resume=False,
    question_delay=0.0,
    check_collection=True,
    label=None,
    progress=None,
):
    """Evaluate ONE configuration on the dataset. Returns the run dict (also
    saved as <runs_dir>/<run_id>.json after every question).

    config        preset name, dict or PipelineConfig (None = baseline vector search)
    dataset       a Dataset (default: evaluation/dataset.json); use dataset.filter(...) for a subset
    judge         a Judge; default builds one with an on-disk cache. use_judge=False skips all LLM judging
    match_threshold  evidence must be at least this much inside a chunk (0.8 tolerates
                  a snippet that a chunk boundary cut slightly; 1.0 = strict)
    resume        continue an interrupted run with the same run_id
    progress      function(index, total, question, row), called after every question
    """
    ks = tuple(sorted(set(ks)))
    dataset = dataset if dataset is not None else load_dataset()

    if not len(dataset):
        raise EvaluationSetupError("The dataset has no questions.")

    base = resolve_config(config)
    depth = min(max(max(ks), base.top_k), MAX_TOP_K)
    eval_config = base.with_changes(retrieval_depth=depth)

    run_id = run_id or next_run_id(runs_dir)
    path = run_path(run_id, runs_dir)
    store = store or get_store()

    unreachable, collection_info = [], None

    if check_collection:
        unreachable, collection_info = _check_collection(base, dataset, match_threshold)

    if use_judge and judge is None:
        judge = Judge(cache=JudgeCache(JUDGE_CACHE_PATH))

    judge = judge if use_judge else None

    settings = {
        "ks": list(ks),
        "match_threshold": match_threshold,
        "low_rank": low_rank,
        "judge": judge is not None,
        "judge_model": judge._model_name() if judge is not None else None,
        "retrieval_depth": depth,
    }

    if path.exists():
        if not resume:
            raise EvaluationSetupError(
                f"{path.name} already exists. Use resume=True to continue it, or choose another run_id."
            )

        run = load_run(path)
        problems = []

        if run["config_fingerprint"] != base.fingerprint():
            problems.append("the configuration is different")

        if run["dataset"]["doc_id"] != dataset.doc_id:
            problems.append("the dataset is for a different document")

        for key in ("ks", "match_threshold", "judge", "judge_model"):
            if run["settings"].get(key) != settings[key]:
                problems.append(f"setting '{key}' is different")

        if problems:
            raise EvaluationSetupError(f"Cannot resume {path.name}: " + "; ".join(problems) + ".")

        run["status"] = "running"
    else:
        run = {
            "format_version": RUN_FORMAT_VERSION,
            "run_id": run_id,
            "status": "running",
            "created_at": _now(),
            "finished_at": None,
            "label": label or base.label(),
            "config": base.to_dict(),
            "config_fingerprint": base.fingerprint(),
            "dataset": {
                "source": dataset.source,
                "doc_id": dataset.doc_id,
                "n": len(dataset),
                "ids": [q.id for q in dataset],
                "reviewed": sum(1 for q in dataset if q.reviewed),
            },
            "collection": collection_info,
            "settings": settings,
            "rows": [],
            "summary": None,
        }

    done = {row["question_id"] for row in run["rows"]}
    total = len(dataset)

    try:
        for index, question in enumerate(dataset, start=1):
            if question.id in done:
                continue

            row = _evaluate_question(question, eval_config, run_id, store, judge, ks, match_threshold)
            run["rows"].append(row)
            _write_json(path, run)

            if progress:
                progress(index, total, question, row)

            if question_delay:
                time.sleep(question_delay)
    except KeyboardInterrupt:
        run["status"] = "interrupted"
        _write_json(path, run)
        raise

    run["rows"].sort(key=lambda r: r["question_id"])
    run["summary"] = summarize(
    run["rows"],
    ks,
    low_rank,
    unreachable,
    judge_enabled=judge is not None,
)
    run["status"] = "complete"
    run["finished_at"] = _now()
    _write_json(path, run)

    return run