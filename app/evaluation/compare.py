"""
Compare evaluation runs: "did changing the setting really help?"

Place this file at: app/evaluation/compare.py

Two runs are compared QUESTION BY QUESTION (paired): the same question is looked
at in both runs, and a bootstrap confidence interval is computed on the
per-question differences. That is far more sensitive than comparing two averages,
and it tells you whether a difference is real ("significant": the interval does
not include 0) or just noise from a small dataset.

    cmp = compare_runs("run_001", "run_002")     # B vs A (A is the baseline)
    cmp["metrics"][0]  ->  {"metric": "retrieval.hit@5", "a": 0.65, "b": 0.78,
                             "diff": 0.13, "lo": 0.04, "hi": 0.22, "significant": True, ...}

Metric names are "family.name":
    retrieval.hit@K / recall@K / ndcg@K / precision@K / all_groups@K / mrr
    context.hit@K ...            (same metrics on the chunks the LLM received)
    answer.correctness           judged score 1 / 0.5 / 0 (judged answerable questions)
    answer.faithfulness          judged score 0..1
    end_to_end.answerable_ok     1 if the answer was judged CORRECT (all answerable questions)
    end_to_end.unanswerable_ok   1 if the system correctly refused (all unanswerable questions)
    efficiency.latency_ms / efficiency.tokens   (lower is better)
"""

from app.evaluation.metrics import paired_difference
from app.evaluation.runner import load_run
from app.pipeline_config import PipelineConfig

# how bad is a diagnosis? (lower = better) used to say "improved" / "worsened"
SEVERITY = {
    "ok": 0, "correct_refusal": 0,
    "partial": 1,
    "unjudged": 2,
    "lost_before_llm": 3, "generation_error": 3,
    "retrieval_miss": 4, "false_refusal": 4, "hallucination": 4,
    "pipeline_error": 5,
}

LOWER_IS_BETTER = {"efficiency.latency_ms", "efficiency.tokens"}


def default_metrics(run):
    """The headline metrics for a run (depends on its k values and on judging)."""
    ks = run["settings"]["ks"]
    top = [k for k in ks if k in (1, 5)] or ks[:1]
    names = [f"retrieval.hit@{k}" for k in top] + [f"retrieval.recall@{max(ks)}", "retrieval.mrr", f"retrieval.ndcg@{max(ks)}"]

    if run["settings"].get("judge"):
        names += ["answer.correctness", "answer.faithfulness"]

    names += ["end_to_end.answerable_ok", "end_to_end.unanswerable_ok", "efficiency.latency_ms", "efficiency.tokens"]

    return names


def question_values(run, metric):
    """{question_id: value} for one metric. Questions the metric does not apply
    to (e.g. retrieval metrics of unanswerable questions) are absent."""
    family, _, name = metric.partition(".")
    values = {}

    for row in run["rows"]:
        qid = row["question_id"]

        if family in ("retrieval", "context"):
            result = row.get(family)

            if result and name in result:
                values[qid] = result[name]

        elif family == "answer":
            judge = row.get("judge") or {}
            part = judge.get({"correctness": "correctness", "faithfulness": "faithfulness"}.get(name, name)) or {}

            if part.get("score") is not None:
                values[qid] = part["score"]

        elif family == "end_to_end":
            if name == "answerable_ok" and row["expected"] == "answerable":
                values[qid] = 1.0 if row["diagnosis"] == "ok" else 0.0
            elif name == "unanswerable_ok" and row["expected"] == "unanswerable":
                values[qid] = 1.0 if row["diagnosis"] == "correct_refusal" else 0.0

        elif family == "efficiency":
            key = {"latency_ms": "latency_ms", "tokens": "tokens"}.get(name)

            if key:
                values[qid] = row[key]

        else:
            raise ValueError(f"Unknown metric '{metric}'.")

    return values


def comparability_warnings(run_a, run_b):
    """Reasons why the two runs may not be directly comparable."""
    warnings = []

    for label, run in (("A", run_a), ("B", run_b)):
        if run.get("status") != "complete":
            warnings.append(f"Run {label} ({run['run_id']}) is '{run.get('status')}', not complete.")

    if run_a["dataset"]["doc_id"] != run_b["dataset"]["doc_id"]:
        warnings.append("The runs used datasets for different documents.")

    ids_a, ids_b = {r["question_id"] for r in run_a["rows"]}, {r["question_id"] for r in run_b["rows"]}

    if ids_a != ids_b:
        warnings.append(
            f"The runs cover different questions ({len(ids_a)} vs {len(ids_b)}); "
            f"only the {len(ids_a & ids_b)} common ones are compared."
        )

    for key, text in (("match_threshold", "evidence match threshold"), ("ks", "k values")):
        if run_a["settings"].get(key) != run_b["settings"].get(key):
            warnings.append(f"Different {text}: {run_a['settings'].get(key)} vs {run_b['settings'].get(key)}.")

    if run_a["settings"].get("judge") != run_b["settings"].get("judge"):
        warnings.append("Only one of the runs used the LLM judge; answer metrics cannot be compared.")
    elif run_a["settings"].get("judge_model") != run_b["settings"].get("judge_model"):
        warnings.append(
            f"Different judge models ({run_a['settings'].get('judge_model')} vs "
            f"{run_b['settings'].get('judge_model')}): answer scores are not on the same scale."
        )

    if run_a["dataset"].get("reviewed") != run_b["dataset"].get("reviewed"):
        warnings.append("The datasets have a different number of human-reviewed questions.")

    return warnings


def config_diff(run_a, run_b):
    """{setting: (value in A, value in B)} for every setting that differs."""
    a, b = PipelineConfig.from_dict(run_a["config"]), PipelineConfig.from_dict(run_b["config"])

    return b.diff_from(a)


def diagnosis_changes(run_a, run_b):
    """Which questions got better or worse, by diagnosis."""
    by_a = {r["question_id"]: r["diagnosis"] for r in run_a["rows"]}
    by_b = {r["question_id"]: r["diagnosis"] for r in run_b["rows"]}
    improved, worsened, same = [], [], 0

    for qid in sorted(set(by_a) & set(by_b)):
        before, after = by_a[qid], by_b[qid]

        if before == after:
            same += 1
            continue

        item = {"question_id": qid, "from": before, "to": after}
        change = SEVERITY[after] - SEVERITY[before]

        if change < 0:
            improved.append(item)
        elif change > 0:
            worsened.append(item)
        else:
            same += 1  # a different failure label of the same severity

    return {"improved": improved, "worsened": worsened, "unchanged": same}


def compare_runs(run_a, run_b, metrics=None, confidence=0.95, runs_dir=None):
    """Compare run B against run A (A = baseline). Returns a dict (see module doc)."""
    kwargs = {"runs_dir": runs_dir} if runs_dir else {}
    run_a, run_b = load_run(run_a, **kwargs), load_run(run_b, **kwargs)
    names = metrics or [m for m in default_metrics(run_a) if m in default_metrics(run_b)]

    result_rows = []

    for name in names:
        values_a, values_b = question_values(run_a, name), question_values(run_b, name)
        common = sorted(set(values_a) & set(values_b))
        test = paired_difference(values_a, values_b, confidence=confidence)
        higher_is_better = name not in LOWER_IS_BETTER

        if test["mean_diff"] is None:
            better = None
        elif not test["significant"]:
            better = None
        else:
            better = (test["mean_diff"] > 0) == higher_is_better

        result_rows.append({
            "metric": name,
            "n": len(common),
            "a": sum(values_a[i] for i in common) / len(common) if common else None,
            "b": sum(values_b[i] for i in common) / len(common) if common else None,
            "diff": test["mean_diff"],
            "lo": test["lo"],
            "hi": test["hi"],
            "significant": test["significant"],
            "b_is_better": better,  # True / False when the difference is significant, else None
        })

    return {
        "a": run_a["run_id"],
        "b": run_b["run_id"],
        "label_a": run_a.get("label"),
        "label_b": run_b.get("label"),
        "config_diff": config_diff(run_a, run_b),
        "warnings": comparability_warnings(run_a, run_b),
        "metrics": result_rows,
        "diagnosis_changes": diagnosis_changes(run_a, run_b),
    }


def leaderboard(runs, metric="end_to_end.answerable_ok", runs_dir=None):
    """Rank runs by one metric (best first, using each run's own mean)."""
    kwargs = {"runs_dir": runs_dir} if runs_dir else {}
    rows = []

    for item in runs:
        run = load_run(item, **kwargs)
        values = list(question_values(run, metric).values())

        rows.append({
            "run_id": run["run_id"],
            "label": run.get("label"),
            "mean": sum(values) / len(values) if values else None,
            "n": len(values),
        })

    reverse = metric not in LOWER_IS_BETTER

    return sorted(rows, key=lambda r: (r["mean"] is None, -(r["mean"] or 0) if reverse else (r["mean"] or 0)))