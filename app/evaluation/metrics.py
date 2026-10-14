"""
Evaluation metrics. Pure functions: no LLM, no API, no database.

Place this file at: app/evaluation/metrics.py

Input is a ranked list of chunk TEXTS (best first) and a dataset Question.
A chunk is "relevant" when it contains an evidence snippet of the question.
Evidence snippets come in GROUPS: one group = one required fact.

Per-question retrieval metrics (answerable questions only), for each k:

  hit@k        1 if at least ONE evidence group is found in the top k
  recall@k     fraction of the evidence groups found in the top k
  all_groups@k 1 if EVERY group is found in the top k
               (for multi-hop questions this is "could the LLM answer from
               these chunks?")
  precision@k  relevant chunks in the top k / k  (missing results count as
               not relevant, so a retriever that returns fewer than k chunks
               is not rewarded)
  ndcg@k       ordering quality. Gain of a chunk = number of NEW groups it
               covers; position i is discounted by 1/log2(i+1). The ideal is one
               new group per chunk at ranks 1, 2, 3... Result is capped at 1.
  mrr          1 / rank of the first relevant chunk (0 if none; no k cut-off)

Examples (one group, 5 results):
  relevant chunk at rank 1 -> hit@1 = 1, mrr = 1.0,  ndcg@5 = 1.0
  relevant chunk at rank 3 -> hit@1 = 0, hit@3 = 1, mrr = 0.33, ndcg@5 = 0.5
  nothing relevant          -> every metric 0, outcome "miss"

Aggregates are means over answerable questions. With a small dataset every
mean is noisy, so mean_ci() gives a bootstrap confidence interval and
paired_difference() compares two configurations on the SAME questions.
"""

import math
import random

from app.config import EVAL_K_VALUES

_NON_METRIC_KEYS = {
    "question_id",
    "type",
    "n_groups",
    "returned",
    "group_ranks",
    "first_relevant_rank",
    "last_group_rank",
}


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def group_ranks(question, texts, threshold=1.0):
    """{group: rank of the first text containing that group, or None}."""
    ranks = {number: None for number in question.groups}

    for rank, text in enumerate(texts, start=1):
        for number, items in question.groups.items():
            if ranks[number] is None and any(
                item.matches(text, threshold) for item in items
            ):
                ranks[number] = rank

    return ranks


def _log_discount(rank):
    return 1.0 / math.log2(rank + 1)


def evaluate_retrieval(question, texts, ks=EVAL_K_VALUES, threshold=1.0):
    """All retrieval metrics for ONE question. Returns None for unanswerable
    questions (there is nothing to find). `texts` is the ranked chunk list."""
    if not question.answerable:
        return None

    ranks = group_ranks(question, texts, threshold)
    n_groups = len(ranks)
    found_ranks = [rank for rank in ranks.values() if rank is not None]
    first = min(found_ranks) if found_ranks else None

    # which chunks contain any evidence at all (for precision)
    relevant = [bool(question.groups_found([text], threshold)) for text in texts]

    row = {
        "question_id": question.id,
        "type": question.type,
        "n_groups": n_groups,
        "returned": len(texts),
        "group_ranks": {str(number): rank for number, rank in ranks.items()},
        "first_relevant_rank": first,
        "last_group_rank": max(found_ranks) if len(found_ranks) == n_groups else None,
        "mrr": 1.0 / first if first else 0.0,
    }

    for k in ks:
        in_top_k = [rank for rank in found_ranks if rank <= k]

        gains = {}
        for rank in in_top_k:
            gains[rank] = gains.get(rank, 0) + 1

        dcg = sum(gain * _log_discount(rank) for rank, gain in gains.items())
        ideal = sum(_log_discount(rank) for rank in range(1, min(n_groups, k) + 1))

        row[f"hit@{k}"] = 1.0 if in_top_k else 0.0
        row[f"recall@{k}"] = len(in_top_k) / n_groups
        row[f"all_groups@{k}"] = 1.0 if len(in_top_k) == n_groups else 0.0
        row[f"precision@{k}"] = sum(relevant[:k]) / k
        row[f"ndcg@{k}"] = min(1.0, dcg / ideal) if ideal else 0.0

    return row


def classify_retrieval(row, low_rank=3):
    """Failure bucket for the dashboard.

      n/a       unanswerable question
      miss      no evidence group found anywhere in the list
      partial   some groups found, some not (multi-hop)
      low_rank  everything found, but at least one group below rank `low_rank`
                (a reranker or hybrid search could help)
      ok        everything found at rank <= low_rank
    """
    if row is None:
        return "n/a"

    found = [rank for rank in row["group_ranks"].values() if rank is not None]

    if not found:
        return "miss"

    if len(found) < row["n_groups"]:
        return "partial"

    return "low_rank" if max(found) > low_rank else "ok"


def unreachable_questions(dataset, chunk_texts, threshold=1.0):
    """Ids of answerable questions whose evidence can NOT be found inside any
    single chunk of this chunking (e.g. a chunk boundary cut the snippet).

    No retriever can score on those, so when comparing chunking strategies
    compare on the questions reachable under EVERY strategy, or report this
    list next to the metrics."""
    return [
        q.id
        for q in dataset
        if q.answerable
        and len(q.groups_found(chunk_texts, threshold)) < len(q.groups)
    ]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(rows):
    """Mean of every metric over the (non-None) rows, plus n."""
    rows = [row for row in rows if row is not None]
    result = {"n": len(rows)}

    if not rows:
        return result

    keys = [
        key for key in rows[0]
        if key not in _NON_METRIC_KEYS and isinstance(rows[0][key], (int, float))
    ]

    for key in keys:
        result[key] = sum(row[key] for row in rows) / len(rows)

    return result


def aggregate_by_type(rows):
    groups = {}

    for row in rows:
        if row is not None:
            groups.setdefault(row["type"], []).append(row)

    return {name: aggregate(items) for name, items in sorted(groups.items())}


def aggregate_by_tag(rows, dataset):
    tags = {q.id: q.tags for q in dataset}
    groups = {}

    for row in rows:
        if row is None:
            continue

        for tag in tags.get(row["question_id"], []):
            groups.setdefault(tag, []).append(row)

    return {name: aggregate(items) for name, items in sorted(groups.items())}


def outcome_counts(rows, low_rank=3):
    counts = {"ok": 0, "low_rank": 0, "partial": 0, "miss": 0}

    for row in rows:
        label = classify_retrieval(row, low_rank)

        if label in counts:
            counts[label] += 1

    return counts


# ---------------------------------------------------------------------------
# Uncertainty (small datasets are noisy)
# ---------------------------------------------------------------------------

def mean_ci(values, confidence=0.95, n_boot=2000, seed=0):
    """Bootstrap confidence interval of the mean. Deterministic (fixed seed)."""
    values = [v for v in values if v is not None]

    if not values:
        return {"mean": None, "lo": None, "hi": None, "n": 0}

    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_boot))
    low = means[int((1 - confidence) / 2 * n_boot)]
    high = means[min(n_boot - 1, int((1 + confidence) / 2 * n_boot))]

    return {"mean": sum(values) / n, "lo": low, "hi": high, "n": n}


def paired_difference(a_by_id, b_by_id, confidence=0.95, n_boot=2000, seed=0):
    """Is configuration B better than A? Both are {question_id: value}.
    Only questions present in both are used, and they are resampled TOGETHER
    (paired), which is much more sensitive than comparing two separate means.

    significant = the confidence interval of (B - A) does not include 0."""
    ids = sorted(set(a_by_id) & set(b_by_id))
    differences = [
        b_by_id[i] - a_by_id[i]
        for i in ids
        if a_by_id[i] is not None and b_by_id[i] is not None
    ]

    ci = mean_ci(differences, confidence, n_boot, seed)

    if not differences:
        return {"mean_diff": None, "lo": None, "hi": None, "n": 0, "significant": False}

    return {
        "mean_diff": ci["mean"],
        "lo": ci["lo"],
        "hi": ci["hi"],
        "n": ci["n"],
        "significant": ci["lo"] > 0 or ci["hi"] < 0,
    }


# ---------------------------------------------------------------------------
# Refusal and efficiency (no LLM needed)
# ---------------------------------------------------------------------------

def refusal_metrics(rows):
    """rows: dicts with "expected" ("answerable"/"unanswerable") and "refused".

    correct_refusal_rate  of the unanswerable questions, how many were refused
    hallucination_rate    1 - correct_refusal_rate: an answer was invented
    false_refusal_rate    of the answerable questions, how many were refused
    """
    unanswerable = [r for r in rows if r["expected"] == "unanswerable"]
    answerable = [r for r in rows if r["expected"] == "answerable"]

    correct = sum(1 for r in unanswerable if r["refused"])
    false = sum(1 for r in answerable if r["refused"])

    return {
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "correct_refusal_rate": correct / len(unanswerable) if unanswerable else None,
        "hallucination_rate": 1 - correct / len(unanswerable) if unanswerable else None,
        "false_refusal_rate": false / len(answerable) if answerable else None,
    }


def _percentile(sorted_values, fraction):
    if not sorted_values:
        return 0

    index = min(len(sorted_values) - 1, round(fraction * (len(sorted_values) - 1)))

    return sorted_values[index]


def efficiency_metrics(traces):
    """Latency, token and LLM-call statistics from a list of Trace objects."""
    if not traces:
        return {"n": 0}

    latencies = sorted(t.total_latency_ms for t in traces)
    n = len(traces)

    return {
        "n": n,
        "avg_latency_ms": round(sum(latencies) / n),
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "avg_tokens": round(sum(t.usage.get("total_tokens", 0) for t in traces) / n),
        "avg_llm_calls": round(sum(t.usage.get("llm_calls", 0) for t in traces) / n, 2),
        "errors": sum(1 for t in traces if t.action == "error"),
    }


# ---------------------------------------------------------------------------
# Trace adapters
# ---------------------------------------------------------------------------

def texts_from_trace(trace, source="final"):
    """The ranked chunk texts of a Trace.

    source "final"     what the LLM received (after rerank / CRAG)
    source "retrieve"  what the retrieve stage returned, before rerank
                       (up to 1500 characters per chunk are stored)
    """
    if source == "final":
        return [chunk["text"] for chunk in trace.final_chunks]

    if source == "retrieve":
        stage = trace.get_stage("retrieve")

        if stage is None or "results" not in stage.details:
            raise ValueError("This trace has no retrieve-stage results.")

        return [item.get("preview", "") for item in stage.details["results"]]

    raise ValueError('source must be "final" or "retrieve".')