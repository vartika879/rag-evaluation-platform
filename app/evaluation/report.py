"""
Reports: markdown for one run, and for a comparison of two runs.

Place this file at: app/evaluation/report.py

EVERY sentence about results is generated from the numbers of the run. (The old
report script printed "5/5 = 100%" from hand-typed constants; this one cannot
say something the data does not say.) The only fixed text is the general advice
and the caveats, which are labelled as such.
"""

from pathlib import Path

from app.config import EVAL_DIR
from app.evaluation.compare import LOWER_IS_BETTER
from app.pipeline_config import PipelineConfig

REPORTS_DIR = EVAL_DIR / "reports"

ADVICE = {
    "retrieval_miss": "the evidence is never retrieved: try hybrid search, another chunk size, HyDE or query expansion",
    "lost_before_llm": "the evidence is retrieved but does not reach the LLM: raise top_k, or add a reranker",
    "generation_error": "the LLM had the evidence and still answered wrongly: check the prompt, the context order and the model",
    "hallucination": "unanswerable questions got an answer: make the refusal rule stricter, or add CRAG / a score gate",
    "false_refusal": "answerable questions were refused: the refusal rule or the retrieved context may be too strict",
    "pipeline_error": "the pipeline crashed: open these traces in the TraceStore and look at the failed stage",
    "provider_rate_limit": (
    "check provider quota and retry limits, then rerun the affected questions"
),
}

DIAGNOSIS_TEXT = {
    "ok": "answered correctly",
    "partial": "answer only partly correct",
    "false_refusal": "refused an answerable question",
    "retrieval_miss": "wrong answer, evidence never retrieved",
    "lost_before_llm": "wrong answer, evidence retrieved but did not reach the LLM",
    "generation_error": "wrong answer although the LLM had the evidence",
    "correct_refusal": "correctly refused an unanswerable question",
    "hallucination": "answered an unanswerable question",
    "pipeline_error": "pipeline crashed",
    "unjudged": "no judge verdict",
    "provider_rate_limit": "provider quota or rate limit prevented evaluation",
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def pct(value, digits=0):
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def num(value, digits=2):
    return "n/a" if value is None else f"{value:.{digits}f}"


def ci_text(ci):
    if not ci or ci.get("mean") is None:
        return "n/a"

    return f"{pct(ci['mean'])} (95% CI {pct(ci['lo'])} to {pct(ci['hi'])}, n={ci['n']})"


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]

    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")

    return "\n".join(lines)


def _ids(ids, limit=12):
    shown = ", ".join(f"Q{i}" for i in ids[:limit])

    return shown + (f" ... (+{len(ids) - limit})" if len(ids) > limit else "") if ids else "-"


def _short(text, limit=70):
    text = " ".join((text or "").split())

    return text if len(text) <= limit else text[: limit - 3] + "..."


def _non_default_settings(config_dict):
    defaults = PipelineConfig().to_dict()

    return [(k, v) for k, v in config_dict.items() if defaults.get(k) != v and k != "retrieval_depth"]


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def render_run_report(run):
    rows = run["rows"]
    summary = run.get("summary")
    ds = run["dataset"]
    out = [f"# Evaluation report: {run['run_id']}  ({run.get('label')})", ""]

    out.append(
        f"Created {run.get('created_at')}, status **{run.get('status')}**. "
        f"Dataset: {ds['n']} questions"
        + (f" for document `{ds['doc_id']}`" if ds.get("doc_id") else "")
        + f", {ds.get('reviewed', 0)} reviewed by a human."
    )

    if summary is None:
        out += ["", f"> This run is not complete: {len(rows)} of {ds['n']} questions are done, so there is no summary yet. "
                    "Continue it with `run_evaluation(..., resume=True)`."]
        return "\n".join(out) + "\n"

    ks = run["settings"]["ks"]
    kmax = max(ks)
    counts = summary["counts"]
    ret_all = summary["retrieval"]["all"]
    diag = summary["end_to_end"]["diagnosis"]
    answerable = [r for r in rows if r["expected"] == "answerable"]

    # ---- configuration ----
    changed = _non_default_settings(run["config"])
    out += ["", "## Configuration", "",
            f"`{run.get('label')}` (fingerprint `{run['config_fingerprint']}`). "
            + ("Settings that differ from the baseline (plain vector search):" if changed else "All settings are the defaults (plain vector search).")]
    if changed:
        out += ["", table(["setting", "value"], [(k, v) for k, v in changed])]
    out += ["", f"Evaluation settings: k = {ks}, evidence match threshold {run['settings']['match_threshold']}, "
                f"retrieval depth {run['settings']['retrieval_depth']}, "
                + (f"LLM judge `{run['settings']['judge_model']}`." if run["settings"]["judge"] else "no LLM judge (retrieval metrics only).")]

    # ---- headline ----
    ci = summary["ci"]
    out += ["", "## Headline", ""]
    out.append(f"- Answerable questions answered correctly (end to end): **{ci_text(ci.get('end_to_end.answerable_ok'))}**"
               if summary["answer"] else
               "- Answer correctness: not measured (run without the LLM judge).")
    out.append(f"- Unanswerable questions correctly refused: **{ci_text(ci.get('end_to_end.unanswerable_ok'))}**")
    out.append(f"- Evidence retrieved within the top {kmax} (Hit@{kmax}): **{ci_text(ci.get(f'retrieval.hit@{kmax}'))}**")
    out.append(f"- Evidence found at rank 1 (Hit@1): {ci_text(ci.get('retrieval.hit@1'))}" if 1 in ks else "")
    out.append(f"- MRR: {num(ret_all.get('mrr'))}")

    # ---- retrieval ----
    out += ["", "## Retrieval (what the retrieve stage returned)", "",
            table(["k", "Hit@k", "Recall@k", "All groups@k", "nDCG@k", "Precision@k"],
                  [(k, pct(ret_all.get(f"hit@{k}")), pct(ret_all.get(f"recall@{k}")), pct(ret_all.get(f"all_groups@{k}")),
                    num(ret_all.get(f"ndcg@{k}")), num(ret_all.get(f"precision@{k}"))) for k in ks]),
            "", f"MRR {num(ret_all.get('mrr'))}. Based on {ret_all.get('n', 0)} answerable questions."
            " (Precision is naturally low here: each question has only one or two relevant chunks.)"]

    outcomes = summary["retrieval"]["outcomes"]
    ids = summary["retrieval"]["outcome_ids"]
    out += ["", "Where the evidence ended up:", "",
            table(["outcome", "questions", "which"],
                  [("found in the top 3", outcomes["ok"], ""), ("found, but below rank 3", outcomes["low_rank"], _ids(ids["low_rank"])),
                   ("only some of the facts found", outcomes["partial"], _ids(ids["partial"])), ("not found at all", outcomes["miss"], _ids(ids["miss"]))])]

    by_type = summary["retrieval"]["by_type"]
    if len(by_type) > 1:
        out += ["", "By question type:", "",
                table(["type", "n", f"Hit@{kmax}", f"Recall@{kmax}", "MRR"],
                      [(t, v["n"], pct(v.get(f"hit@{kmax}")), pct(v.get(f"recall@{kmax}")), num(v.get("mrr"))) for t, v in by_type.items()])]

    # ---- context given to the LLM ----
    in_context = sum(1 for r in answerable if r["context"] and all(x is not None for x in r["context"]["group_ranks"].values()))
    retrieved = sum(1 for r in answerable if r["retrieval"] and all(x is not None for x in r["retrieval"]["group_ranks"].values()))
    out += ["", "## Evidence reaching the LLM", "",
            f"All the evidence was retrieved for {retrieved} of {len(answerable)} answerable questions, "
            f"and was inside the context the LLM received for {in_context} of {len(answerable)}."]

    # ---- answers ----
    answer = summary["answer"]
    if answer:
        out += ["", "## Answer quality (LLM judge)", "",
                table(["measure", "value"],
                      [("faithfulness (mean score)", num(answer["faithfulness_mean"])), ("faithfulness pass rate", pct(answer["faithfulness_pass_rate"])),
                       ("judged CORRECT", pct(answer["correct_rate"])), ("judged PARTIAL", pct(answer["partial_rate"])),
                       ("judged INCORRECT", pct(answer["incorrect_rate"])),
                       ("correct refusal rate (unanswerable)", pct(answer["correct_refusal_rate"])),
                       ("hallucination rate (unanswerable answered)", pct(answer["hallucination_rate"])),
                       ("false refusal rate (answerable refused)", pct(answer["false_refusal_rate"]))])]
        unknown = answer["faithfulness_unknown"] + answer["correctness_unknown"] + answer["refusal_unknown"]
        if unknown:
            out += ["", f"> {unknown} judgments could not be read (the judge failed or answered in a wrong format). They are left out of the means above, "
                        "so these numbers describe only the judgments that worked."]

    # ---- diagnosis ----
    total = counts["questions"]
    out += ["", "## Why questions failed (diagnosis)", "",
            table(["diagnosis", "meaning", "questions", "share", "which"],
                  [(name, DIAGNOSIS_TEXT[name], d["count"], pct(d["count"] / total if total else None), _ids(d["ids"]))
                   for name, d in diag.items() if d["count"]])]

    suggestions = [f"- **{name}** ({diag[name]['count']}): {text}." for name, text in ADVICE.items() if diag[name]["count"]]
    if suggestions:
        out += ["", "General advice for the failures above (not derived from this run's numbers):", ""] + suggestions

    failed = [r for r in rows if r["diagnosis"] not in ("ok", "correct_refusal")]
    if failed:
        out += ["", "### The failed questions", "",
                table(["Q", "diagnosis", "question", "answer given"],
                      [(f"Q{r['question_id']}", r["diagnosis"], _short(r["question"]), _short(r["answer"], 60)) for r in failed[:25]])]
        if len(failed) > 25:
            out.append(f"\n(+{len(failed) - 25} more; see the run file.)")

    # ---- efficiency ----
    eff = summary["efficiency"]
    out += ["", "## Cost and speed", "",
            f"Latency p50 {eff.get('p50_latency_ms')} ms, p95 {eff.get('p95_latency_ms')} ms (mean {eff.get('avg_latency_ms')} ms). "
            f"Average {eff.get('avg_tokens')} tokens and {eff.get('avg_llm_calls')} LLM calls per question"
            + (f"; the judge made {summary['judge_llm_calls']} extra LLM calls." if summary["judge_llm_calls"] else ".")]

    # ---- caveats ----
    notes = []
    if ds.get("reviewed", 0) == 0:
        notes.append("No question in the dataset has been reviewed by a human yet, so the ground truth itself is unchecked.")
    elif ds["reviewed"] < ds["n"]:
        notes.append(f"Only {ds['reviewed']} of {ds['n']} dataset questions are human-reviewed.")
    n_ans = counts["answerable"]
    notes.append(f"Only {n_ans} answerable questions: look at the confidence intervals, a difference of a few points between two runs is noise.")
    if summary["unreachable_ids"]:
        notes.append(f"For {len(summary['unreachable_ids'])} questions ({_ids(summary['unreachable_ids'])}) the evidence does not fit inside any single chunk of this chunking, "
                     "so no retriever can score on them.")
    if counts["pipeline_errors"]:
        notes.append(f"{counts['pipeline_errors']} questions crashed. They count as failures (as retrieval misses if the crash was before retrieval), never as missing data.")
    if counts.get("provider_rate_limits", 0):
        notes.append(
           f"{counts['provider_rate_limits']} questions failed due to provider "
           "rate limits. Their answers were not successfully evaluated."
    )
    if run["settings"]["judge"]:
        notes.append("Judge verdicts are opinions of an LLM. Calibrate the judge against hand labels (`make_label_sheet`, `calibrate`) before trusting small differences.")
    out += ["", "## Caveats", ""] + [f"- {n}" for n in notes]

    return "\n".join(line for line in out if line is not None) + "\n"


# ---------------------------------------------------------------------------
# A comparison of two runs
# ---------------------------------------------------------------------------

def _signed(value, digits=2, as_pct=False):
    if value is None:
        return "n/a"

    text = f"{value * 100:+.0f}%" if as_pct else f"{value:+.{digits}f}"

    return text


def render_comparison(cmp):
    out = [f"# Comparison: {cmp['b']} vs {cmp['a']}", "",
           f"**A (baseline)** `{cmp['a']}` = {cmp['label_a']}   |   **B** `{cmp['b']}` = {cmp['label_b']}", ""]

    if cmp["warnings"]:
        out += ["> **Check before you trust this comparison:**"] + [f"> - {w}" for w in cmp["warnings"]] + [""]

    if cmp["config_diff"]:
        out += ["## What changed in the configuration", "",
                table(["setting", "A", "B"], [(k, a, b) for k, (a, b) in cmp["config_diff"].items()]), ""]
    else:
        out += ["The two runs use the same pipeline settings.", ""]

    rows = []
    for m in cmp["metrics"]:
        lower = m["metric"] in LOWER_IS_BETTER
        fraction = not (m["metric"].startswith("efficiency") or m["metric"].endswith("mrr") or "ndcg" in m["metric"] or m["metric"].startswith("answer."))
        fmt = (lambda v: pct(v)) if fraction else (lambda v: num(v, 0 if lower else 2))
        verdict = ("**B better**" if m["b_is_better"] else "**B worse**") if m["significant"] else "no clear difference"
        interval = "n/a" if m["lo"] is None else f"{_signed(m['lo'], 2, fraction)} to {_signed(m['hi'], 2, fraction)}"
        rows.append((m["metric"], m["n"], fmt(m["a"]), fmt(m["b"]), _signed(m["diff"], 2, fraction), interval, verdict))

    out += ["## Metrics (paired, question by question)", "",
            table(["metric", "n", "A", "B", "B - A", "95% interval of the difference", "verdict"], rows), ""]

    better = [m["metric"] for m in cmp["metrics"] if m["significant"] and m["b_is_better"]]
    worse = [m["metric"] for m in cmp["metrics"] if m["significant"] and m["b_is_better"] is False]
    quiet = [m["metric"] for m in cmp["metrics"] if not m["significant"]]
    out += ["## Summary", "",
            f"- B is clearly better on {len(better)} metrics" + (f": {', '.join(better)}" if better else "") + ".",
            f"- B is clearly worse on {len(worse)} metrics" + (f": {', '.join(worse)}" if worse else "") + ".",
            f"- No clear difference on {len(quiet)} metrics" + (f": {', '.join(quiet)}" if quiet else "") + ".",
            "", "\"Clearly\" means the 95% interval of the per-question difference does not include 0. "
                "With few questions, real but small effects will usually show up as \"no clear difference\".", ""]

    changes = cmp["diagnosis_changes"]
    out += ["## Question by question", "",
            f"{len(changes['improved'])} questions improved, {len(changes['worsened'])} got worse, {changes['unchanged']} unchanged."]
    for title, items in (("Improved", changes["improved"]), ("Got worse", changes["worsened"])):
        if items:
            out += ["", f"**{title}:**", "", table(["Q", "A", "B"], [(f"Q{i['question_id']}", i["from"], i["to"]) for i in items[:20]])]

    return "\n".join(out) + "\n"


def save_report(text, name, reports_dir=REPORTS_DIR):
    path = Path(reports_dir) / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

    return path