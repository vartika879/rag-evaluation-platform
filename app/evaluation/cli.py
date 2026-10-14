
import argparse
import logging
import sys
from pathlib import Path
 
from app.config import DATASET_PATH, RUNS_DIR
from app.evaluation import report as R
from app.evaluation.compare import compare_runs
from app.evaluation.dataset import Dataset, load_dataset
from app.evaluation.runner import (
    DEFAULT_MATCH_THRESHOLD,
    EvaluationSetupError,
    list_runs,
    load_run,
    run_evaluation,
)
from app.pipeline_config import PRESETS, PipelineConfig, get_preset
 
 
# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
 
def parse_value(text):
    """'true' -> True, '3' -> 3, '0.5' -> 0.5, 'null' -> None, else the text."""
    lowered = text.strip().lower()
 
    if lowered in ("true", "false"):
        return lowered == "true"
 
    if lowered in ("null", "none"):
        return None
 
    for convert in (int, float):
        try:
            return convert(text)
        except ValueError:
            continue
 
    return text
 
 
def parse_overrides(items):
    overrides = {}
 
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got '{item}'.")
 
        key, _, value = item.partition("=")
        overrides[key.strip()] = parse_value(value)
 
    return overrides
 
 
def build_config(preset, overrides):
    """A preset with --set overrides, validated (unknown keys are rejected)."""
    base = get_preset(preset).to_dict()
 
    return PipelineConfig.from_dict({**base, **overrides})
 
 
def limit_dataset(dataset, n):
    """First n questions, keeping about a quarter unanswerable (repeatable)."""
    unanswerable = [q for q in dataset if not q.answerable][: n // 4]
    answerable = [q for q in dataset if q.answerable][: n - len(unanswerable)]
    chosen = sorted(answerable + unanswerable, key=lambda q: q.id)
 
    return Dataset(chosen, dataset.version, dataset.source, dataset.doc_id, dataset.pages, dataset.description)
 
 
def select_dataset(args):
    dataset = load_dataset(args.dataset)
 
    if args.reviewed_only:
        dataset = dataset.filter(reviewed=True)
 
    if args.type:
        dataset = dataset.filter(type=args.type)
 
    if args.tag:
        dataset = dataset.filter(tag=args.tag)
 
    if args.ids:
        dataset = dataset.filter(ids=[int(x) for x in args.ids.split(",")])
 
    if args.limit:
        dataset = limit_dataset(dataset, args.limit)
 
    return dataset
 
 
def progress_printer(index, total, question, row):
    retrieval = row["retrieval"]
    parts = []
 
    if retrieval:
        rank = retrieval["first_relevant_rank"]
        parts.append(f"evidence rank {rank if rank else '-'}")
 
    verdict = ((row["judge"] or {}).get("correctness") or {}).get("verdict")
 
    if verdict:
        parts.append(verdict)
 
    print(f"  [{index:>2}/{total}] Q{question.id:<3} {row['diagnosis']:<17} {row['latency_ms'] / 1000:>5.1f}s  {' | '.join(parts)}")
 
 
def print_headline(run):
    summary = run["summary"]
    ci = summary["ci"]
    kmax = max(run["settings"]["ks"])
 
    print(f"\n  {run['run_id']}  {run['label']}  ({len(run['rows'])} questions)")
    print(f"    retrieval  Hit@{kmax}: {R.ci_text(ci.get(f'retrieval.hit@{kmax}'))}   MRR: {R.num(summary['retrieval']['all'].get('mrr'))}")
 
    if summary["answer"]:
        print(f"    answers    correct: {R.ci_text(ci.get('end_to_end.answerable_ok'))}")
 
    print(f"    refusals   correct: {R.ci_text(ci.get('end_to_end.unanswerable_ok'))}")
 
    failures = {k: v["count"] for k, v in summary["end_to_end"]["diagnosis"].items() if v["count"] and k not in ("ok", "correct_refusal")}
    print(f"    failures   {failures if failures else 'none'}")
 
    eff = summary["efficiency"]
    print(f"    speed      p50 {eff.get('p50_latency_ms')} ms, p95 {eff.get('p95_latency_ms')} ms, {eff.get('avg_tokens')} tokens/question")
 
 
# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
 
def cmd_run(args):
    presets = args.preset or ["vector"]
 
    if args.run_id and len(presets) > 1:
        print("--run-id can only be used with one --preset.")
        return 2
 
    overrides = parse_overrides(args.set)
    dataset = select_dataset(args)
    print(f"Dataset: {len(dataset)} questions ({sum(q.answerable for q in dataset)} answerable)")
 
    finished = []
 
    for preset in presets:
        config = build_config(preset, overrides)
        print(f"\nRunning '{config.label()}' ...")
 
        run = run_evaluation(
            config,
            dataset,
            run_id=args.run_id,
            use_judge=not args.no_judge,
            match_threshold=args.threshold,
            runs_dir=args.runs_dir,
            resume=args.resume,
            question_delay=args.delay,
            label=preset if not overrides else f"{preset}+{'+'.join(f'{k}={v}' for k, v in overrides.items())}",
            progress=progress_printer,
        )
        finished.append(run)
        print_headline(run)
 
        if not args.no_report:
            path = R.save_report(R.render_run_report(run), run["run_id"], args.reports_dir)
            print(f"    report     {path}")
 
    if len(finished) > 1:
        base = finished[0]
 
        for other in finished[1:]:
            cmp = compare_runs(base, other)
            path = R.save_report(R.render_comparison(cmp), f"compare_{base['run_id']}_{other['run_id']}", args.reports_dir)
            better = [m["metric"] for m in cmp["metrics"] if m["significant"] and m["b_is_better"]]
            worse = [m["metric"] for m in cmp["metrics"] if m["significant"] and m["b_is_better"] is False]
            print(f"\n  {other['label']} vs {base['label']}: clearly better on {better or 'nothing'}, clearly worse on {worse or 'nothing'}")
            print(f"    comparison {path}")
 
    return 0
 
 
def cmd_list(args):
    rows = list_runs(args.runs_dir)
 
    if not rows:
        print("No runs yet.")
        return 0
 
    print(R.table(
        ["run", "label", "status", "created", "n", "Hit@5", "MRR", "answers ok", "refusals ok"],
        [(r["run_id"], r["label"], r["status"], (r["created_at"] or "")[:16], r["questions"], R.pct(r["hit@5"]),
          R.num(r["mrr"]), R.pct(r["answerable_ok_rate"]), R.pct(r["unanswerable_ok_rate"])) for r in rows],
    ))
 
    return 0
 
 
def cmd_show(args):
    print(R.render_run_report(load_run(args.run, args.runs_dir)))
 
    return 0
 
 
def cmd_report(args):
    run = load_run(args.run, args.runs_dir)
    path = R.save_report(R.render_run_report(run), run["run_id"], args.reports_dir)
    print(f"Report written: {path}")
 
    return 0
 
 
def cmd_compare(args):
    cmp = compare_runs(args.a, args.b, runs_dir=args.runs_dir)
    text = R.render_comparison(cmp)
    path = R.save_report(text, f"compare_{cmp['a']}_{cmp['b']}", args.reports_dir)
    print(text)
    print(f"(saved to {path})")
 
    return 0
 
 
def cmd_check_dataset(args):
    dataset = load_dataset(args.dataset)
    stats = dataset.stats()
    print(f"Dataset v{dataset.version}: {stats['questions']} questions ({stats['answerable']} answerable, "
          f"{stats['unanswerable']} unanswerable), {stats['reviewed']} reviewed")
    print("types:", stats["by_type"])
 
    if args.pdf:
        from app.ingestion.loader import load_pdf
 
        problems = dataset.verify_against_documents(load_pdf(args.pdf))
        print(f"verification against {args.pdf}: {len(problems)} problem(s)")
 
        for problem in problems:
            print("  -", problem)
 
        return 1 if problems else 0
 
    return 0
 
 
# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
 
def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    commands = parser.add_subparsers(dest="command", required=True)
 
    def common(p):
        p.add_argument("--runs-dir", default=RUNS_DIR, help="where run files live")
        p.add_argument("--reports-dir", default=R.REPORTS_DIR, help="where markdown reports are written")
 
    run = commands.add_parser("run", help="run the evaluation for one or more presets")
    run.add_argument("--preset", action="append", choices=sorted(PRESETS), help="repeat to compare several (default: vector)")
    run.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a pipeline setting, e.g. top_k=3")
    run.add_argument("--dataset", default=DATASET_PATH)
    run.add_argument("--ids", help="only these question ids, e.g. 1,7,22")
    run.add_argument("--type", choices=["single_hop", "multi_hop", "unanswerable"])
    run.add_argument("--tag", help="only questions with this tag, e.g. project_relevant")
    run.add_argument("--limit", type=int, help="only about N questions (a quarter unanswerable)")
    run.add_argument("--reviewed-only", action="store_true", help="only human-reviewed questions")
    run.add_argument("--no-judge", action="store_true", help="skip the LLM judge (retrieval metrics only, no judge cost)")
    run.add_argument("--threshold", type=float, default=DEFAULT_MATCH_THRESHOLD, help="evidence match threshold, 1.0 = strict")
    run.add_argument("--delay", type=float, default=0.0, help="seconds to wait between questions (rate limits)")
    run.add_argument("--run-id")
    run.add_argument("--resume", action="store_true", help="continue the run given by --run-id")
    run.add_argument("--no-report", action="store_true")
    common(run)
    run.set_defaults(func=cmd_run)
 
    lst = commands.add_parser("list", help="list all runs")
    common(lst)
    lst.set_defaults(func=cmd_list)
 
    show = commands.add_parser("show", help="print the report of a run")
    show.add_argument("run")
    common(show)
    show.set_defaults(func=cmd_show)
 
    report = commands.add_parser("report", help="write the markdown report of a run")
    report.add_argument("run")
    common(report)
    report.set_defaults(func=cmd_report)
 
    compare = commands.add_parser("compare", help="compare run B against run A")
    compare.add_argument("a")
    compare.add_argument("b")
    common(compare)
    compare.set_defaults(func=cmd_compare)
 
    check = commands.add_parser("check-dataset", help="validate the dataset (and verify it against a PDF)")
    check.add_argument("--dataset", default=DATASET_PATH)
    check.add_argument("--pdf")
    check.set_defaults(func=cmd_check_dataset)
 
    return parser
 
 
def main(argv=None):
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
 
    try:
        return args.func(args)
    except (EvaluationSetupError, ValueError, FileNotFoundError) as error:
        print(f"ERROR: {error}")
        return 2
 
 
if __name__ == "__main__":
    raise SystemExit(main())