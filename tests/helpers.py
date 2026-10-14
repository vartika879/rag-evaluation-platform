"""Test helpers shared by several test files."""
from app.evaluation.runner import diagnose_row
from app.pipeline_config import PipelineConfig


def make_run(run_id, per_q, label="x", status="complete", doc_id="d", ks=(1, 5), judge=True, judge_model="m", threshold=0.8, reviewed=0, **config):
    """per_q: {qid: dict(hit=1.0, correct='CORRECT', faith=1.0, latency=100, tokens=100)}  (answerable)
              {qid: dict(expected='unanswerable', refused=True)}"""
    rows = []
    for qid, s in per_q.items():
        if s.get("expected") == "unanswerable":
            r = {"question_id": qid, "question": f"q{qid}", "type": "unanswerable", "tags": [], "expected": "unanswerable", "action": "answered", "refused": s["refused"],
                 "answer": "x", "error": None, "latency_ms": s.get("latency", 100), "tokens": s.get("tokens", 100), "llm_calls": 1, "retrieval": None,
                 "retrieval_unavailable": False, "context": None, "judge": {"skipped": None, "expected": "unanswerable", "refused": s["refused"], "faithfulness": None, "correctness": None, "llm_calls": 0} if judge else None}
        else:
            hit = s.get("hit", 1.0)
            res = {"question_id": qid, "type": "single_hop", "n_groups": 1, "returned": 5, "group_ranks": {"1": 1 if hit else None}, "first_relevant_rank": 1 if hit else None,
                   "mrr": s.get("mrr", hit)}
            for k in ks:   # every metric a real retrieval row has, for every k
                res.update({f"hit@{k}": hit, f"recall@{k}": hit, f"all_groups@{k}": hit, f"precision@{k}": hit / k, f"ndcg@{k}": s.get("mrr", hit)})
            ctx = {**res, "group_ranks": {"1": 1 if s.get("ctx", hit) else None}}
            ctx.update({f"hit@{k}": s.get("ctx", hit) for k in ks})
            j = None
            if judge and s.get("correct"):
                j = {"skipped": None, "expected": "answerable", "refused": False, "llm_calls": 2,
                     "correctness": {"verdict": s["correct"], "score": {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}[s["correct"]]},
                     "faithfulness": {"verdict": "FAITHFUL", "score": s.get("faith", 1.0), "passed": True}}
            r = {"question_id": qid, "question": f"q{qid}", "type": "single_hop", "tags": [], "expected": "answerable", "action": "answered", "refused": False, "answer": "x",
                 "error": None, "latency_ms": s.get("latency", 100), "tokens": s.get("tokens", 100), "llm_calls": 1, "retrieval": res, "retrieval_unavailable": False, "context": ctx, "judge": j}
        r["diagnosis"] = diagnose_row(r)
        rows.append(r)
    cfg = PipelineConfig().with_changes(**config)
    return {"format_version": 1, "run_id": run_id, "status": status, "label": label, "config": cfg.to_dict(), "config_fingerprint": cfg.fingerprint(),
            "dataset": {"doc_id": doc_id, "n": len(rows), "reviewed": reviewed, "ids": [r["question_id"] for r in rows], "source": "s"},
            "settings": {"ks": list(ks), "match_threshold": threshold, "judge": judge, "judge_model": judge_model if judge else None, "low_rank": 3, "retrieval_depth": max(ks)},
            "rows": rows, "summary": None}