
import csv
import hashlib
import json
import logging
import random
import re
import time
from pathlib import Path
 
from app.config import (
    CORRECTNESS_PASS,
    FAITHFULNESS_PASS,
    JUDGE_DELAY_SECONDS,
    JUDGE_MODEL,
)
from app.evaluation.metrics import refusal_metrics
from app.generation.llm import complete_text, create_llm, is_refusal
 
logger = logging.getLogger(__name__)

JUDGE_MAX_CONTEXT_CHARS = 8000
MAX_ANSWER_CHARS = 2000
MAX_REFERENCE_CHARS = 1000
 
FAITHFULNESS_LABELS = ("FAITHFUL", "PARTIAL", "UNFAITHFUL")
CORRECTNESS_LABELS = ("CORRECT", "PARTIAL", "INCORRECT")
SUFFICIENCY_LABELS = ("SUFFICIENT", "PARTIAL", "INSUFFICIENT")
CORRECTNESS_SCORES = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}
FAITHFULNESS_SCORES = {"FAITHFUL": 1.0, "PARTIAL": 0.5, "UNFAITHFUL": 0.0}
SUFFICIENCY_SCORES = {"SUFFICIENT": 1.0, "PARTIAL": 0.5, "INSUFFICIENT": 0.0}
 
# wording that suggests "I can't answer"; only then is the refusal LLM asked
_HEDGES = (
    "don't have", "do not have", "not enough information", "insufficient",
    "cannot", "can't", "unable", "not mention", "no information", "not provided",
    "not specified", "not stated", "not say", "doesn't say", "does not say",
    "not contain", "doesn't contain", "no mention", "not available", "don't know",
    "do not know", "not clear from", "not in the",
)
 
# short codes used in the human label sheet
_CODES = {
    "faithfulness": {"F": "FAITHFUL", "P": "PARTIAL", "U": "UNFAITHFUL"},
    "correctness": {"C": "CORRECT", "P": "PARTIAL", "I": "INCORRECT"},
}
 
 
# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
 
def _clip(text, limit):
    text = text or ""
 
    return text if len(text) <= limit else text[:limit] + " ..."
 
 
def parse_label(text, labels, key="VERDICT"):
    """'VERDICT: **partial**' -> 'PARTIAL'. Falls back to a first line that is
    just the label. Anything else -> None (never guessed from running text)."""
    text = text or ""
    options = "|".join(labels)
 
    labelled = re.search(rf"\b{key}\s*[:\-]?\W*({options})\b", text, re.IGNORECASE)
 
    if labelled:
        return labelled.group(1).upper()
 
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    only = re.match(rf"^\W*({options})\W*$", first_line, re.IGNORECASE)
 
    return only.group(1).upper() if only else None
 
 
def _parse_reason(text):
    match = re.search(r"\breason\s*[:\-]\s*(.+)", text or "", re.IGNORECASE)
 
    return match.group(1).strip() if match else ""
 
 
def _parse_int(text, key):
    match = re.search(rf"\b{key}\s*[:\-]?\s*(\d+)", text or "", re.IGNORECASE)
 
    return int(match.group(1)) if match else None
 
 
def looks_like_hedge(answer):
    lowered = (answer or "").lower().replace("\u2019", "'")
 
    return any(phrase in lowered for phrase in _HEDGES)
 
 
# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
 
def build_faithfulness_prompt(question, answer, context):
    return f"""
You are checking whether an ANSWER is supported by the CONTEXT.
 
Rules:
- Use ONLY the context. Ignore your own knowledge, even when the answer is true in the real world.
- Split the answer into its factual claims. A claim is supported only if the context states it or clearly implies it.
- Opinions and connecting words are not claims.
 
Question:
{question}
 
Context:
{_clip(context, JUDGE_MAX_CONTEXT_CHARS)}
 
Answer:
{_clip(answer, MAX_ANSWER_CHARS)}
 
Reply in exactly this format:
CLAIMS: <number of factual claims in the answer>
SUPPORTED: <how many of them the context supports>
VERDICT: <FAITHFUL or PARTIAL or UNFAITHFUL>
REASON: <one short sentence, name the unsupported claim if any>
"""
 
 
def build_correctness_prompt(question, answer, reference):
    return f"""
You are grading an ANSWER against a REFERENCE answer for a question.
 
Rules:
- Judge the meaning, not the wording. Extra correct detail is fine.
- CORRECT: the answer contains the key facts of the reference and nothing that contradicts it.
- PARTIAL: it contains some key facts but misses an important one.
- INCORRECT: it misses the key facts, contradicts the reference, or declines to answer.
 
Question:
{question}
 
Reference answer:
{_clip(reference, MAX_REFERENCE_CHARS)}
 
Answer to grade:
{_clip(answer, MAX_ANSWER_CHARS)}
 
Reply in exactly this format:
VERDICT: <CORRECT or PARTIAL or INCORRECT>
REASON: <one short sentence>
"""
 
 
def build_refusal_prompt(answer):
    return f"""
Does the ANSWER decline to answer because the information is not available (for example "I don't know", "the documents do not say", "there is not enough information")?
An answer that gives a concrete answer, even a wrong one, is NOT a refusal.
 
Answer:
{_clip(answer, MAX_ANSWER_CHARS)}
 
Reply in exactly this format:
REFUSED: <YES or NO>
"""
 
 
def build_sufficiency_prompt(question, context, reference):
    return f"""
You are checking retrieved text for a question-answering system.
 
Question:
{question}
 
Reference answer (what a correct answer says):
{_clip(reference, MAX_REFERENCE_CHARS)}
 
Retrieved context:
{_clip(context, JUDGE_MAX_CONTEXT_CHARS)}
 
Does the context contain the information needed to give the reference answer?
SUFFICIENT: all of it is there. PARTIAL: some of it. INSUFFICIENT: none of it.
 
Reply in exactly this format:
VERDICT: <SUFFICIENT or PARTIAL or INSUFFICIENT>
REASON: <one short sentence>
"""
 
 
# ---------------------------------------------------------------------------
# Cache (same prompt + same judge model = same judgment, no new LLM call)
# ---------------------------------------------------------------------------
 
class JudgeCache:
    """In-memory, optionally saved to a JSONL file (one judgment per line)."""
 
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self._data = {}
 
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                    self._data[item["key"]] = item["value"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue  # a damaged line is simply ignored
 
    def __len__(self):
        return len(self._data)
 
    def get(self, key):
        return self._data.get(key)
 
    def set(self, key, value):
        self._data[key] = value
 
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
 
            with open(self.path, "a", encoding="utf-8") as file:
                file.write(json.dumps({"key": key, "value": value}) + "\n")
 
 
# ---------------------------------------------------------------------------
# The judge
# ---------------------------------------------------------------------------
 
def _result(kind, verdict, score, reason="", **extra):
    base = {
        "kind": kind,
        "verdict": verdict,
        "score": score,
        "reason": reason,
        "llm_called": False,
        "cached": False,
        "fallback_reason": None,
        "latency_ms": 0,
        "usage": None,
    }
    base.update(extra)
 
    return base
 
 
class Judge:
    def __init__(self, llm=None, cache=None, delay=None):
        self._llm = llm
        self.cache = cache if cache is not None else JudgeCache()
        self.delay = JUDGE_DELAY_SECONDS if delay is None else delay
 
    @property
    def llm(self):
        if self._llm is None:
            self._llm = create_llm(JUDGE_MODEL)
 
        return self._llm
 
    def _model_name(self):
        return getattr(self.llm, "model_name", None) or JUDGE_MODEL
 
    def _ask(self, kind, prompt):
        """One cached LLM call -> (raw text or None, bookkeeping dict)."""
        key = hashlib.sha1(f"{kind}|{self._model_name()}|{prompt}".encode()).hexdigest()
        hit = self.cache.get(key)
 
        if hit is not None:
            return hit["text"], {"llm_called": False, "cached": True, "latency_ms": 0, "usage": None}
 
        try:
            response = complete_text(prompt, self.llm)
        except Exception as error:
            logger.warning("Judge call failed (%s): %s", kind, error)
            return None, {"llm_called": True, "cached": False, "latency_ms": 0, "usage": None,
                          "fallback_reason": f"llm_error: {error}"}
 
        self.cache.set(key, {"text": response["text"]})
 
        if self.delay:
            time.sleep(self.delay)
 
        return response["text"], {"llm_called": True, "cached": False,
                                  "latency_ms": response["latency_ms"], "usage": response["usage"]}
 
    # ---- faithfulness ----------------------------------------------------
 
    def faithfulness(self, question, answer, context):
        """Is the answer supported by the context? Result has verdict, score
        (0..1), claims/supported, passed (score >= FAITHFULNESS_PASS)."""
        if not (answer or "").strip():
            return _result("faithfulness", "UNKNOWN", None, "empty answer", passed=None)
 
        if is_refusal(answer):  # a refusal makes no claims
            return _result("faithfulness", "FAITHFUL", 1.0, "refusal makes no claims", passed=True)
 
        if not (context or "").strip():  # claims with no context cannot be supported
            return _result("faithfulness", "UNFAITHFUL", 0.0, "no context was given", passed=False)
 
        text, info = self._ask("faithfulness", build_faithfulness_prompt(question, answer, context))
        base = _result("faithfulness", "UNKNOWN", None, passed=None, **info)
 
        if text is None:
            return base
 
        verdict = parse_label(text, FAITHFULNESS_LABELS)
        claims, supported = _parse_int(text, "CLAIMS"), _parse_int(text, "SUPPORTED")
        counts_ok = claims is not None and supported is not None and claims > 0 and supported <= claims
 
        if counts_ok:
            score = supported / claims
        elif verdict:
            score = FAITHFULNESS_SCORES[verdict]
        else:
            base["fallback_reason"] = "unparseable_output"
            base["reason"] = text[:120]
            return base
 
        if verdict is None:
            verdict = "FAITHFUL" if score >= 0.999 else ("UNFAITHFUL" if score <= 0.001 else "PARTIAL")
 
        base.update(verdict=verdict, score=score, reason=_parse_reason(text),
                    claims=claims, supported=supported, passed=score >= FAITHFULNESS_PASS)
 
        return base
 
    # ---- correctness -----------------------------------------------------
 
    def correctness(self, question, answer, reference):
        """Does the answer match the reference? score 1 / 0.5 / 0."""
        if not (reference or "").strip():
            raise ValueError("correctness needs a reference answer (ground_truth).")
 
        if not (answer or "").strip():
            return _result("correctness", "INCORRECT", 0.0, "empty answer", passed=False)
 
        if is_refusal(answer):  # declining an answerable question is a wrong answer
            return _result("correctness", "INCORRECT", 0.0, "refused an answerable question", passed=False)
 
        text, info = self._ask("correctness", build_correctness_prompt(question, answer, reference))
        base = _result("correctness", "UNKNOWN", None, passed=None, **info)
 
        if text is None:
            return base
 
        verdict = parse_label(text, CORRECTNESS_LABELS)
 
        if verdict is None:
            base["fallback_reason"] = "unparseable_output"
            base["reason"] = text[:120]
            return base
 
        score = CORRECTNESS_SCORES[verdict]
        base.update(verdict=verdict, score=score, reason=_parse_reason(text), passed=score >= CORRECTNESS_PASS)
 
        return base
 
    # ---- refusal ---------------------------------------------------------
 
    def refusal(self, answer):
        """Did the answer decline? verdict YES / NO (UNKNOWN if the judge fails)."""
        if not (answer or "").strip():
            return _result("refusal", "NO", 0.0, "empty answer", refused=False)
 
        if is_refusal(answer):
            return _result("refusal", "YES", 1.0, "exact refusal sentence", refused=True)
 
        if not looks_like_hedge(answer):  # concrete answer, no need to ask an LLM
            return _result("refusal", "NO", 0.0, "no refusal wording", refused=False)
 
        text, info = self._ask("refusal", build_refusal_prompt(answer))
        base = _result("refusal", "UNKNOWN", None, refused=None, **info)
 
        if text is None:
            return base
 
        verdict = parse_label(text, ("YES", "NO"), key="REFUSED")
 
        if verdict is None:
            base["fallback_reason"] = "unparseable_output"
            return base
 
        base.update(verdict=verdict, score=1.0 if verdict == "YES" else 0.0, refused=verdict == "YES")
 
        return base
 
    # ---- sufficiency (optional) ------------------------------------------
 
    def sufficiency(self, question, context, reference):
        """Does the context contain what the reference answer needs?"""
        if not (context or "").strip():
            return _result("sufficiency", "INSUFFICIENT", 0.0, "no context")
 
        text, info = self._ask("sufficiency", build_sufficiency_prompt(question, context, reference))
        base = _result("sufficiency", "UNKNOWN", None, **info)
 
        if text is None:
            return base
 
        verdict = parse_label(text, SUFFICIENCY_LABELS)
 
        if verdict is None:
            base["fallback_reason"] = "unparseable_output"
            return base
 
        base.update(verdict=verdict, score=SUFFICIENCY_SCORES[verdict], reason=_parse_reason(text))
 
        return base
 
    # ---- one whole trace -------------------------------------------------
 
    def judge_trace(self, trace, question, include_sufficiency=False):
        """Judge one Trace against its dataset Question. Returns a dict with
        expected, refused (semantic), refusal, faithfulness, correctness,
        sufficiency (each a result dict or None) and llm_calls."""
        out = {
            "trace_id": trace.trace_id,
            "question_id": question.id,
            "expected": question.expected,
            "skipped": None,
            "refused": None,
            "refusal": None,
            "faithfulness": None,
            "correctness": None,
            "sufficiency": None,
        }
 
        if trace.action == "error":
            out["skipped"] = "pipeline_error"
            out["llm_calls"] = 0
            return out
 
        refusal = self.refusal(trace.answer)
        out["refusal"] = refusal
        out["refused"] = refusal["refused"]
        refused = refusal["refused"] is True
 
        if question.answerable:
            out["correctness"] = self.correctness(question.question, trace.answer, question.ground_truth)
 
            if include_sufficiency:
                out["sufficiency"] = self.sufficiency(question.question, trace.context, question.ground_truth)
 
        if not refused:  # an answer was given: is it grounded?
            out["faithfulness"] = self.faithfulness(question.question, trace.answer, trace.context)
 
        parts = [out[name] for name in ("refusal", "faithfulness", "correctness", "sufficiency") if out[name]]
        out["llm_calls"] = sum(1 for part in parts if part["llm_called"])
 
        return out
 
 
# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
 
def _mean_and_unknown(results):
    scores = [r["score"] for r in results if r and r.get("score") is not None]
    unknown = sum(1 for r in results if r and r.get("score") is None)
 
    return (sum(scores) / len(scores) if scores else None), unknown
 
 
def aggregate_judgments(judged):
    """Summary numbers over a list of judge_trace() dicts.
 
    UNKNOWN judgments (judge failed) are excluded from means and COUNTED in
    *_unknown, so a broken judge cannot silently inflate a score."""
    rows = [j for j in judged if not j.get("skipped")]
    answered_answerable = [j for j in rows if j["expected"] == "answerable"]
 
    faith = [j["faithfulness"] for j in rows if j["faithfulness"]]
    correct = [j["correctness"] for j in answered_answerable if j["correctness"]]
 
    faith_mean, faith_unknown = _mean_and_unknown(faith)
    correct_mean, correct_unknown = _mean_and_unknown(correct)
 
    known_correct = [r for r in correct if r["score"] is not None]
    known_faith = [r for r in faith if r["score"] is not None]
 
    refusal = refusal_metrics(
        [{"expected": j["expected"], "refused": bool(j["refused"])} for j in rows if j["refused"] is not None]
    )
 
    return {
        "n": len(rows),
        "skipped": len(judged) - len(rows),
        "faithfulness_mean": faith_mean,
        "faithfulness_pass_rate": (
            sum(1 for r in known_faith if r["passed"]) / len(known_faith) if known_faith else None
        ),
        "faithfulness_unknown": faith_unknown,
        "correctness_mean": correct_mean,
        "correct_rate": (
            sum(1 for r in known_correct if r["verdict"] == "CORRECT") / len(known_correct) if known_correct else None
        ),
        "partial_rate": (
            sum(1 for r in known_correct if r["verdict"] == "PARTIAL") / len(known_correct) if known_correct else None
        ),
        "incorrect_rate": (
            sum(1 for r in known_correct if r["verdict"] == "INCORRECT") / len(known_correct) if known_correct else None
        ),
        "correctness_unknown": correct_unknown,
        "refusal_unknown": sum(1 for j in rows if j["refused"] is None),
        **{key: value for key, value in refusal.items()},
        "judge_llm_calls": sum(j.get("llm_calls", 0) for j in rows),
    }
 
 
# ---------------------------------------------------------------------------
# Measuring the judge itself
# ---------------------------------------------------------------------------
 
def agreement(predicted, human):
    """How well do two label lists agree? accuracy, Cohen's kappa (agreement
    beyond chance: 0 = chance, 1 = perfect, < 0.4 = weak) and a confusion table
    {human label: {judge label: count}}."""
    pairs = [(p, h) for p, h in zip(predicted, human) if p is not None and h is not None]
    n = len(pairs)
 
    if not n:
        return {"n": 0, "accuracy": None, "kappa": None, "confusion": {}}
 
    labels = sorted({x for pair in pairs for x in pair})
    accuracy = sum(1 for p, h in pairs if p == h) / n
    expected = sum(
        (sum(1 for p, _ in pairs if p == label) / n) * (sum(1 for _, h in pairs if h == label) / n)
        for label in labels
    )
    kappa = 1.0 if expected >= 1 else (accuracy - expected) / (1 - expected)
 
    confusion = {}
 
    for p, h in pairs:
        confusion.setdefault(h, {}).setdefault(p, 0)
        confusion[h][p] += 1
 
    return {"n": n, "accuracy": accuracy, "kappa": kappa, "confusion": confusion}
 
 
_SHEET_COLUMNS = [
    "trace_id", "question_id", "question", "ground_truth", "answer", "context_preview",
    "human_faithfulness", "human_correctness", "human_refused", "note",
]
 
 
def make_label_sheet(traces_with_questions, path, n=30, seed=0):
    """Write a CSV for you to label by hand (open it in Excel / Google Sheets).
 
    traces_with_questions: list of (Trace, Question). A mix of answerable and
    unanswerable questions is sampled (seeded, so the sample is repeatable).
    Fill in the three human_* columns:
        human_faithfulness  F = faithful, P = partial, U = unfaithful   (judge the ANSWER against the CONTEXT)
        human_correctness   C = correct,  P = partial, I = incorrect    (answerable questions only)
        human_refused       Y or N                                      (did the answer decline?)
    """
    pairs = [(t, q) for t, q in traces_with_questions if t.action != "error"]
    rng = random.Random(seed)
    answerable = [p for p in pairs if p[1].answerable]
    unanswerable = [p for p in pairs if not p[1].answerable]
    rng.shuffle(answerable)
    rng.shuffle(unanswerable)
 
    want_un = min(len(unanswerable), max(1, round(n * 0.25)))
    chosen = unanswerable[:want_un] + answerable[: max(0, n - want_un)]
    rng.shuffle(chosen)
 
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
 
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=_SHEET_COLUMNS)
        writer.writeheader()
 
        for trace, question in chosen:
            writer.writerow({
                "trace_id": trace.trace_id,
                "question_id": question.id,
                "question": question.question,
                "ground_truth": question.ground_truth or "",
                "answer": trace.answer,
                "context_preview": " ".join(trace.context.split())[:400],
                "human_faithfulness": "", "human_correctness": "", "human_refused": "", "note": "",
            })
 
    return len(chosen)
 
 
def _decode(value, kind):
    value = (value or "").strip()
 
    if not value:
        return None
 
    options = _CODES[kind]
    upper = value.upper()
 
    if upper in options:
        return options[upper]
 
    if upper in options.values():
        return upper
 
    raise ValueError(f"Unknown {kind} label '{value}'. Use one of {sorted(options)}.")
 
 
def load_labels(path):
    """Read the filled-in sheet. Rows you have not labeled at all are skipped."""
    labels = []
 
    with open(path, newline="", encoding="utf-8-sig") as file:
        for line_number, row in enumerate(csv.DictReader(file), start=2):
            try:
                item = {
                    "trace_id": row["trace_id"],
                    "question_id": int(row["question_id"]),
                    "faithfulness": _decode(row.get("human_faithfulness"), "faithfulness"),
                    "correctness": _decode(row.get("human_correctness"), "correctness"),
                    "refused": {"Y": True, "N": False}.get((row.get("human_refused") or "").strip().upper()),
                }
            except ValueError as error:
                raise ValueError(f"{Path(path).name}, line {line_number}: {error}") from error
 
            if any(item[key] is not None for key in ("faithfulness", "correctness", "refused")):
                labels.append(item)
 
    return labels
 
 
def calibrate(judge, labels, get_trace, dataset):
    """Compare the judge with your hand labels.
 
    get_trace(trace_id) -> Trace (e.g. TraceStore.get). Returns, per dimension,
    agreement() plus the trace ids where judge and human disagree (read those
    first: they show what the judge misunderstands)."""
    predicted = {"faithfulness": [], "correctness": [], "refused": []}
    humans = {"faithfulness": [], "correctness": [], "refused": []}
    disagreements = {"faithfulness": [], "correctness": [], "refused": []}
 
    for label in labels:
        trace = get_trace(label["trace_id"])
 
        if trace is None:
            continue
 
        judged = judge.judge_trace(trace, dataset.get(label["question_id"]))
 
        guess = {
            "faithfulness": (judged["faithfulness"] or {}).get("verdict"),
            "correctness": (judged["correctness"] or {}).get("verdict"),
            "refused": judged["refused"],
        }
 
        # a refused answer is faithful by definition (judge_trace skips that call)
        if judged["refused"] is True and guess["faithfulness"] is None:
            guess["faithfulness"] = "FAITHFUL"
 
        for dimension in predicted:
            if label[dimension] is None:
                continue
 
            value = None if guess[dimension] == "UNKNOWN" else guess[dimension]
            predicted[dimension].append(value)
            humans[dimension].append(label[dimension])
 
            if value != label[dimension]:
                disagreements[dimension].append(label["trace_id"])
 
    report = {}
 
    for dimension in predicted:
        report[dimension] = agreement(predicted[dimension], humans[dimension])
        report[dimension]["disagreements"] = disagreements[dimension]
 
    return report