"""
Golden dataset: load, validate, inspect, and verify against the PDF.

Place this file at: app/evaluation/dataset.py   (add an empty app/evaluation/__init__.py)

A dataset file (evaluation/dataset.json, version 2) looks like:

    {"version": 2, "source": "rag_survey.pdf", "doc_id": "e1f2...", "pages": 67,
     "questions": [
        {"id": 7, "question": "...", "type": "single_hop", "expected": "answerable",
         "ground_truth": "...",
         "evidence": [{"group": 1, "page": 5, "snippet": "verbatim text from page 5"}],
         "tags": ["project_relevant"], "section_hint": "2 Overview", "reviewed": false},
        {"id": 51, "question": "...", "type": "unanswerable", "expected": "unanswerable",
         "ground_truth": null, "evidence": [], ...}]}

Evidence is stored as TEXT (snippet + page), never as chunk ids, because chunk
ids change whenever the chunking strategy changes. A retrieved chunk "hits" an
evidence snippet when the snippet's text appears in the chunk.

Evidence groups: every group is ONE required fact. Several snippets with the
same group number are alternatives expressing the same fact (any one is
enough). Hit@k asks "is at least one group found?", Recall@k asks "what
fraction of the groups are found?".

Matching ("squash"): lowercase and keep only letters and digits. This makes the
match immune to line breaks, hyphenation, spacing and quote style differences
between PDF text extractors.
"""

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import DATASET_PATH

QUESTION_TYPES = ("single_hop", "multi_hop", "unanswerable")
EXPECTED_VALUES = ("answerable", "unanswerable")


class DatasetError(ValueError):
    """The dataset file is missing required structure (message lists everything)."""


def squash(text):
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Evidence:
    group: int
    page: int
    snippet: str
    _squashed: str = field(default="", repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "_squashed", squash(self.snippet))

    def coverage(self, text):
        """0..1: how much of the snippet appears (as one unbroken piece) in `text`.
        1.0 = the whole snippet is in the text."""
        if not self._squashed:
            return 0.0

        haystack = squash(text)

        if self._squashed in haystack:
            return 1.0

        match = SequenceMatcher(None, self._squashed, haystack, autojunk=False).find_longest_match(
            0, len(self._squashed), 0, len(haystack)
        )

        return match.size / len(self._squashed)

    def matches(self, text, threshold=1.0):
        """Strict by default (the whole snippet is inside the text).

        threshold in (0, 1] relaxes this: True when an unbroken piece of at
        least threshold * len(snippet) characters appears in the text. That is
        the same as 'longest common substring >= threshold', but computed fast:
        some window of that length must be found. Use ~0.8 to tolerate a
        snippet that a chunk boundary cut slightly."""
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1].")

        if not self._squashed:
            return False

        haystack = squash(text)

        if self._squashed in haystack:
            return True

        if threshold >= 1.0:
            return False

        needed = max(1, math.ceil(threshold * len(self._squashed)))
        snippet = self._squashed

        return any(
            snippet[start:start + needed] in haystack
            for start in range(len(snippet) - needed + 1)
        )


# ---------------------------------------------------------------------------
# Question
# ---------------------------------------------------------------------------

@dataclass
class Question:
    id: int
    question: str
    type: str
    expected: str
    ground_truth: Optional[str] = None
    evidence: List[Evidence] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    section_hint: Optional[str] = None
    reviewed: bool = False

    @property
    def answerable(self):
        return self.expected == "answerable"

    @property
    def groups(self):
        """{group number: [Evidence, ...]} (alternatives share a group)."""
        grouped = {}

        for item in self.evidence:
            grouped.setdefault(item.group, []).append(item)

        return grouped

    def groups_found(self, texts, threshold=1.0):
        """Which evidence groups appear in ANY of the given texts?"""
        found = set()

        for number, items in self.groups.items():
            if any(item.matches(text, threshold) for item in items for text in texts):
                found.add(number)

        return found

    def to_dict(self):
        return {
            "id": self.id,
            "question": self.question,
            "type": self.type,
            "expected": self.expected,
            "ground_truth": self.ground_truth,
            "evidence": [
                {"group": e.group, "page": e.page, "snippet": e.snippet}
                for e in self.evidence
            ],
            "tags": list(self.tags),
            "section_hint": self.section_hint,
            "reviewed": self.reviewed,
        }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    questions: List[Question]
    version: int = 2
    source: Optional[str] = None
    doc_id: Optional[str] = None
    pages: Optional[int] = None
    description: str = ""

    def __len__(self):
        return len(self.questions)

    def __iter__(self):
        return iter(self.questions)

    def get(self, question_id):
        for question in self.questions:
            if question.id == question_id:
                return question

        raise KeyError(f"No question with id {question_id}.")

    def filter(self, expected=None, type=None, tag=None, reviewed=None, ids=None):
        """A new Dataset with only the matching questions (e.g. for a quick run)."""
        selected = [
            q for q in self.questions
            if (expected is None or q.expected == expected)
            and (type is None or q.type == type)
            and (tag is None or tag in q.tags)
            and (reviewed is None or q.reviewed == reviewed)
            and (ids is None or q.id in set(ids))
        ]

        return Dataset(selected, self.version, self.source, self.doc_id, self.pages, self.description)

    def stats(self):
        by_type, by_tag = {}, {}

        for q in self.questions:
            by_type[q.type] = by_type.get(q.type, 0) + 1

            for tag in q.tags:
                by_tag[tag] = by_tag.get(tag, 0) + 1

        return {
            "questions": len(self.questions),
            "answerable": sum(q.answerable for q in self.questions),
            "unanswerable": sum(not q.answerable for q in self.questions),
            "reviewed": sum(q.reviewed for q in self.questions),
            "by_type": dict(sorted(by_type.items())),
            "by_tag": dict(sorted(by_tag.items())),
            "evidence_snippets": sum(len(q.evidence) for q in self.questions),
        }

    # ------------------------------------------------------------------

    def validate(self):
        """Return a list of problems (empty list = valid)."""
        problems = []

        if self.version not in (1, 2):
            problems.append(f"version must be 1 or 2, got {self.version!r}.")

        if not self.questions:
            problems.append("the dataset has no questions.")

        seen_ids, seen_texts = set(), {}
        needs_evidence = self.version >= 2

        for q in self.questions:
            where = f"question {q.id!r}"

            if not isinstance(q.id, int) or isinstance(q.id, bool) or q.id < 1:
                problems.append(f"{where}: id must be an integer >= 1.")
            elif q.id in seen_ids:
                problems.append(f"{where}: duplicate id.")

            seen_ids.add(q.id)

            if not isinstance(q.question, str) or not q.question.strip():
                problems.append(f"{where}: question text is empty.")
            else:
                key = squash(q.question)

                if key in seen_texts:
                    problems.append(f"{where}: same question as id {seen_texts[key]}.")

                seen_texts[key] = q.id

            if q.expected not in EXPECTED_VALUES:
                problems.append(f"{where}: expected must be one of {EXPECTED_VALUES}.")

            if q.type not in QUESTION_TYPES:
                problems.append(f"{where}: type must be one of {QUESTION_TYPES}.")

            if (q.expected == "unanswerable") != (q.type == "unanswerable"):
                problems.append(f"{where}: type '{q.type}' does not match expected '{q.expected}'.")

            if not isinstance(q.tags, list) or not all(isinstance(t, str) for t in q.tags):
                problems.append(f"{where}: tags must be a list of strings.")

            if q.expected == "answerable":
                if not (q.ground_truth or "").strip():
                    problems.append(f"{where}: answerable question needs a ground_truth.")

                if needs_evidence and not q.evidence:
                    problems.append(f"{where}: answerable question needs evidence.")

                if q.type == "multi_hop" and len({e.group for e in q.evidence}) < 2:
                    problems.append(f"{where}: multi_hop needs evidence in at least 2 groups.")
            elif q.expected == "unanswerable":
                if q.evidence:
                    problems.append(f"{where}: unanswerable question must have no evidence.")

                if (q.ground_truth or "").strip():
                    problems.append(f"{where}: unanswerable question must have no ground_truth.")

            groups = sorted({e.group for e in q.evidence})

            if groups and groups != list(range(1, len(groups) + 1)):
                problems.append(f"{where}: evidence groups must be numbered 1, 2, 3... (got {groups}).")

            for e in q.evidence:
                if not e.snippet.strip():
                    problems.append(f"{where}: empty evidence snippet.")

                if not isinstance(e.page, int) or e.page < 1 or (self.pages and e.page > self.pages):
                    problems.append(f"{where}: evidence page {e.page!r} is outside 1..{self.pages or '?'}.")

        return problems

    def verify_against_documents(self, documents):
        """Check the dataset against the PDF as the loader sees it.

        documents: output of load_pdf() (one Document per page, with
        metadata["page_number"] and metadata["doc_id"]).
        Returns a list of problems (empty = every snippet is on its page)."""
        problems = []
        page_text = {}

        for doc in documents:
            number = doc.metadata.get("page_number")
            page_text[number] = page_text.get(number, "") + "\n" + doc.page_content

        loaded_ids = {doc.metadata.get("doc_id") for doc in documents} - {None}

        if self.doc_id and loaded_ids and self.doc_id not in loaded_ids:
            problems.append(
                f"dataset is for document {self.doc_id}, but the PDF is {sorted(loaded_ids)}. "
                "Evidence may not match this PDF."
            )

        for q in self.questions:
            for e in q.evidence:
                if e.matches(page_text.get(e.page, "")):
                    continue

                elsewhere = [p for p, text in page_text.items() if e.matches(text)]
                hint = f" (found on pages {elsewhere})" if elsewhere else " (not found anywhere)"
                problems.append(f"question {q.id}: snippet not on page {e.page}{hint}: {e.snippet[:60]}...")

        return problems

    def to_dict(self):
        return {
            "version": self.version,
            "description": self.description,
            "source": self.source,
            "doc_id": self.doc_id,
            "pages": self.pages,
            "match": "squash",
            "questions": [q.to_dict() for q in self.questions],
        }


# ---------------------------------------------------------------------------
# Reading and writing
# ---------------------------------------------------------------------------

def _question_from_dict(raw):
    evidence = [
        Evidence(group=item.get("group", index), page=item.get("page"), snippet=item.get("snippet", ""))
        for index, item in enumerate(raw.get("evidence") or [], start=1)
    ]

    return Question(
        id=raw.get("id"),
        question=raw.get("question"),
        type=raw.get("type"),
        expected=raw.get("expected"),
        ground_truth=raw.get("ground_truth"),
        evidence=evidence,
        tags=raw.get("tags", []),
        section_hint=raw.get("section_hint"),
        reviewed=bool(raw.get("reviewed", False)),
    )


def dataset_from_raw(data):
    if isinstance(data, list):  # the old v1 file: a plain list, no evidence
        questions = []

        for raw in data:
            raw = dict(raw)
            raw["type"] = raw.get("type") or (
                "unanswerable" if raw.get("expected") == "unanswerable" else "single_hop"
            )
            questions.append(_question_from_dict(raw))

        return Dataset(questions, version=1)

    if not isinstance(data, dict):
        raise DatasetError("dataset file must contain an object (v2) or a list (v1).")

    return Dataset(
        questions=[_question_from_dict(raw) for raw in data.get("questions", [])],
        version=data.get("version", 2),
        source=data.get("source"),
        doc_id=data.get("doc_id"),
        pages=data.get("pages"),
        description=data.get("description", ""),
    )


def load_dataset(path=None, validate=True):
    """Read a dataset file. Raises DatasetError listing EVERY problem found."""
    path = Path(path or DATASET_PATH)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DatasetError(f"{path.name} is not valid JSON: {error}") from error

    dataset = dataset_from_raw(data)

    if validate:
        problems = dataset.validate()

        if problems:
            raise DatasetError(
                f"{path.name} has {len(problems)} problem(s):\n  - " + "\n  - ".join(problems)
            )

    return dataset


def save_dataset(dataset, path=None):
    path = Path(path or DATASET_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataset.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    return path


# ---------------------------------------------------------------------------
# Command line:  python -m app.evaluation.dataset [dataset.json] [--pdf file.pdf]
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate the golden dataset.")
    parser.add_argument("path", nargs="?", default=None, help="dataset file (default: config DATASET_PATH)")
    parser.add_argument("--pdf", default=None, help="also verify every snippet against this PDF")
    args = parser.parse_args()

    dataset = load_dataset(args.path)
    stats = dataset.stats()

    print(f"Dataset v{dataset.version}: {stats['questions']} questions "
          f"({stats['answerable']} answerable, {stats['unanswerable']} unanswerable), "
          f"{stats['reviewed']} reviewed")
    print("types:", stats["by_type"])
    print("tags: ", stats["by_tag"])

    if args.pdf:
        from app.ingestion.loader import load_pdf

        problems = dataset.verify_against_documents(load_pdf(args.pdf))
        print(f"\nverification against {args.pdf}: {len(problems)} problem(s)")

        for problem in problems:
            print("  -", problem)

        raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()