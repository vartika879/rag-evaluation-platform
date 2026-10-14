"""
PipelineConfig: the "order slip" for one run of the RAG pipeline.

Place this file at: app/pipeline_config.py

Every technique built in Phases 1-3 is a switch here. The default config is
the plain baseline (vector search -> LLM), so any other config can be compared
against it. A config is validated the moment it is created, so a bad setting
fails immediately with a clear message instead of deep inside a run.

Used by: pipeline.py (to decide what to run), the evaluation runner (one
config per experiment), the API / dashboard (settings panel).
"""

import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from typing import Optional

from app.config import (
    CANDIDATE_POOL,
    DEFAULT_TOP_K,
    HYBRID_VECTOR_WEIGHT,
    MAX_TOP_K,
    USE_RERANKER,
)

RETRIEVAL_METHODS = ("vector", "bm25", "hybrid")
QUERY_MODES = ("plain", "expand", "hyde", "multi_query")
FUSION_METHODS = ("rrf", "weighted")
CRAG_STRATEGIES = ("expand", "multi_query")
EMBED_MODES = ("document", "query")
MAX_VARIANTS = 10
MEASUREMENT_ONLY = ('retrieval_depth',)


@dataclass
class PipelineConfig:
    # ---- where to search --------------------------------------------------
    collection_name: Optional[str] = None      # None = default collection / alias
    doc_id: Optional[str] = None
    page_min: Optional[int] = None             # 1-indexed, as users see pages
    page_max: Optional[int] = None

    # ---- retrieval --------------------------------------------------------
    retrieval_method: str = "vector"           # vector | bm25 | hybrid
    top_k: int = DEFAULT_TOP_K                 # chunks finally given to the LLM
    candidate_pool: int = CANDIDATE_POOL       # per-retriever candidates for hybrid
    fusion_method: str = "rrf"                 # rrf | weighted (hybrid only)
    vector_weight: float = HYBRID_VECTOR_WEIGHT

    # ---- query side (use at most one of expand / hyde / multi_query) ------
    query_mode: str = "plain"                  # plain | expand | hyde | multi_query
    rewrite_with_history: bool = False         # standalone-question rewrite (needs chat history)
    num_query_variants: int = 3                # multi_query: extra wordings to ask for
    hyde_embed_as: str = "document"            # document | query
    hyde_query_weight: float = 0.0             # 0 = pure HyDE, 1 = ignore HyDE

    # ---- rerank -----------------------------------------------------------
    use_reranker: bool = USE_RERANKER
    rerank_candidates: int = 20                # how many chunks the reranker reads
    rerank_min_score: Optional[float] = None

    # ---- corrective / compression ----------------------------------------
    use_crag: bool = False
    crag_strategy: str = "expand"              # expand | multi_query
    use_compression: bool = False

    # ---- generation -------------------------------------------------------
    cite_sources: bool = False

    # ---- measurement only (does NOT change the answer) --------------------
    # Retrieve this many chunks (so Hit@10 can be measured) while the LLM still
    # only gets top_k. 0 = off. Ignored by fingerprint() and diff_from().
    retrieval_depth: int = 0

    def __post_init__(self):
        self.validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self):
        def check_choice(name, value, choices):
            if value not in choices:
                raise ValueError(f"{name} must be one of {choices}, got '{value}'.")

        check_choice("retrieval_method", self.retrieval_method, RETRIEVAL_METHODS)
        check_choice("query_mode", self.query_mode, QUERY_MODES)
        check_choice("fusion_method", self.fusion_method, FUSION_METHODS)
        check_choice("crag_strategy", self.crag_strategy, CRAG_STRATEGIES)
        check_choice("hyde_embed_as", self.hyde_embed_as, EMBED_MODES)

        def check_int(name, value, low, high):
            if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
                raise ValueError(f"{name} must be an integer between {low} and {high}.")

        check_int("top_k", self.top_k, 1, MAX_TOP_K)
        check_int("candidate_pool", self.candidate_pool, 1, MAX_TOP_K)
        check_int("rerank_candidates", self.rerank_candidates, 1, MAX_TOP_K)
        check_int("num_query_variants", self.num_query_variants, 1, MAX_VARIANTS)

        if self.retrieval_depth != 0:
            check_int("retrieval_depth", self.retrieval_depth, self.top_k, MAX_TOP_K)

        if self.retrieval_method == "hybrid" and self.candidate_pool < self.top_k:
            raise ValueError("candidate_pool must be >= top_k.")

        if self.use_reranker and self.rerank_candidates < self.top_k:
            raise ValueError("rerank_candidates must be >= top_k.")

        for name in ("vector_weight", "hyde_query_weight"):
            value = getattr(self, name)

            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a number between 0 and 1.")

        if self.query_mode == "hyde" and self.retrieval_method != "vector":
            raise ValueError(
                "query_mode 'hyde' builds a search vector, so it needs "
                "retrieval_method 'vector'."
            )

        for name in ("page_min", "page_max"):
            value = getattr(self, name)

            if value is not None and (not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be an integer >= 1 (pages start at 1).")

        if (
            self.page_min is not None
            and self.page_max is not None
            and self.page_min > self.page_max
        ):
            raise ValueError("page_min cannot be greater than page_max.")

        if self.rerank_min_score is not None and not 0 <= self.rerank_min_score <= 1:
            raise ValueError("rerank_min_score must be between 0 and 1.")

        return self

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        known = {field.name for field in fields(cls)}
        unknown = set(data or {}) - known

        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")

        return cls(**(data or {}))

    def with_changes(self, **changes):
        """A validated copy with some settings changed."""
        return replace(self, **changes)

    # ------------------------------------------------------------------
    # Identity (for traces, run comparison and the dashboard)
    # ------------------------------------------------------------------

    def label(self):
        """Short human name, e.g. 'hybrid-rrf+rerank+crag'."""
        method = (
            f"hybrid-{self.fusion_method}"
            if self.retrieval_method == "hybrid"
            else self.retrieval_method
        )
        parts = [method]

        if self.query_mode != "plain":
            parts.append(self.query_mode)

        if self.rewrite_with_history:
            parts.append("rewrite")

        if self.use_reranker:
            parts.append("rerank")

        if self.use_crag:
            parts.append("crag")

        if self.use_compression:
            parts.append("compress")

        return "+".join(parts)

    def fingerprint(self):
        """8-character id of the settings that change the ANSWER. Same settings
        = same fingerprint. Measurement-only settings are left out."""
        data = {k: v for k, v in self.to_dict().items() if k not in MEASUREMENT_ONLY}
        text = json.dumps(data, sort_keys=True)

        return hashlib.sha1(text.encode()).hexdigest()[:8]

    def diff_from(self, other):
        """{setting: (other_value, this_value)} for every setting that differs.
        Shows exactly what changed between two runs."""
        mine, theirs = self.to_dict(), other.to_dict()

        return {
            key: (theirs[key], mine[key])
            for key in mine
            if mine[key] != theirs[key] and key not in MEASUREMENT_ONLY
        }


# ---------------------------------------------------------------------------
# Ready-made experiments
# ---------------------------------------------------------------------------

PRESETS = {
    "vector": PipelineConfig(),
    "bm25": PipelineConfig(retrieval_method="bm25"),
    "hybrid": PipelineConfig(retrieval_method="hybrid"),
    "hybrid_weighted": PipelineConfig(retrieval_method="hybrid", fusion_method="weighted"),
    "vector_rerank": PipelineConfig(use_reranker=True),
    "hybrid_rerank": PipelineConfig(retrieval_method="hybrid", use_reranker=True),
    "hyde": PipelineConfig(query_mode="hyde"),
    "expand": PipelineConfig(retrieval_method="hybrid", query_mode="expand"),
    "multi_query": PipelineConfig(retrieval_method="hybrid", query_mode="multi_query"),
    "hybrid_rerank_crag": PipelineConfig(
        retrieval_method="hybrid", use_reranker=True, use_crag=True
    ),
}


def get_preset(name, **changes):
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Choose from: {sorted(PRESETS)}")

    return PRESETS[name].with_changes(**changes)