
import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
 
from app.config import TRACE_MAX_CONTEXT_CHARS
 
MAX_CHUNK_TEXT_CHARS = 1500
 
# Stage names used by the pipeline (for reference; any name is accepted):
# rewrite, query_transform, retrieve, rerank, crag, compress, context, generate
 
 
@dataclass
class StageRecord:
    name: str
    status: str = "ok"            # ok | skipped | fallback | error
    latency_ms: int = 0
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
 
 
@dataclass
class Trace:
    trace_id: str
    created_at: str
    question: str
    config: Dict[str, Any]
    config_label: str = ""
    config_fingerprint: str = ""
 
    stages: List[StageRecord] = field(default_factory=list)
 
    final_chunks: List[Dict[str, Any]] = field(default_factory=list)
    context: str = ""
    prompt: str = ""
    answer: str = ""
    refused: bool = False
 
    # answered | refused_no_context | refused_no_evidence | refused_by_llm | error
    action: str = ""
 
    usage: Dict[str, int] = field(
        default_factory=lambda: {
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    )
    total_latency_ms: int = 0
    error: Optional[str] = None
 
    _started: float = field(default=0.0, repr=False, compare=False)
 
    # ------------------------------------------------------------------
    # Creating
    # ------------------------------------------------------------------
 
    @classmethod
    def new(cls, question, config):
        return cls(
            trace_id=uuid.uuid4().hex[:12],
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            question=question,
            config=config.to_dict(),
            config_label=config.label(),
            config_fingerprint=config.fingerprint(),
            _started=time.perf_counter(),
        )
 
    # ------------------------------------------------------------------
    # Recording stages
    # ------------------------------------------------------------------
 
    @contextmanager
    def stage(self, name, **details):
        """Time a stage automatically. Set record.status = "fallback" and
        record.details[...] inside the block as needed. If the block raises,
        the stage is saved with status "error" and the exception continues."""
        record = StageRecord(name=name, details=dict(details))
        started = time.perf_counter()
 
        try:
            yield record
        except Exception as error:
            record.status = "error"
            record.error = str(error)
            raise
        finally:
            record.latency_ms = round((time.perf_counter() - started) * 1000)
            self.stages.append(record)
 
    def skip(self, name, reason, **details):
        self.stages.append(
            StageRecord(name=name, status="skipped", details={"reason": reason, **details})
        )
 
    def get_stage(self, name):
        for record in self.stages:
            if record.name == name:
                return record
 
        return None
 
    # ------------------------------------------------------------------
    # Recording results
    # ------------------------------------------------------------------
 
    def add_usage(self, usage):
        """Add one LLM call's token usage (None values are treated as 0)."""
        self.usage["llm_calls"] += 1
 
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            self.usage[key] += int((usage or {}).get(key) or 0)
 
    def set_chunks(self, chunks):
        records = []
 
        for chunk in chunks:
            data = chunk.to_dict(include_text=True)
            data["text"] = data["text"][:MAX_CHUNK_TEXT_CHARS]
            records.append(data)
 
        self.final_chunks = records
 
    def set_texts(self, context=None, prompt=None):
        """Store the context / prompt, capped so trace files stay small."""
        if context is not None:
            self.context = context[:TRACE_MAX_CONTEXT_CHARS]
 
        if prompt is not None:
            self.prompt = prompt[:TRACE_MAX_CONTEXT_CHARS]
 
    def finish(self):
        if self._started:
            self.total_latency_ms = round((time.perf_counter() - self._started) * 1000)
 
        return self
 
    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------
 
    def to_dict(self):
        """JSON-safe dict. Embedding vectors (any key named 'vector') are
        dropped because 1024 numbers per trace would bloat the files."""
        data = asdict(self)
        data.pop("_started", None)
 
        return _drop_vectors(json.loads(json.dumps(data, default=str)))
 
    def to_json(self, indent=2):
        return json.dumps(self.to_dict(), indent=indent)
 
    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        stages = [StageRecord(**stage) for stage in data.pop("stages", [])]
 
        known = set(cls.__dataclass_fields__) - {"_started"}
        data = {key: value for key, value in data.items() if key in known}
 
        return cls(stages=stages, **data)
 
    def summary(self):
        """One short row for lists / tables."""
        return {
            "trace_id": self.trace_id,
            "created_at": self.created_at,
            "question": self.question,
            "config_label": self.config_label,
            "action": self.action,
            "refused": self.refused,
            "answer_preview": " ".join(self.answer.split())[:120],
            "chunks": len(self.final_chunks),
            "latency_ms": self.total_latency_ms,
            "total_tokens": self.usage.get("total_tokens", 0),
            "llm_calls": self.usage.get("llm_calls", 0),
            "failed_stage": next(
                (s.name for s in self.stages if s.status == "error"), None
            ),
        }
 
 
def _drop_vectors(value):
    if isinstance(value, dict):
        return {k: _drop_vectors(v) for k, v in value.items() if k != "vector"}
 
    if isinstance(value, list):
        return [_drop_vectors(item) for item in value]
 
    return value
 