
import json
import logging
import sqlite3
from contextlib import closing
from functools import lru_cache
from pathlib import Path
 
from app.config import TRACE_ENABLED, TRACES_DIR
from app.tracing.trace import Trace
 
logger = logging.getLogger(__name__)
 
DB_FILENAME = "traces.db"
 
SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id           TEXT PRIMARY KEY,
    created_at         TEXT NOT NULL,
    run_id             TEXT,
    question           TEXT NOT NULL,
    config_label       TEXT,
    config_fingerprint TEXT,
    action             TEXT,
    refused            INTEGER NOT NULL DEFAULT 0,
    failed_stage       TEXT,
    latency_ms         INTEGER,
    total_tokens       INTEGER,
    summary_json       TEXT NOT NULL,
    payload_json       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_traces_created ON traces(created_at);
CREATE INDEX IF NOT EXISTS idx_traces_run ON traces(run_id);
CREATE INDEX IF NOT EXISTS idx_traces_fingerprint ON traces(config_fingerprint);
CREATE INDEX IF NOT EXISTS idx_traces_action ON traces(action);
"""
 
_ORDER = "ORDER BY created_at DESC, rowid DESC"
 
 
def _percentile(sorted_values, fraction):
    if not sorted_values:
        return 0
 
    index = min(len(sorted_values) - 1, round(fraction * (len(sorted_values) - 1)))
 
    return sorted_values[index]
 
 
class TraceStore:
    def __init__(self, path=None):
        directory = Path(path).parent if path else Path(TRACES_DIR)
        self.path = Path(path) if path else directory / DB_FILENAME
 
        self.path.parent.mkdir(parents=True, exist_ok=True)
 
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)
            conn.commit()
 
    def _connect(self):
        conn = sqlite3.connect(str(self.path), timeout=30)
        conn.row_factory = sqlite3.Row
 
        return conn
 
    # ------------------------------------------------------------------
    # Filters (shared by list / count / stats). Values are always passed as
    # SQL parameters, never pasted into the query text.
    # ------------------------------------------------------------------
 
    @staticmethod
    def _where(
        run_id=None,
        config_label=None,
        config_fingerprint=None,
        action=None,
        refused=None,
        failed_only=False,
        question_contains=None,
    ):
        clauses, params = [], []
 
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
 
        if config_label is not None:
            clauses.append("config_label = ?")
            params.append(config_label)
 
        if config_fingerprint is not None:
            clauses.append("config_fingerprint = ?")
            params.append(config_fingerprint)
 
        if action is not None:
            clauses.append("action = ?")
            params.append(action)
 
        if refused is not None:
            clauses.append("refused = ?")
            params.append(1 if refused else 0)
 
        if failed_only:
            clauses.append("failed_stage IS NOT NULL")
 
        if question_contains:
            escaped = (
                question_contains.replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            clauses.append("question LIKE ? ESCAPE '\\'")
            params.append(f"%{escaped}%")
 
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
 
        return where, params
 
    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
 
    def save(self, trace, run_id=None):
        """Insert, or replace if this trace_id already exists."""
        summary = trace.summary()
        summary["run_id"] = run_id
 
        with closing(self._connect()) as conn:
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO traces
                    (trace_id, created_at, run_id, question, config_label,
                     config_fingerprint, action, refused, failed_stage,
                     latency_ms, total_tokens, summary_json, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trace.trace_id,
                        trace.created_at,
                        run_id,
                        trace.question,
                        trace.config_label,
                        trace.config_fingerprint,
                        trace.action,
                        1 if trace.refused else 0,
                        summary["failed_stage"],
                        trace.total_latency_ms,
                        trace.usage.get("total_tokens", 0),
                        json.dumps(summary),
                        trace.to_json(indent=None),
                    ),
                )
 
    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
 
    def get(self, trace_id):
        """The full Trace, or None if it does not exist."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT payload_json FROM traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
 
        return Trace.from_dict(json.loads(row["payload_json"])) if row else None
 
    def list(self, limit=50, offset=0, **filters):
        """Newest first. Returns short summary dicts (plus run_id)."""
        where, params = self._where(**filters)
 
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT summary_json FROM traces {where} {_ORDER} LIMIT ? OFFSET ?",
                params + [limit, offset],
            ).fetchall()
 
        return [json.loads(row["summary_json"]) for row in rows]
 
    def count(self, **filters):
        where, params = self._where(**filters)
 
        with closing(self._connect()) as conn:
            return conn.execute(
                f"SELECT COUNT(*) FROM traces {where}", params
            ).fetchone()[0]
 
    def list_runs(self):
        """One row per evaluation run, newest first."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT run_id,
                       COUNT(*)                         AS traces,
                       MIN(created_at)                  AS first_at,
                       MAX(created_at)                  AS last_at,
                       GROUP_CONCAT(DISTINCT config_label) AS labels
                FROM traces
                WHERE run_id IS NOT NULL
                GROUP BY run_id
                ORDER BY last_at DESC, run_id DESC
                """
            ).fetchall()
 
        return [dict(row) for row in rows]
 
    def stats(self, **filters):
        """Numbers for dashboard cards."""
        where, params = self._where(**filters)
 
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT action, refused, failed_stage, latency_ms, total_tokens "
                f"FROM traces {where}",
                params,
            ).fetchall()
 
        total = len(rows)
        latencies = sorted(row["latency_ms"] or 0 for row in rows)
        by_action = {}
 
        for row in rows:
            key = row["action"] or "unknown"
            by_action[key] = by_action.get(key, 0) + 1
 
        return {
            "traces": total,
            "refused": sum(row["refused"] for row in rows),
            "with_errors": sum(1 for row in rows if row["failed_stage"]),
            "avg_latency_ms": round(sum(latencies) / total) if total else 0,
            "p50_latency_ms": _percentile(latencies, 0.50),
            "p95_latency_ms": _percentile(latencies, 0.95),
            "avg_tokens": (
                round(sum(row["total_tokens"] or 0 for row in rows) / total)
                if total
                else 0
            ),
            "by_action": by_action,
        }
 
    # ------------------------------------------------------------------
    # Delete (only when called explicitly)
    # ------------------------------------------------------------------
 
    def delete(self, trace_id):
        with closing(self._connect()) as conn:
            with conn:
                cursor = conn.execute(
                    "DELETE FROM traces WHERE trace_id = ?", (trace_id,)
                )
 
        return cursor.rowcount
 
    def delete_run(self, run_id):
        with closing(self._connect()) as conn:
            with conn:
                cursor = conn.execute("DELETE FROM traces WHERE run_id = ?", (run_id,))
 
        return cursor.rowcount
 
    def prune(self, keep_last=1000):
        """Keep only the newest `keep_last` traces; returns how many were deleted."""
        if not isinstance(keep_last, int) or keep_last < 0:
            raise ValueError("keep_last must be an integer >= 0.")
 
        with closing(self._connect()) as conn:
            with conn:
                cursor = conn.execute(
                    f"""
                    DELETE FROM traces WHERE trace_id NOT IN (
                        SELECT trace_id FROM traces {_ORDER} LIMIT ?
                    )
                    """,
                    (keep_last,),
                )
 
        return cursor.rowcount
 
 
# ---------------------------------------------------------------------------
# Shared store + safe save
# ---------------------------------------------------------------------------
 
@lru_cache(maxsize=1)
def get_store():
    return TraceStore()
 
 
def save_trace(trace, run_id=None):
    """Save a trace WITHOUT ever raising: tracing must never break an answer.
    Returns True if saved, False if tracing is disabled or saving failed."""
    if not TRACE_ENABLED:
        return False
 
    try:
        get_store().save(trace, run_id)
        return True
    except Exception as error:
        logger.warning("Could not save trace: %s", error)
        return False
 