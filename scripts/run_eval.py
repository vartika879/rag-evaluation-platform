"""Thin wrapper so that `python scripts/run_eval.py ...` works from the project root.

Place this file at: scripts/run_eval.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())