"""agent/logger.py — Owner E.

Append one JSON object per agent iteration to ``runlog.jsonl``.

This module is deliberately tiny and defensive: logging is evidence for the
Autonomy/Robustness judging criteria, but a logging failure must never crash a
long autonomous run. ``log_iteration`` therefore never raises.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(HERE, "runlog.jsonl")

# Shared run-log schema. Owner A should populate these keys before calling
# log_iteration(). Missing keys are written as their safe defaults so every
# JSONL row has a stable shape. Extra keys are preserved for forward
# compatibility.
ENTRY_DEFAULTS: dict[str, Any] = {
    "ts": None,
    "iter": None,
    "mode": None,            # "propose" or "repair"
    "hypothesis": None,
    "code_diff": None,       # unified diff: previous best -> candidate
    "primary": None,
    "gauc": None,
    "ndcg5": None,
    "delta": None,           # primary - official validation baseline (0.6016)
    "kept": False,
    "errors": [],
    "recovered": False,
    "tokens_in": 0,
    "tokens_out": 0,
    "duration_s": 0.0,
    "intervention": False,
}


def _json_default(value: Any) -> Any:
    """Best-effort conversion for numpy-like scalars without importing numpy."""
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:  # noqa: BLE001 - logger must remain best-effort
            pass
    return str(value)


def log_iteration(entry: dict, path: str = LOG_PATH) -> None:
    """Append exactly one UTF-8 JSON line. Never raises.

    ``entry`` is copied rather than mutated. Missing schema keys receive safe
    defaults; additional keys supplied by the loop are retained.
    """
    try:
        record: dict[str, Any] = {}
        for key, default in ENTRY_DEFAULTS.items():
            # Avoid sharing the mutable default list between records.
            if isinstance(default, list):
                record[key] = list(entry.get(key, default))
            else:
                record[key] = entry.get(key, default)

        if not record["ts"]:
            record["ts"] = datetime.now().isoformat(timespec="seconds")

        # Preserve any future fields that are not yet in ENTRY_DEFAULTS.
        for key, value in entry.items():
            if key not in record:
                record[key] = value

        # Build the complete line before opening the file, reducing the chance
        # of leaving a half-written JSON object if serialization fails.
        line = json.dumps(record, ensure_ascii=False, default=_json_default)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a", encoding="utf-8", newline="") as f:
            f.write(line + "\n")
    except Exception as exc:  # noqa: BLE001 - no-crash contract is intentional
        try:
            print(f"  [logger] failed to write run log: {exc}")
        except Exception:
            pass
