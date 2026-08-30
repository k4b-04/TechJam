"""agent/logger.py - TEMP, Owner E replaces this and owns the schema."""
import json
import os
from datetime import datetime

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runlog.jsonl")


def log_iteration(entry: dict, path: str = LOG_PATH) -> None:
    """Never raises - a logging failure must not kill the run."""
    try:
        entry.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"  [logger] failed to write: {exc}")
