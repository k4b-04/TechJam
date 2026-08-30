"""agent/summarise.py — Owner E.

Read ``runlog.jsonl`` and print a README-ready Markdown summary of the
agent's autonomous run. Malformed lines are skipped rather than crashing.

Usage:
    python summarise.py
    python summarise.py path/to/runlog.jsonl
    python summarise.py --gpu-hours 0
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG_PATH = os.path.join(HERE, "runlog.jsonl")
DEFAULT_RUN_SUMMARY = os.path.join(HERE, "run_summary.json")
BASELINE_PRIMARY = 0.6016  # official validation baseline used by loop.py


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def read_runlog(path: str) -> tuple[list[dict], int]:
    """Return parsed JSON-object rows and the number of skipped bad lines."""
    entries: list[dict] = []
    skipped = 0

    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    skipped += 1
                    continue
                if not isinstance(obj, dict):
                    skipped += 1
                    continue
                entries.append(obj)
    except (OSError, UnicodeError):
        # The CLI reports a useful empty summary instead of throwing a stack
        # trace. This mirrors the rest of the agent's no-crash philosophy.
        return [], 1

    return entries, skipped


def _failure_type(error: Any) -> str:
    """Collapse error strings into useful categories for the README table."""
    text = str(error or "unknown").strip()
    lower = text.lower()

    if lower.startswith("llm:"):
        if "rate" in lower or "429" in lower:
            return "llm/rate_limit"
        return "llm"

    if lower.startswith("metrics:"):
        if "missing" in lower:
            return "metrics/missing"
        if "parse" in lower or "malformed" in lower:
            return "metrics/malformed"
        return "metrics"

    if lower.startswith("run:"):
        if "syntax" in lower:
            return "run/syntax_error"
        if "timeout" in lower:
            return "run/timeout"
        if "exit_code" in lower or "exit code" in lower:
            return "run/nonzero_exit"
        return "run"

    # Fallback for older logs that did not include llm:/run:/metrics: prefixes.
    if "syntax" in lower:
        return "run/syntax_error"
    if "timeout" in lower:
        return "run/timeout"
    return text.split(":", 1)[0][:60] or "unknown"


def _load_run_summary(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, UnicodeError):
        return {}


def _fmt_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    whole = int(round(seconds))
    h, rem = divmod(whole, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s ({seconds:.1f}s)"
    if m:
        return f"{m}m {s}s ({seconds:.1f}s)"
    return f"{seconds:.1f}s"


def build_summary(
    entries: list[dict],
    skipped: int = 0,
    run_summary: dict | None = None,
    gpu_hours: float = 0.0,
) -> dict:
    """Aggregate the facts requested by the Day-2 Owner-E handoff."""
    run_summary = run_summary or {}

    scored = [e for e in entries if _as_number(e.get("primary")) is not None]
    primaries = [_as_number(e.get("primary")) for e in scored]
    primaries = [p for p in primaries if p is not None]
    best_primary = max(primaries) if primaries else None

    kept = sum(bool(e.get("kept")) for e in entries)
    interventions = sum(bool(e.get("intervention")) for e in entries)
    recovered = sum(bool(e.get("recovered")) for e in entries)

    failures = collections.Counter()
    failed_iterations = 0
    for entry in entries:
        errors = entry.get("errors", [])
        if isinstance(errors, str):
            errors = [errors]
        if not isinstance(errors, list):
            errors = [str(errors)]
        nonempty = [err for err in errors if err]
        if nonempty:
            failed_iterations += 1
        for err in nonempty:
            failures[_failure_type(err)] += 1

    tokens_in = sum(int(e.get("tokens_in") or 0) for e in entries)
    tokens_out = sum(int(e.get("tokens_out") or 0) for e in entries)

    # A's loop.py writes the actual end-to-end wall clock to run_summary.json.
    # Prefer it when present; otherwise fall back to the sum of logged iteration
    # durations (useful for a partial/in-progress run).
    wall_clock = _as_number(run_summary.get("wall_clock_s"))
    wall_clock_source = "run_summary.json"
    if wall_clock is None:
        wall_clock = sum(float(e.get("duration_s") or 0.0) for e in entries)
        wall_clock_source = "sum of logged iteration durations"

    return {
        "logged_iterations": len(entries),
        "scored_iterations": len(scored),
        "best_primary": best_primary,
        "best_delta_vs_baseline": (
            best_primary - BASELINE_PRIMARY if best_primary is not None else None
        ),
        "kept_iterations": kept,
        "failed_iterations": failed_iterations,
        "failures_by_type": dict(sorted(failures.items())),
        "recovered_failures": recovered,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tokens_total": tokens_in + tokens_out,
        "wall_clock_s": wall_clock,
        "wall_clock_source": wall_clock_source,
        "gpu_hours": float(gpu_hours),
        "interventions": interventions,
        "skipped_lines": skipped,
    }


def render_markdown(s: dict) -> str:
    best = s["best_primary"]
    delta = s["best_delta_vs_baseline"]
    best_s = f"{best:.4f}" if best is not None else "N/A"
    delta_s = f"{delta:+.4f}" if delta is not None else "N/A"

    lines = [
        "## Autonomous Run Summary",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Iterations logged | {s['logged_iterations']} |",
        f"| Iterations that scored | {s['scored_iterations']} |",
        f"| Best validation primary | {best_s} |",
        f"| Delta vs official validation baseline ({BASELINE_PRIMARY:.4f}) | {delta_s} |",
        f"| Candidate iterations kept | {s['kept_iterations']} |",
        f"| Failed iterations | {s['failed_iterations']} |",
        f"| Failures recovered on a later repair iteration | {s['recovered_failures']} |",
        f"| Manual interventions | {s['interventions']} |",
        f"| LLM input tokens | {s['tokens_in']:,} |",
        f"| LLM output tokens | {s['tokens_out']:,} |",
        f"| LLM total tokens | {s['tokens_total']:,} |",
        f"| Total wall-clock | {_fmt_duration(s['wall_clock_s'])} |",
        f"| GPU-hours | {s['gpu_hours']:.4f} |",
    ]

    lines += ["", "### Failure / Recovery Breakdown", ""]
    if s["failures_by_type"]:
        lines += ["| Failure type | Count |", "|---|---:|"]
        for kind, count in s["failures_by_type"].items():
            lines.append(f"| `{kind}` | {count} |")
    else:
        lines.append("No failures were recorded.")

    if s["skipped_lines"]:
        lines += [
            "",
            f"> Note: skipped {s['skipped_lines']} unparseable/non-object JSONL line(s).",
        ]

    lines += [
        "",
        f"_Wall-clock source: {s['wall_clock_source']}._",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=DEFAULT_LOG_PATH,
                    help="path to runlog.jsonl")
    ap.add_argument("--gpu-hours", type=float, default=0.0,
                    help="GPU-hours used by the run (default 0 for CPU-only)")
    ap.add_argument("--run-summary", default=None,
                    help="optional path to loop.py's run_summary.json")
    args = ap.parse_args()

    entries, skipped = read_runlog(args.path)
    summary_path = args.run_summary or os.path.join(
        os.path.dirname(os.path.abspath(args.path)), "run_summary.json"
    )
    run_summary = _load_run_summary(summary_path)
    summary = build_summary(entries, skipped, run_summary, args.gpu_hours)
    print(render_markdown(summary))


if __name__ == "__main__":
    main()
