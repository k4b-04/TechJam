"""agent/loop.py — Owner A. The orchestrator.

Read this before you run it. You own this file and you will be asked about it —
the whole submission is judged on what happens in this loop.

    python loop.py                      # default: 20 iterations, 2h budget
    python loop.py --max-iters 3        # short smoke run
    python loop.py --budget-s 3600      # 1 hour

Calls, in order, once per iteration:
    llm.propose_change()   (B)  ->  hypothesis + candidate code
    run_candidate()        (C)  ->  executed in a sandbox, with a timeout
    metrics.read_metrics() (D)  ->  the validation score
    logger.log_iteration() (E)  ->  one JSON line of evidence

Design rules, all four load-bearing:
  1. Nothing raises. Every helper returns {"ok": bool, "error": str|None}.
     A crash at 3am ends the overnight run; a logged failure does not.
  2. A failed iteration NEVER updates best_code. It feeds its error back to B
     as last_error, which flips B's prompt into repair mode.
  3. Every file operation passes encoding="utf-8". Windows defaults to cp1252
     and silently corrupts generated code. This cost us two hours on Day 1.
  4. We score against the VALIDATION split only. The test split is popped out
     of memory inside pipeline.py and must stay that way.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import time

from llm import propose_change, METER
from metrics import read_metrics
from logger import log_iteration

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

# The official baseline's validation primary. Deltas in the log are measured
# against this, so the numbers mean the same thing as the judges' scoring.
BASELINE_PRIMARY = 0.6016

# Convergence rule, fixed by the organisers: stop when the score has not
# improved by more than EPSILON across PATIENCE consecutive iterations.
EPSILON = 0.002
PATIENCE = 3


# ---------------------------------------------------------------------------
# TEMPORARY runner — Owner C replaces this with runner.run_candidate().
# Kept here so the loop is runnable before C's module lands. Same contract.
# ---------------------------------------------------------------------------

def run_candidate(code: str, workdir: str, timeout: int = 600) -> dict:
    """Execute model-written code in its own process. Never raises."""
    base = {"ok": False, "stdout": "", "stderr": "", "error": None, "duration_s": 0.0}

    # Cheap gate first: don't spend 100 seconds finding out it won't parse.
    try:
        ast.parse(code)
    except SyntaxError as exc:
        base["error"] = f"syntax_error: {exc}"
        return base

    os.makedirs(workdir, exist_ok=True)
    with open(os.path.join(workdir, "candidate.py"), "w", encoding="utf-8") as f:
        f.write(code)

    # pipeline.py imports data.py / evaluate.py from the repo root and reads the
    # dataset. Neither is reachable from workdir, so we point it back via env.
    env = {
        **os.environ,
        "KUAIRAND_REPO_ROOT": REPO_ROOT,
        "KUAIRAND_DATA_DIR": os.path.join(REPO_ROOT, "KuaiRand-Pure", "data"),
    }

    t0 = time.time()
    try:
        p = subprocess.run(
            [sys.executable, "candidate.py"],
            cwd=workdir, env=env,
            capture_output=True, text=True, encoding="utf-8",
            timeout=timeout,
        )
        base["duration_s"] = round(time.time() - t0, 1)
        base["stdout"] = (p.stdout or "")[-3000:]
        base["stderr"] = (p.stderr or "")[-3000:]
        base["ok"] = p.returncode == 0
        if not base["ok"]:
            base["error"] = f"exit_code_{p.returncode}: {base['stderr'][-800:]}"
    except subprocess.TimeoutExpired:
        base["duration_s"] = round(time.time() - t0, 1)
        base["error"] = f"timeout after {timeout}s"

    return base


# ---------------------------------------------------------------------------

def read_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def has_converged(scores: list[float]) -> bool:
    """True when the last PATIENCE iterations produced no real improvement.

    'Real' means more than EPSILON above the best score seen before that
    window opened — not merely better than the previous iteration, which
    would let noise keep the run alive forever.
    """
    if len(scores) < PATIENCE + 1:
        return False
    before = max(scores[:-PATIENCE])
    recent = max(scores[-PATIENCE:])
    return (recent - before) <= EPSILON


# ---------------------------------------------------------------------------

def run_agent(max_iters: int = 20, budget_s: int = 7200, timeout_s: int = 600) -> dict:
    started = time.time()

    best_code = read_text(os.path.join(HERE, "pipeline.py"))
    best_score = None          # filled by iteration 1
    current_code = best_code

    history: list[dict] = []   # what B sees — hypothesis + score + verdict
    scores: list[float] = []   # successful primaries only, for convergence
    last_error: str | None = None
    stop_reason = "max_iters"

    print(f"agent starting | max_iters={max_iters} budget={budget_s}s timeout={timeout_s}s")

    for i in range(1, max_iters + 1):
        elapsed = time.time() - started
        if elapsed > budget_s:
            stop_reason = "budget_exhausted"
            break

        print(f"\n=== iteration {i} | {elapsed:.0f}s elapsed | best={best_score} ===")
        t_iter = time.time()

        entry = {
            "iter": i,
            "hypothesis": None, "primary": None, "gauc": None, "ndcg5": None,
            "delta": None, "kept": False, "errors": [], "recovered": False,
            "tokens_in": 0, "tokens_out": 0, "duration_s": 0.0,
            "intervention": False, "mode": "repair" if last_error else "propose",
        }

        # --- 1. ask B ------------------------------------------------------
        prop = propose_change(current_code, history, last_error, best_score)
        entry["hypothesis"] = prop["hypothesis"]
        entry["tokens_in"] = prop["tokens_in"]
        entry["tokens_out"] = prop["tokens_out"]

        if not prop["ok"]:
            entry["errors"].append(f"llm: {prop['error']}")
            entry["duration_s"] = round(time.time() - t_iter, 1)
            log_iteration(entry)
            print(f"  llm failed: {prop['error']}")
            last_error = None          # not the pipeline's fault — don't repair
            continue

        print(f"  hypothesis: {prop['hypothesis']}")

        # --- 2. run it (C) -------------------------------------------------
        workdir = os.path.join(HERE, "runs", f"iter_{i:03d}")
        run = run_candidate(prop["code"], workdir, timeout=timeout_s)
        entry["duration_s"] = run["duration_s"]

        if not run["ok"]:
            entry["errors"].append(f"run: {run['error']}")
            log_iteration(entry)
            print(f"  run failed after {run['duration_s']}s: {run['error']}")
            # Hand the traceback to B next round; keep best_code untouched.
            last_error = run["error"] + "\n" + run["stderr"][-1500:]
            current_code = best_code
            continue

        # --- 3. score it (D) -----------------------------------------------
        m = read_metrics(workdir)
        if not m["ok"]:
            entry["errors"].append(f"metrics: {m['error']}")
            log_iteration(entry)
            print(f"  no usable score: {m['error']}")
            last_error = f"The run finished but produced no valid metrics.json: {m['error']}"
            current_code = best_code
            continue

        primary = m["primary"]
        entry.update(primary=primary, gauc=m["gauc"], ndcg5=m["ndcg5"],
                     delta=round(primary - BASELINE_PRIMARY, 4))
        entry["recovered"] = last_error is not None   # this run fixed a broken one
        last_error = None

        # --- 4. keep or revert ---------------------------------------------
        if best_score is None or primary > best_score:
            best_score, best_code = primary, prop["code"]
            current_code = best_code
            entry["kept"] = True
            write_text(os.path.join(HERE, "best_pipeline.py"), best_code)
            print(f"  primary {primary:.4f} (delta {entry['delta']:+.4f}) -> KEPT")
        else:
            current_code = best_code      # revert; next proposal builds on best
            print(f"  primary {primary:.4f} (delta {entry['delta']:+.4f}) -> rejected")

        scores.append(primary)
        log_iteration(entry)

        # --- 5. converged? --------------------------------------------------
        if has_converged(scores):
            stop_reason = "converged"
            break

    summary = {
        "stop_reason": stop_reason,
        "iterations": len(scores),
        "best_primary": best_score,
        "best_delta_vs_baseline": round(best_score - BASELINE_PRIMARY, 4) if best_score else None,
        "wall_clock_s": round(time.time() - started, 1),
        **METER.report(),
    }
    with open(os.path.join(HERE, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== finished ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-iters", type=int, default=20)
    ap.add_argument("--budget-s", type=int, default=7200)
    ap.add_argument("--timeout-s", type=int, default=600)
    a = ap.parse_args()
    run_agent(max_iters=a.max_iters, budget_s=a.budget_s, timeout_s=a.timeout_s)
