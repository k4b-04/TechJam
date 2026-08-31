"""Dry run: exercises the ENTIRE loop — runner.py, metrics.py, logger.py,
keep/revert, convergence, run_summary.json — using a fake propose_change()
so it costs nothing and needs no API key.

This is not a substitute for a real run (see bottom of this file for that).
It answers a narrower question: "does the plumbing hold together," which you
want to know before spending tokens on the real thing.

Usage: python3 test_loop_dry_run.py
"""
import json
import os
import shutil
import sys
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))

# The fake model reply: iteration 1 just returns pipeline.py unchanged, so we
# get a real train+score pass (via the REAL runner.py + metrics.py) without
# a real LLM call. This proves run_candidate, read_metrics, log_iteration,
# and the keep/revert + convergence logic in loop.py all actually work
# together, end to end.
def fake_propose_change(current_code, history, last_error, best_score):
    return {
        "hypothesis": "(dry run) re-run the seed pipeline unchanged, sanity check the loop",
        "code": current_code,
        "tokens_in": 0, "tokens_out": 0, "raw": "(dry run, no real call)",
        "ok": True, "error": None,
    }

def main():
    # Use a scratch run_summary / runlog path so this never clobbers a real run.
    runs_dir = os.path.join(HERE, "runs")
    log_path = os.path.join(HERE, "runlog_dryrun.jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)

    with mock.patch("loop.propose_change", fake_propose_change), \
         mock.patch("loop.log_iteration") as mock_log:

        import loop
        mock_log.side_effect = lambda entry: __import__("logger").log_iteration(entry, path=log_path)

        summary = loop.run_agent(max_iters=1, budget_s=180, timeout_s=180)

    print("\n--- checks ---")
    assert summary["iterations"] >= 1, "expected at least one successful iteration"
    assert summary["best_primary"] is not None, "expected a real score"
    assert abs(summary["best_primary"] - 0.6016) < 0.01, (
        f"score {summary['best_primary']} is far from published 0.6016 — "
        "something in the real chain (runner/pipeline/metrics) is off"
    )
    assert os.path.exists(log_path), "log_iteration never wrote a line"
    with open(log_path) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    assert len(lines) >= 1, "runlog_dryrun.jsonl is empty"
    assert lines[0]["hypothesis"] is not None, "hypothesis missing from logged entry"
    print(f"[ok] {len(lines)} line(s) in runlog_dryrun.jsonl, first hypothesis: "
          f"{lines[0]['hypothesis']!r}")
    print(f"[ok] best_primary={summary['best_primary']:.4f} "
          f"(published baseline: 0.6016)")
    print(f"[ok] stop_reason={summary['stop_reason']}")
    print("\nfull loop plumbing holds together: runner + metrics + logger "
          "all confirmed working through the real loop.run_agent()")

    os.remove(log_path)
    shutil.rmtree(runs_dir, ignore_errors=True)

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# To test the REAL full run (real LLM calls, real proposed changes):
#
#   cd agent
#   cp .env.example .env        # fill in your ANTHROPIC_API_KEY or OPENAI_API_KEY
#   python3 smoke_test.py       # confirm one live call works, costs a fraction of a cent
#   python3 loop.py --max-iters 1 --budget-s 300   # ONE real iteration first
#
# Check afterward:
#   - runlog.jsonl has one real entry (real hypothesis, real code_diff)
#   - agent/run_summary.json has a sensible stop_reason
#   - if kept=True, best_pipeline.py was updated
#   - METER.report() (printed in run_summary) shows non-zero tokens
#
# Only once that one real iteration looks right, scale up:
#   python3 loop.py --max-iters 20 --budget-s 7200
# ---------------------------------------------------------------------------
