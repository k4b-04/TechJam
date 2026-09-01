"""agent/verify_seeds.py — Owner D. Run ONCE, by hand, on the agent's kept
best candidate to confirm it isn't fooling itself on a single lucky seed
before trusting it as the real result.

NOT part of the agent's editable loop, and deliberately NOT a flag on
pipeline.py's own main(). Two reasons:
  1. main() is what run_candidate() executes every single iteration (see
     runner.py) — it must stay single-seed and fast. Averaging 3 seeds per
     iteration triples the cost of every candidate the loop evaluates, not
     just the ones where the extra precision matters. That tripling is
     exactly what combined with the O(batch_size^2) BPR bug (see prompts.py
     rule 9) to blow through the 600s timeout on the very first iteration.
  2. Every iteration's candidate.py is a complete file rewrite by the LLM
     (see prompts.py's OUTPUT FORMAT) — a CLI flag baked into main() would
     have no guarantee of surviving the next rewrite. A standalone script
     that just calls train_and_score() per seed has no such dependency.

Usage:
    python agent/verify_seeds.py                    # verifies agent/best_pipeline.py
    python agent/verify_seeds.py --seeds 0,1,2,3,4
    python agent/verify_seeds.py path/to/pipeline.py
"""
from __future__ import annotations

import argparse
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Measured seed-to-seed noise floor on the FM baseline (see prompts.py's
# CALIBRATION section) — the yardstick for whether a candidate's improvement
# is real or just got a lucky seed.
NOISE_FLOOR = 0.0008


def _load_pipeline(path: str):
    spec = importlib.util.spec_from_file_location("candidate_pipeline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=os.path.join(HERE, "best_pipeline.py"),
                     help="pipeline file to verify (default: agent/best_pipeline.py)")
    ap.add_argument("--seeds", default="0,1,2,3,4",
                     help="comma-separated seeds to average over (default: 0,1,2,3,4)")
    args = ap.parse_args()

    if not os.path.exists(args.path):
        raise SystemExit(
            f"{args.path} does not exist — has the agent loop kept an improved "
            f"candidate yet? (agent/loop.py only writes best_pipeline.py once a "
            f"candidate beats the baseline.)"
        )

    seeds = [int(s) for s in args.seeds.split(",")]
    mod = _load_pipeline(args.path)

    print(f"verifying {args.path} over seeds {seeds} ...")
    results = [mod.train_and_score(seed=s, verbose=False) for s in seeds]

    gauc = float(np.mean([r[0] for r in results]))
    ndcg5 = float(np.mean([r[1] for r in results]))
    primary = float(np.mean([r[2] for r in results]))
    primary_std = float(np.std([r[2] for r in results]))

    print(f"  primary = {primary:.4f} ± {primary_std:.4f}   (per-seed: "
          + ", ".join(f"{r[2]:.4f}" for r in results) + ")")
    print(f"  gauc    = {gauc:.4f}")
    print(f"  ndcg5   = {ndcg5:.4f}")

    if primary_std > 2 * NOISE_FLOOR:
        print(f"  WARNING: seed spread ({primary_std:.4f}) is more than 2x the "
              f"measured noise floor ({NOISE_FLOOR}) — this candidate's improvement "
              f"may not be real. Don't report it as the final result without more seeds.")


if __name__ == "__main__":
    main()
