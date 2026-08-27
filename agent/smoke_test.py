"""
Live test — THIS ONE COSTS MONEY (a fraction of a cent). Needs an API key.

    python3 smoke_test.py

It sends one real request with a tiny fake "pipeline" and checks that a real
model returns something we can parse. If this passes, your module works and
Owner A can plug it in.
"""

import os
import sys

from llm import propose_change, METER

FAKE_PIPELINE = '''\
"""A deliberately terrible ranking pipeline, for testing the agent loop."""
import numpy as np


def train(rows):
    """Currently ignores every feature and predicts a constant."""
    return 0.5


def predict(model, rows):
    return np.full(len(rows), model)
'''


def main():
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")):
        print("No API key found in the environment.")
        print("  export ANTHROPIC_API_KEY=sk-ant-...   (or OPENAI_API_KEY)")
        return 1

    print(f"provider = {os.getenv('LLM_PROVIDER', 'anthropic')}")
    print("sending one request...\n")

    result = propose_change(
        current_code=FAKE_PIPELINE,
        history=[
            {"iter": 1, "hypothesis": "predict item popularity instead of a constant",
             "primary": 0.5715, "delta": 0.0881, "kept": True},
        ],
        best_score=0.5715,
    )

    if not result["ok"]:
        print(f"FAILED: {result['error']}")
        print("\n--- raw reply ---")
        print(result["raw"][:2000] or "(empty)")
        return 1

    print("HYPOTHESIS")
    print(f"  {result['hypothesis']}\n")
    print("CODE (first 20 lines)")
    for line in result["code"].splitlines()[:20]:
        print(f"  {line}")
    print(f"  ... {len(result['code'].splitlines())} lines total\n")

    print("TOKENS")
    print(f"  in  {result['tokens_in']:,}")
    print(f"  out {result['tokens_out']:,}")
    print(f"  running total for this process: {METER.report()}\n")

    # sanity: the model should have returned valid python
    import ast
    try:
        ast.parse(result["code"])
        print("returned code parses as valid Python  ✓")
    except SyntaxError as exc:
        print(f"returned code is NOT valid Python: {exc}")
        print("(this is fine and expected sometimes — Owner C's runner catches it)")

    print("\nsmoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
