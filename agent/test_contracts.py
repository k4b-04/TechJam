"""Run this after touching llm.py, runner.py, metrics.py, or logger.py.
It doesn't test correctness — test_parsing.py and test_runner.py do that.
It only checks the shape: does the real return value have exactly the
keys contracts.py says it should. This is what would have caught
contracts.py drifting from llm.py before tonight's integration.

Modules that don't exist yet are skipped, not failed — land your file,
then this starts checking it for free.
"""
import tempfile, shutil
from contracts import ProposeResult, RunResult, MetricsResult

failures = []

def check(name, result: dict, expected_type):
    expected_keys = set(expected_type.__annotations__.keys())
    actual_keys = set(result.keys())
    if actual_keys != expected_keys:
        failures.append(
            f"{name}: keys differ\n"
            f"  contract expects: {sorted(expected_keys)}\n"
            f"  actually returned: {sorted(actual_keys)}"
        )
    else:
        print(f"[ok] {name} matches contract: {sorted(actual_keys)}")

# ---- Owner B: propose_change ----
try:
    from llm import propose_change
    r = propose_change(current_code="print('hi')", history=[], last_error=None, best_score=None)
    check("propose_change", r, ProposeResult)
except ImportError:
    print("[skip] llm.py not importable (missing key/deps) — run smoke_test.py separately")

# ---- Owner C: run_candidate ----
from runner import run_candidate
workdir = tempfile.mkdtemp()
try:
    r = run_candidate("print('hi')", workdir, timeout=5)
    check("run_candidate", r, RunResult)
finally:
    shutil.rmtree(workdir, ignore_errors=True)

# ---- Owner D: read_metrics (not built yet) ----
try:
    from metrics import read_metrics
    r = read_metrics(workdir="/tmp")
    check("read_metrics", r, MetricsResult)
except ImportError:
    print("[skip] metrics.py doesn't exist yet — Owner D")

print()
if failures:
    print(f"{len(failures)} CONTRACT MISMATCH(ES):\n")
    for f in failures:
        print(f)
    raise SystemExit(1)
else:
    print("all present modules match contracts.py")
