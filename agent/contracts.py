"""contracts.py — the interface everyone codes to.

Source of truth for what each of the four handoff functions returns.
These are TypedDicts, not enforced at runtime by Python itself — that's
what test_contracts.py is for: it calls the real functions and checks
their output matches what's declared here. If this file and the actual
code ever disagree, that test fails, not silently drifts.

Match the loop as it's actually built: four functions, dict in/dict out,
nothing raises. Owner A's run_agent() is the only thing that calls all four.
"""
from typing import TypedDict, Optional


class ProposeResult(TypedDict):
    """Return type of llm.propose_change(). Owner B."""
    hypothesis: Optional[str]   # one-line rationale, goes in the run log
    code: Optional[str]         # complete modified pipeline file, or None
    tokens_in: int
    tokens_out: int
    raw: str                    # untouched model reply, log this too
    ok: bool
    error: Optional[str]


class RunResult(TypedDict):
    """Return type of runner.run_candidate(). Owner C."""
    ok: bool
    stdout: str
    stderr: str
    error: Optional[str]
    duration_s: float


class MetricsResult(TypedDict):
    """Return type of metrics.read_metrics(). Owner D."""
    ok: bool
    primary: Optional[float]
    gauc: Optional[float]
    ndcg5: Optional[float]
    error: Optional[str]


# ---- Function signatures, for reference. Real implementations live in
# llm.py / runner.py / metrics.py / logger.py — this file declares the
# shape, it doesn't implement anything. ----

def propose_change(current_code: str, history: list[dict],
                    last_error: Optional[str], best_score: Optional[float]) -> ProposeResult:
    ...

def run_candidate(code: str, workdir: str, timeout: int) -> RunResult:
    ...

def read_metrics(workdir: str) -> MetricsResult:
    ...

def log_iteration(entry: dict) -> None:
    """Appends one JSON line. Returns nothing — Owner E."""
    ...

def run_agent(max_iters: int, budget_s: float) -> dict:
    """Owner A: calls all four in a loop. Returns a summary of why the run
    ended (converged / budget / max_iterations) and the final best state."""
    ...
