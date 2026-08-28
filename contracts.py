# contracts.py — the interface everyone else codes to.
# Matches the 6-step loop: Think -> Write -> Run -> Score -> Judge -> repeat/Stop

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Attempt:
    """One row in the history we show the LLM and log to disk."""
    iteration: int
    hypothesis: str            # why we tried this
    code_diff: str
    valid_score: Optional[float]
    accepted: bool              # True if it became the new best
    error: Optional[str] = None


@dataclass
class ProposedChange:
    hypothesis: str             # the LLM's stated reasoning
    modified_code: str          # raw text pulled from the LLM's reply


@dataclass
class WriteResult:
    file_path: str
    is_valid_syntax: bool
    syntax_error: Optional[str] = None


@dataclass
class RunResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    duration_s: float


@dataclass
class JudgeResult:
    accepted: bool               # True = new best, False = reverted
    current_best_score: float
    current_best_file: str


# ---- Step 1: Think ----
def think(history: list[Attempt]) -> ProposedChange:
    """Owner A/B: send code + history to the LLM, get back one proposed change."""
    raise NotImplementedError


# ---- Step 2: Write ----
def write_code(change: ProposedChange) -> WriteResult:
    """Owner A/C: extract code from the LLM reply, save to file, validate syntax
    (e.g. via ast.parse or py_compile) before ever running it."""
    raise NotImplementedError


# ---- Step 3: Run ----
def run_code(write_result: WriteResult, timeout_s: int = 300) -> RunResult:
    """Owner C: execute as a subprocess with a hard timeout, capture stdout/stderr."""
    raise NotImplementedError


# ---- Step 4: Score ----
def score_result(run_result: RunResult) -> Optional[float]:
    """Owner A: parse run output, evaluate on the validation split only.
    Returns None if the run failed or produced no usable score."""
    raise NotImplementedError


# ---- Step 5: Judge ----
def judge(new_score: Optional[float], current_best: JudgeResult) -> JudgeResult:
    """Owner A: better than current best -> accept as new base.
    Worse (or None) -> revert to previous best. Always logs what happened."""
    raise NotImplementedError


# ---- Step 6: repeat or stop ----
def should_stop(history: list[Attempt], epsilon: float = 0.002, n: int = 3,
                 time_budget_s: Optional[float] = None,
                 elapsed_s: float = 0.0) -> bool:
    """Owner A: True if the last n iterations improved by < epsilon,
    or if the time budget has been hit."""
    raise NotImplementedError
