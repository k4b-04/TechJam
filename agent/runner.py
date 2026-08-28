"""Owner C — sandbox. Runs model-written code and guarantees a dict comes back,
never an exception.

Candidate code is untrusted input: syntax errors, infinite loops, missing
imports, crashes. None of that may propagate past this module.
"""
import ast
import os
import subprocess
import sys
import time

TAIL_CHARS = 3000  # tracebacks put the useful part at the end


def _tail(s: str, limit: int = TAIL_CHARS) -> str:
    return s if len(s) <= limit else s[-limit:]


def run_candidate(code: str, workdir: str, timeout: int) -> dict:
    """Syntax-check, write, and execute one candidate pipeline file.

    Returns a dict that never raises:
        {"ok": bool, "stdout": str, "stderr": str, "error": str | None,
         "duration_s": float}

    ok=False + error set covers every failure mode: bad syntax (never
    executed), a timeout, or a non-zero exit. The loop checks "ok", not
    exceptions.
    """
    os.makedirs(workdir, exist_ok=True)

    # Syntax-check first. Invalid code is never executed.
    try:
        ast.parse(code)
    except SyntaxError as e:
        return {"ok": False, "stdout": "", "stderr": "",
                "error": f"SyntaxError: {e}", "duration_s": 0.0}

    path = os.path.join(workdir, "candidate.py")
    with open(path, "w") as fh:
        fh.write(code)

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, "candidate.py"],   # never the string "python"
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration = time.monotonic() - t0
        return {
            "ok": proc.returncode == 0,
            "stdout": _tail(proc.stdout),
            "stderr": _tail(proc.stderr),
            "error": None if proc.returncode == 0 else f"exit code {proc.returncode}",
            "duration_s": duration,
        }
    except subprocess.TimeoutExpired as e:
        duration = time.monotonic() - t0
        return {
            "ok": False,
            "stdout": _tail(e.stdout or ""),
            "stderr": _tail(e.stderr or ""),
            "error": f"timeout after {timeout}s",
            "duration_s": duration,
        }
