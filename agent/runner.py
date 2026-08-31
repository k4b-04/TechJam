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


def run_candidate(code: str, workdir: str, timeout: int = 600) -> dict:
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
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(code)

    # candidate.py runs with cwd=workdir (a scratch dir) but still needs to
    # `import data` / `import evaluate`, which live at the repo root — one
    # level above this file. Two owners independently landed on this exact
    # convention: loop.py's temporary run_candidate() sets these same env
    # vars, and pipeline.py reads them (falling back to its own __file__
    # location, which breaks once code is copied into a tempdir as text —
    # see pipeline.py's own docstring, which flags this as an open question
    # for this module). Set both these AND PYTHONPATH: the env vars are what
    # pipeline.py actually reads today, PYTHONPATH is a general fallback for
    # any future candidate code that just does a bare `import data`.
    env = os.environ.copy()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["KUAIRAND_REPO_ROOT"] = repo_root
    env["KUAIRAND_DATA_DIR"] = os.path.join(repo_root, "KuaiRand-Pure", "data")
    env["PYTHONPATH"] = os.pathsep.join([repo_root, env.get("PYTHONPATH", "")])
    env["PYTHONIOENCODING"] = "utf-8"  # child's own stdout/stderr, not just how
    env["PYTHONUTF8"] = "1"            # the parent decodes the pipe — Windows
                                        # defaults both to cp1252 otherwise

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, "candidate.py"],   # never the string "python"
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",       # Windows defaults to cp1252 otherwise —
            errors="replace",        # never crash the runner over a decode error
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