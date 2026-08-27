import subprocess
import sys
import time
from dataclasses import dataclass, asdict

@dataclass
class RunResult:
    ok: bool
    returncode: int | None   # None if timed out
    timed_out: bool
    wall_time_s: float
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    error: str | None = None  # set if we can't even launch the process


def run_step(path, args = None, timeout_s = 30.0, max_capture_chars = 20000, cwd = None) -> RunResult:
    # Run python path *args as a subprocess, capture output, enforce timeout.
    args = args or []
    cmd = [sys.executable, path, *args]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        wall = time.monotonic() - t0
        out, out_trunc = _truncate(proc.stdout, max_capture_chars)
        err, err_trunc = _truncate(proc.stderr, max_capture_chars)
        return RunResult(
            ok=(proc.returncode == 0),
            returncode=proc.returncode,
            timed_out=False,
            wall_time_s=wall,
            stdout=out,
            stderr=err,
            stdout_truncated=out_trunc,
            stderr_truncated=err_trunc,
        )
    except subprocess.TimeoutExpired as e:
        wall = time.monotonic() - t0
        out, out_trunc = _truncate(e.stdout or "", max_capture_chars)
        err, err_trunc = _truncate(e.stderr or "", max_capture_chars)
        return RunResult(
            ok=False,
            returncode=None,
            timed_out=True,
            wall_time_s=wall,
            stdout=out,
            stderr=err,
            stdout_truncated=out_trunc,
            stderr_truncated=err_trunc,
            error=f"timed out after {timeout_s}s",
        )
    except OSError as e:
        # e.g. file not found, not executable, permission denied
        wall = time.monotonic() - t0
        return RunResult(
            ok=False, 
            returncode=None, 
            timed_out=False, 
            wall_time_s=wall,
            stdout="", 
            stderr="", 
            stdout_truncated=False, 
            stderr_truncated=False,
            error=f"launch failed: {e}",
        )


def _truncate(s: str, limit: int) -> tuple[str, bool]:
    if len(s) <= limit:
        return s, False
    return s[:limit] + f"\n...[truncated, {len(s) - limit} more chars]", True


if __name__ == "__main__":
    import json
    target = sys.argv[1]
    timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    r = run_step(target, timeout_s=timeout)
    print(json.dumps(asdict(r), indent=2)[:3000])
