import shutil, tempfile
from runner import run_candidate

CASES = {
    "syntax_error": "def f(x\n    return x + 1\n",
    "infinite_loop": "import time\nwhile True:\n    time.sleep(0.1)\n",
    "missing_import": "import this_package_does_not_exist\n",
    "exit_1": "import sys\nprint('failing on purpose')\nsys.exit(1)\n",
    "ok": "print('candidate ran fine')\n",
}

failures = []
for name, code in CASES.items():
    workdir = tempfile.mkdtemp()
    try:
        timeout = 2 if name == "infinite_loop" else 10
        r = run_candidate(code, workdir, timeout)
        expected_ok = (name == "ok")
        status = "ok " if r["ok"] == expected_ok else "FAIL"
        if r["ok"] != expected_ok:
            failures.append(name)
        print(f"[{status}] {name:16s} ok={r['ok']!s:6s} "
              f"duration={r['duration_s']:.2f}s error={r['error']}")
        assert isinstance(r, dict)
        assert set(r.keys()) == {"ok", "stdout", "stderr", "error", "duration_s"}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
else:
    print("all cases returned a dict, none raised: 5/5 passed")
