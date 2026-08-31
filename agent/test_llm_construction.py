"""Regression test for the client-construction crash.

Bug: LLMClient.__post_init__ let ImportError (missing anthropic/openai
package) and provider-construction failures (bad/missing auth, version
mismatches) propagate straight out of propose_change() instead of coming
back as {"ok": False, "error": ...}. An unattended overnight run cannot
survive an uncaught exception on the very first LLM call.

Fix: __post_init__ now wraps client construction and converts any failure
there into LLMError, which propose_change's existing `except LLMError`
already knows how to turn into a dict.

This needs no API key — every case here fails before any network call.
"""
import os
import sys

failures = []

def check(name, env_overrides, expect_substring):
    old = {k: os.environ.get(k) for k in env_overrides}
    os.environ.update(env_overrides)
    try:
        # llm.py must be re-imported fresh per case: get_client() caches a
        # module-level singleton, and importing it also runs the dotenv
        # load — both would leak state between cases otherwise.
        for mod in ("llm", "prompts"):
            sys.modules.pop(mod, None)
        from llm import propose_change
        try:
            r = propose_change(current_code="print(1)", history=[], last_error=None, best_score=None)
        except Exception as exc:
            failures.append(f"{name}: RAISED instead of returning a dict — {type(exc).__name__}: {exc}")
            return
        if not isinstance(r, dict) or r.get("ok") is not False:
            failures.append(f"{name}: expected ok=False dict, got {r!r}")
        elif expect_substring not in (r.get("error") or ""):
            failures.append(f"{name}: error message missing {expect_substring!r} — got {r.get('error')!r}")
        else:
            print(f"[ok] {name}: {r['error']}")
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

# Missing package: openai isn't installed in this environment.
check("missing_package", {"LLM_PROVIDER": "openai"}, "package not installed")

# Unknown provider name.
check("unknown_provider", {"LLM_PROVIDER": "not_a_real_provider"}, "No model set")

print()
if failures:
    print(f"{len(failures)} FAILED:")
    for f in failures:
        print(" -", f)
    raise SystemExit(1)
print("client-construction crash stays fixed: 2/2 passed")
