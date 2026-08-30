"""agent/metrics.py — Owner D. TRUSTED side of the loop; the agent (via Owner B's
propose_change) only ever gets to rewrite pipeline.py, never this file — that
separation is what makes faking a score structurally impossible rather than just
discouraged.

Contract (matches loop.py's stub exactly, and must match contracts.py's
MetricsResult — run agent/test_contracts.py after touching this file):
    {ok: bool, primary: float|None, gauc: float|None, ndcg5: float|None, error: str|None}

Never raises. A missing/malformed metrics.json is an ordinary, expected outcome
that run_agent() is built to route into repair mode via last_error — not a crash.
"""
import json
import os

# GAUC / nDCG@5 / primary are all bounded in [0, 1] by construction (see evaluate.py).
# A rewritten pipeline.py that just hardcodes a big fake number gets rejected here as
# malformed rather than trusted — this bound check is part of "impossible to cheat",
# not just validation.
_BOUNDS_OK = lambda v: isinstance(v, (int, float)) and 0.0 <= v <= 1.0


def read_metrics(workdir: str) -> dict:
    """Read metrics.json written by pipeline.py into workdir (candidate's own cwd
    when run via run_candidate). Returns the MetricsResult shape above, always."""
    base = {"ok": False, "primary": None, "gauc": None, "ndcg5": None, "error": None}
    path = os.path.join(workdir, "metrics.json")

    if not os.path.exists(path):
        base["error"] = "metrics_file_missing"
        return base

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:                      # noqa: BLE001 — any parse failure
        base["error"] = f"metrics_parse_failed: {exc}"
        return base

    if not isinstance(data, dict):
        base["error"] = "metrics.json did not contain a JSON object"
        return base

    gauc, ndcg5, primary = data.get("gauc"), data.get("ndcg5"), data.get("primary")

    for name, v in (("gauc", gauc), ("ndcg5", ndcg5), ("primary", primary)):
        if v is not None and not _BOUNDS_OK(v):
            base["error"] = f"bad value for {name}: {v!r}"
            return base

    if primary is None and gauc is not None and ndcg5 is not None:
        primary = (gauc + ndcg5) / 2  # same fallback the stub used, kept for compatibility

    if primary is None:
        base["error"] = "no usable primary score in metrics.json"
        return base

    base.update(ok=True, primary=primary, gauc=gauc, ndcg5=ndcg5)
    return base
