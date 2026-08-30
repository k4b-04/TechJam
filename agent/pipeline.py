"""agent/pipeline.py — Owner D. THIS is the file run_candidate() (Owner C) writes
into workdir/candidate.py and executes every iteration — see runner.py. It is also
run_agent()'s seed for current_code (loop.py reads this file's own text at startup).

Contract this file must honor on every run (Decision 1, Day-1 notes):
  - runnable standalone with no arguments: `python3 pipeline.py`
  - writes metrics.json into its OWN cwd (== workdir once wired for real) with
    keys {gauc, ndcg5, primary} — no "ok" key; agent/metrics.py constructs that
    itself from whether the file exists/parses.
  - the hidden test split must be physically unreachable from this file (hard rule
    #1) — not just unused. See splits.pop('test', ...) below.

OPEN QUESTION — not yet confirmed with Owner C/A, flag before trusting this runs
inside the real sandbox: once run_candidate() copies this code into a fresh
tempdir and executes it there, are data.py / evaluate.py / the extracted
KuaiRand-Pure data directory reachable from that location? The two env vars below
are a placeholder answer, not a confirmed one.
"""
import collections
import json
import os
import sys
import time

# Falls back to walking up from this file's own location, which only resolves
# correctly when this file is run from its real position in the repo (agent/..).
# If run_candidate() copies just this file's *text* into an unrelated tempdir,
# this fallback will NOT find data.py/evaluate.py — confirm with Owner C whether
# the runner sets KUAIRAND_REPO_ROOT/PYTHONPATH, or copies the whole repo in.
REPO_ROOT = os.environ.get(
    "KUAIRAND_REPO_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
DATA_DIR = os.environ.get("KUAIRAND_DATA_DIR", os.path.join(REPO_ROOT, "KuaiRand-Pure", "data"))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data import load, encode, FIELDS   # noqa: E402
from evaluate import evaluate            # noqa: E402
import numpy as np                        # noqa: E402


def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

# ---------------- Factorization Machine (same as baseline.py) ----------------
class FM:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def step(self, X, y):
        B = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * g.sum()
        return float(-np.mean(y * np.log(sigmoid(z) + 1e-9) + (1 - y) * np.log(1 - sigmoid(z) + 1e-9)))

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])


def train_and_score(k=16, lr=0.001, epochs=15, bs=8192, patience=4, seed=0, verbose=True):
    """Replaces the placeholder's fixed (0.60, 0.58). Loads train+valid only (test
    split is popped out of memory, never just "unused") and trains the FM baseline.

    epochs capped at 15, not the reference's 40 — measured local runs early-stop at
    epoch 8-11 anyway, so this doesn't change the result, only removes wasted margin.
    Real runtime measured at ~15-25s/epoch (not the README's ~40s TOTAL claim) — see
    the timeout flag below before assuming this fits in run_candidate's default
    120s timeout.
    """
    print(f"loading {DATA_DIR} ...")
    splits = load(DATA_DIR)
    splits.pop("test", None)          # hard removal — test rows do not exist past this line
    enc, dim = encode(splits)
    Xtr, ytr, _ = enc["train"]; Xva, yva, uva = enc["valid"]

    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr)); t0 = time.time()
        losses = [m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]]) for i in range(0, len(idx), bs)]
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    m.V, m.W, m.b = best_state
    r = evaluate(uva, yva, m.predict(Xva))
    return r["GAUC"], r["nDCG@5"], r["primary"]


def main():
    gauc, ndcg5, primary = train_and_score()
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "gauc": float(gauc), 
            "ndcg5": float(ndcg5), 
            "primary": float(primary)
        }, f)
    print(f"primary={primary:.4f} gauc={gauc:.4f} ndcg5={ndcg5:.4f}")


if __name__ == "__main__":
    main()
