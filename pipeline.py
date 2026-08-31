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
  - the FIELDS list + build_encoding() below (not data.encode(), which is no longer
    imported) is what makes feature changes actually take effect — see build_encoding's
    docstring for why. This is the fix for the "agent proposes 3 feature sets, gets the
    identical model 3 times" bug.

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

from data import load   # noqa: E402  — encode/FIELDS deliberately NOT imported; data.encode()
                          # hardcodes 5 fields at fixed tuple positions and ignores FIELDS past
                          # len(), so any agent edit to a field list there silently does nothing.
                          # build_encoding() below is the real, editable replacement.
from evaluate import evaluate            # noqa: E402
import csv                                # noqa: E402
import numpy as np                        # noqa: E402

# ============================================================================
# THIS is what the agent should edit to try a different feature set. Keep the
# same 5 fields as the default — that's the acceptance test (must still
# reproduce valid primary 0.6016). Anything below not in this list, but
# present in EXTRA_USER_COLS/EXTRA_VIDEO_COLS, is available to add.
# ============================================================================
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']

# Optional extra columns loaded from user_features_pure.csv / video_features_basic_pure.csv,
# mirroring ablation_features.py's own column choice (author_id excluded here — it's already
# available via the base row, same as in data.load()).
EXTRA_USER_COLS = ['follow_user_num_range', 'register_days_range', 'fans_user_num_range',
                    'friend_user_num_range', 'user_active_degree']
EXTRA_VIDEO_COLS = ['music_id', 'video_type', 'upload_type']


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


def _bucket_edges(durations, n=10):
    return np.quantile(np.asarray(durations), np.linspace(0, 1, n + 1)[1:-1])


def _load_extra_features(data_dir):
    """Same two CSVs ablation_features.py reads, so the extra columns are ready for
    the agent to reach for without it having to write its own CSV-loading code."""
    u_ext = {}
    with open(os.path.join(data_dir, 'user_features_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            u_ext[r['user_id']] = {k: r.get(k, 'UNK') for k in EXTRA_USER_COLS}
    v_ext = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            v_ext[r['video_id']] = {k: r.get(k, 'UNK') for k in EXTRA_VIDEO_COLS}
    return u_ext, v_ext


def build_encoding(splits, fields, data_dir):
    """Same vocabulary/offset logic as data.py's encode() (lines 44-50) — copied,
    not reimplemented — except raw() is driven by `fields` instead of a hardcoded
    5-tuple. This is the actual fix: previously an agent could edit FIELDS all it
    wanted and the model trained identically every time, because data.encode()
    read tuple positions directly and only ever used FIELDS for len(FIELDS).
    """
    tr = splits['train']
    edges = _bucket_edges([x[5] for x in tr])   # x[5] = duration_ms, same position as data.py
    u_ext, v_ext = _load_extra_features(data_dir)

    def raw(x):
        # x = (date, user_id, video_id, author_id, tab, duration_ms, label) — data.load()'s shape
        base = {
            'user_id': x[1], 'video_id': x[2], 'author_id': x[3], 'tab': x[4],
            'dur_bucket': str(int(np.searchsorted(edges, x[5]))),
        }
        eu, ev = u_ext.get(x[1], {}), v_ext.get(x[2], {})
        out = []
        for f in fields:
            if f in base:
                out.append(base[f])
            elif f in eu:
                out.append(eu[f])
            elif f in ev:
                out.append(ev[f])
            else:
                # loud failure, not a silent no-op — this is the exact bug being fixed.
                # The traceback lands in run_candidate's stderr and flows into
                # last_error, giving the agent something concrete to repair from.
                raise ValueError(
                    f"unknown field {f!r} — not in the base row, "
                    f"EXTRA_USER_COLS {EXTRA_USER_COLS}, or EXTRA_VIDEO_COLS {EXTRA_VIDEO_COLS}"
                )
        return out

    vocabs = [dict() for _ in fields]
    for x in tr:
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    enc = {}
    for name, rws in splits.items():
        X = np.empty((len(rws), len(fields)), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            y[n] = x[6]
            users.append(x[1])
        enc[name] = (X, y, users)
    return enc, int(sum(field_dims))


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
    enc, dim = build_encoding(splits, FIELDS, DATA_DIR)
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


SEEDS = [0, 1, 2]


def main():
    # Multi-seed scoring: gains of 0.0012-0.0021 are close to the seed-noise
    # floor (0.0008, itself measured on a single seed each time before this
    # change) — a single-seed score can't reliably distinguish candidates in
    # that range. verbose=(s == 0) keeps console output to one seed's worth
    # instead of tripling it.
    results = [train_and_score(seed=s, verbose=(s == 0)) for s in SEEDS]
    gauc = sum(r[0] for r in results) / len(results)
    ndcg5 = sum(r[1] for r in results) / len(results)
    primary = sum(r[2] for r in results) / len(results)
    primary_std = float(np.std([r[2] for r in results]))  # costs nothing, real evidence for the writeup

    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "gauc": float(gauc), "ndcg5": float(ndcg5), "primary": float(primary),
            "primary_std": primary_std,
        }, f)
    print(f"primary={primary:.4f}\u00b1{primary_std:.4f} gauc={gauc:.4f} ndcg5={ndcg5:.4f} "
          f"(mean over seeds {SEEDS})")


if __name__ == "__main__":
    main()
