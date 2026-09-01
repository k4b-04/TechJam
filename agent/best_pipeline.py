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
# Selected features: including the 5 baseline fields and all robust, 
# low-cardinality extra categorical features, while excluding high-cardinality music_id.
# ============================================================================
FIELDS = [
    'user_id', 'video_id', 'author_id', 'tab', 'dur_bucket',
    'follow_user_num_range', 'register_days_range', 'fans_user_num_range',
    'friend_user_num_range', 'user_active_degree', 'video_type', 'upload_type'
]

# Optional extra columns loaded from user_features_pure.csv / video_features_basic_pure.csv.
EXTRA_USER_COLS = ['follow_user_num_range', 'register_days_range', 'fans_user_num_range',
                    'friend_user_num_range', 'user_active_degree']
EXTRA_VIDEO_COLS = ['music_id', 'video_type', 'upload_type']


def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

# ---------------- DeepFM (FwFM + MLP) with Sparse/Dense Adam ----------------
class FwFM:
    def __init__(self, dim, num_fields, k=16, lr=0.001, l2=1e-6, seed=0, hidden_dim=32):
        self.rng = np.random.default_rng(seed)
        self.V = self.rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        
        # Field-interaction weights, initialized to 1.0 (excluding diagonal)
        self.R = (np.ones((num_fields, num_fields), dtype=np.float32) - 
                  np.eye(num_fields, dtype=np.float32))
        
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mR = np.zeros_like(self.R); self.vR = np.zeros_like(self.R)
        self.t = 0

        # MLP parameters
        self.num_fields = num_fields
        self.k = k
        self.hidden_dim = hidden_dim
        
        limit1 = np.sqrt(6.0 / (num_fields * k + hidden_dim))
        self.W1 = self.rng.uniform(-limit1, limit1, (num_fields * k, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        
        limit2 = np.sqrt(6.0 / (hidden_dim + 1))
        self.W2 = self.rng.uniform(-limit2, limit2, (hidden_dim, 1)).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)
        
        self.mW1 = np.zeros_like(self.W1); self.vW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1); self.vb1 = np.zeros_like(self.b1)
        self.mW2 = np.zeros_like(self.W2); self.vW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2); self.vb2 = np.zeros_like(self.b2)

    def logits(self, X, mask_w=None, mask_v=None):
        B = len(X)
        E = self.V[X]
        W_vals = self.W[X]
        if mask_v is not None:
            E = E * mask_v[:, :, None]
        if mask_w is not None:
            W_vals = W_vals * mask_w
            
        # Pairwise field-weighted interactions: 0.5 * sum_{i,j} R_{i,j} <E_i, E_j>
        D = np.matmul(E, E.transpose(0, 2, 1))  # (B, F, F)
        inter = 0.5 * (D * self.R).sum((1, 2))  # (B,)
        
        # Non-linear MLP part (Deep component)
        H0 = E.reshape(B, -1)
        Z1 = H0 @ self.W1 + self.b1
        H1 = np.maximum(0.0, Z1)
        H2 = (H1 @ self.W2 + self.b2).reshape(-1)
        
        return self.b + W_vals.sum(1) + inter + H2, E, D, H0, Z1, H1

    def step(self, X, y, drop_rate=0.1):
        B = len(y)
        if drop_rate > 0:
            raw_mask = (self.rng.random((B, X.shape[1]), dtype=np.float32) >= drop_rate).astype(np.float32)
            mask_w = raw_mask / (1.0 - drop_rate)
            mask_v = raw_mask * np.float32(np.sqrt(1.0 / (1.0 - drop_rate)))
        else:
            mask_w = None
            mask_v = None

        z, E, D, H0, Z1, H1 = self.logits(X, mask_w, mask_v)
        g = ((sigmoid(z) - y) / B).astype(np.float32)
        
        gW = np.zeros_like(self.W); gV = np.zeros_like(self.V)
        
        # MLP backward pass
        gH2 = g[:, None]
        gW2 = H1.T @ gH2 + self.l2 * self.W2
        gb2 = gH2.sum(0)
        gH1 = gH2 @ self.W2.T
        gZ1 = gH1 * (Z1 > 0)
        gW1 = H0.T @ gZ1 + self.l2 * self.W1
        gb1 = gZ1.sum(0)
        gH0 = gZ1 @ self.W1.T
        gE_mlp = gH0.reshape(B, self.num_fields, self.k)
        
        # Combine FM and MLP gradients with respect to embeddings
        if mask_v is not None:
            gW_val = g[:, None] * mask_w
            gV_val = (g[:, None, None] * np.matmul(self.R, E) + gE_mlp) * mask_v[:, :, None]
        else:
            gW_val = g[:, None]
            gV_val = g[:, None, None] * np.matmul(self.R, E) + gE_mlp

        np.add.at(gW, X, gW_val)
        np.add.at(gV, X, gV_val)
        
        # Gradient for field-pair interaction weights R
        gR = (g[:, None, None] * D).sum(0) + self.l2 * self.R
        gR = 0.5 * (gR + gR.T)
        np.fill_diagonal(gR, 0)
        
        # Sparse Adam updates for V and W: only update indices present in the batch
        uniq_idx = np.unique(X)
        gW_uniq = gW[uniq_idx] + self.l2 * self.W[uniq_idx]
        gV_uniq = gV[uniq_idx] + self.l2 * self.V[uniq_idx]
        
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        
        # Update W
        self.mW[uniq_idx] = b1 * self.mW[uniq_idx] + (1 - b1) * gW_uniq
        self.vW[uniq_idx] = b2 * self.vW[uniq_idx] + (1 - b2) * (gW_uniq ** 2)
        m_hat_w = self.mW[uniq_idx] / (1 - b1 ** self.t)
        v_hat_w = self.vW[uniq_idx] / (1 - b2 ** self.t)
        self.W[uniq_idx] -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + eps)
        
        # Update V
        self.mV[uniq_idx] = b1 * self.mV[uniq_idx] + (1 - b1) * gV_uniq
        self.vV[uniq_idx] = b2 * self.vV[uniq_idx] + (1 - b2) * (gV_uniq ** 2)
        m_hat_v = self.mV[uniq_idx] / (1 - b1 ** self.t)
        v_hat_v = self.vV[uniq_idx] / (1 - b2 ** self.t)
        self.V[uniq_idx] -= self.lr * m_hat_v / (np.sqrt(v_hat_v) + eps)
        
        # Update R
        self.mR = b1 * self.mR + (1 - b1) * gR
        self.vR = b2 * self.vR + (1 - b2) * (gR ** 2)
        m_hat_R = self.mR / (1 - b1 ** self.t)
        v_hat_R = self.vR / (1 - b2 ** self.t)
        self.R -= self.lr * m_hat_R / (np.sqrt(v_hat_R) + eps)
        
        # Update MLP parameters
        self.mW1 = b1 * self.mW1 + (1 - b1) * gW1
        self.vW1 = b2 * self.vW1 + (1 - b2) * (gW1 ** 2)
        m_hat = self.mW1 / (1 - b1 ** self.t)
        v_hat = self.vW1 / (1 - b2 ** self.t)
        self.W1 -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
        
        self.mb1 = b1 * self.mb1 + (1 - b1) * gb1
        self.vb1 = b2 * self.vb1 + (1 - b2) * (gb1 ** 2)
        m_hat = self.mb1 / (1 - b1 ** self.t)
        v_hat = self.vb1 / (1 - b2 ** self.t)
        self.b1 -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
        
        self.mW2 = b1 * self.mW2 + (1 - b1) * gW2
        self.vW2 = b2 * self.vW2 + (1 - b2) * (gW2 ** 2)
        m_hat = self.mW2 / (1 - b1 ** self.t)
        v_hat = self.vW2 / (1 - b2 ** self.t)
        self.W2 -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
        
        self.mb2 = b1 * self.mb2 + (1 - b1) * gb2
        self.vb2 = b2 * self.vb2 + (1 - b2) * (gb2 ** 2)
        m_hat = self.mb2 / (1 - b1 ** self.t)
        v_hat = self.vb2 / (1 - b2 ** self.t)
        self.b2 -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
        
        self.b -= self.lr * g.sum()
        return float(-np.mean(y * np.log(sigmoid(z) + 1e-9) + (1 - y) * np.log(1 - sigmoid(z) + 1e-9)))

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs], mask_w=None, mask_v=None)[0] for i in range(0, len(X), bs)])


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
            elif f in EXTRA_USER_COLS:
                out.append(eu.get(f, 'UNK'))
            elif f in EXTRA_VIDEO_COLS:
                out.append(ev.get(f, 'UNK'))
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
    """Loads train+valid only (test split is popped out of memory) and trains the DeepFM model."""
    print(f"loading {DATA_DIR} ...")
    splits = load(DATA_DIR)
    splits.pop("test", None)          # hard removal — test rows do not exist past this line
    enc, dim = build_encoding(splits, FIELDS, DATA_DIR)
    Xtr, ytr, _ = enc["train"]; Xva, yva, uva = enc["valid"]

    num_fields = len(FIELDS)
    m = FwFM(dim, num_fields, k=k, lr=lr, seed=seed, hidden_dim=32)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr)); t0 = time.time()
        losses = [m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]], drop_rate=0.1) for i in range(0, len(idx), bs)]
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            best_state = (m.V.copy(), m.W.copy(), m.R.copy(), np.float32(m.b),
                          m.W1.copy(), m.b1.copy(), m.W2.copy(), m.b2.copy())
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    m.V, m.W, m.R, m.b, m.W1, m.b1, m.W2, m.b2 = best_state
    r = evaluate(uva, yva, m.predict(Xva))
    return r["GAUC"], r["nDCG@5"], r["primary"]


def main():
    gauc, ndcg5, primary = train_and_score()
    # evaluate.py returns numpy float32 scalars — json.dump can't serialize those
    # directly, so cast to native Python floats before writing.
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump({"gauc": float(gauc), "ndcg5": float(ndcg5), "primary": float(primary)}, f)
    print(f"primary={primary:.4f} gauc={gauc:.4f} ndcg5={ndcg5:.4f}")


if __name__ == "__main__":
    main()