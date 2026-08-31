"""agent/make_submission.py — run ONCE, by hand, at the end.

NOT part of the agent's editable loop. pipeline.py must never touch the test
split; this script is the one place that's explicitly allowed to, per the
Day-1 rule ("Keep pipeline.py itself test-free... This is a separate script
we run once, by hand, at the end.").

Imports the winning architecture (DeepFM + out-of-fold target encoding +
CWM/video extra columns) from best_pipeline.py, retrains on train with the
same early-stopping-on-validation procedure the pipeline itself uses, then
predicts on the test split and writes submission.csv in the Starter Kit's
required schema (row_id,user_id,video_id,score).

One real gap this script has to fix: best_pipeline.py's compute_target_encodings()
only ever produces encodings for 'train'/'valid' — it has no branch for 'test' at
all, so calling its build_encoding() with a splits dict that includes 'test' would
KeyError. compute_target_encodings_with_test() below extends it: test rows get the
SAME full-train CTR map validation rows already use (never out-of-fold — test was
never part of any training fold to begin with, so this isn't a workaround, it's the
correct non-leaky choice).
"""
import csv
import os
import sys
import time

REPO_ROOT = os.environ.get(
    "KUAIRAND_REPO_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
DATA_DIR = os.environ.get("KUAIRAND_DATA_DIR", os.path.join(REPO_ROOT, "KuaiRand-Pure", "data"))
HERE = os.path.dirname(os.path.abspath(__file__))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from data import load                                                     # noqa: E402
from evaluate import evaluate                                              # noqa: E402
import numpy as np                                                          # noqa: E402

from best_pipeline import (                                                 # noqa: E402
    DeepFM, FIELDS, EXTRA_USER_COLS, EXTRA_VIDEO_COLS,
    _bucket_edges, _load_extra_features,
)


def compute_target_encodings_with_test(splits, m=20, n_buckets=10):
    """Same logic as best_pipeline.compute_target_encodings, extended to also
    produce a 'test' entry. See module docstring for why this is needed."""
    tr, va, te = splits['train'], splits['valid'], splits['test']

    y_tr = np.array([x[6] for x in tr], dtype=np.float32)
    global_mean = float(y_tr.mean())
    n_tr = len(tr)
    fold_ids = np.arange(n_tr) % 5

    out = {'train': {}, 'valid': {}, 'test': {}}

    for col_idx, col_name in [(1, 'user_ctr_bucket'), (3, 'author_ctr_bucket')]:
        raw_tr = [x[col_idx] for x in tr]
        raw_va = [x[col_idx] for x in va]
        raw_te = [x[col_idx] for x in te]

        vocab = {}
        for item in raw_tr:
            if item not in vocab:
                vocab[item] = len(vocab)
        n_vocab = len(vocab)

        idx_tr = np.array([vocab[x] for x in raw_tr], dtype=np.int32)
        idx_va = np.array([vocab.get(x, -1) for x in raw_va], dtype=np.int32)
        idx_te = np.array([vocab.get(x, -1) for x in raw_te], dtype=np.int32)

        oof_ctr = np.empty(n_tr, dtype=np.float32)
        for k in range(5):
            val_mask = (fold_ids == k)
            tr_mask = ~val_mask
            counts = np.bincount(idx_tr[tr_mask], minlength=n_vocab)
            sums = np.bincount(idx_tr[tr_mask], weights=y_tr[tr_mask], minlength=n_vocab)
            ctr_map = (sums + m * global_mean) / (counts + m)
            oof_ctr[val_mask] = ctr_map[idx_tr[val_mask]]

        full_counts = np.bincount(idx_tr, minlength=n_vocab)
        full_sums = np.bincount(idx_tr, weights=y_tr, minlength=n_vocab)
        full_ctr_map = np.where(full_counts > 0, (full_sums + m * global_mean) / (full_counts + m), global_mean)

        val_ctr = np.full(len(va), global_mean, dtype=np.float32)
        known_va = idx_va >= 0
        val_ctr[known_va] = full_ctr_map[idx_va[known_va]]

        test_ctr = np.full(len(te), global_mean, dtype=np.float32)
        known_te = idx_te >= 0
        test_ctr[known_te] = full_ctr_map[idx_te[known_te]]

        edges = np.unique(np.quantile(oof_ctr, np.linspace(0, 1, n_buckets + 1)[1:-1]))
        out['train'][col_name] = np.searchsorted(edges, oof_ctr).astype(str)
        out['valid'][col_name] = np.searchsorted(edges, val_ctr).astype(str)
        out['test'][col_name] = np.searchsorted(edges, test_ctr).astype(str)

    return out


def build_encoding_all(splits, fields, data_dir):
    """Same shape as best_pipeline.build_encoding, extended to cover 'test' too,
    and also returns raw video_id alongside user_id per row (needed for the
    submission schema, which best_pipeline's own encoder never had to return)."""
    tr = splits['train']
    edges = _bucket_edges([x[5] for x in tr])
    u_ext, v_ext = _load_extra_features(data_dir)
    target_data = compute_target_encodings_with_test(splits, m=20, n_buckets=10)

    def raw(x, idx, name):
        base = {
            'user_id': x[1], 'video_id': x[2], 'author_id': x[3], 'tab': x[4],
            'dur_bucket': str(int(np.searchsorted(edges, x[5]))),
            'user_ctr_bucket': target_data[name]['user_ctr_bucket'][idx],
            'author_ctr_bucket': target_data[name]['author_ctr_bucket'][idx],
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
                raise ValueError(f"unknown field {f!r}")
        return out

    vocabs = [dict() for _ in fields]
    for idx, x in enumerate(tr):
        for i, v in enumerate(raw(x, idx, 'train')):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    enc = {}
    for name, rws in splits.items():
        X = np.empty((len(rws), len(fields)), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users, videos = [], []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x, n, name)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            y[n] = x[6]
            users.append(x[1])
            videos.append(x[2])
        enc[name] = (X, y, users, videos)
    return enc, int(sum(field_dims))


def main():
    print(f"loading {DATA_DIR} (train+valid+test — deliberately NOT popping test) ...")
    splits = load(DATA_DIR)
    assert 'test' in splits, "expected a 'test' split from data.load()"

    enc, dim = build_encoding_all(splits, FIELDS, DATA_DIR)
    Xtr, ytr, _, _ = enc['train']
    Xva, yva, uva, _ = enc['valid']
    Xte, yte, ute, vte = enc['test']

    # Same hyperparameters and seed best_pipeline.py's default train_and_score()
    # call uses, so this reproduces the exact model that earned the kept score.
    seed, k, lr, epochs, bs, patience = 0, 16, 0.001, 15, 8192, 4

    m = DeepFM(dim, num_fields=len(FIELDS), k=k, hidden_dims=(64, 32), lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr)); t0 = time.time()
        losses = [m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]]) for i in range(0, len(idx), bs)]
        va = evaluate(uva, yva, m.predict(Xva))
        print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
              f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b),
                           m.W1.copy(), m.b1.copy(), m.W2.copy(), m.b2.copy(),
                           m.W3.copy(), np.float32(m.b3))
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop at epoch {ep}")
                break
    m.V, m.W, m.b, m.W1, m.b1, m.W2, m.b2, m.W3, m.b3 = best_state
    print(f"final validation primary used for early stop: {best:.4f}")

    scores = m.predict(Xte)

    out_path = os.path.join(HERE, "submission.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "user_id", "video_id", "score"])
        for row_id, (u, v, s) in enumerate(zip(ute, vte, scores)):
            w.writerow([row_id, u, v, float(s)])

    print(f"wrote {out_path} — {len(scores)} rows")


if __name__ == "__main__":
    main()
