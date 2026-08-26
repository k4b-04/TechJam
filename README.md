# KuaiRand-Pure Starter Kit

## Dependencies

Python 3.9+ and numpy. **Nothing else.** No torch, pandas, or sklearn required.

## Data

Download from https://kuairand.com (direct Zenodo link, no registration required):

```bash
# Run in the Starter Kit directory, after extraction you'll get ./KuaiRand-Pure/
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

## Running

```bash
python3 baseline.py --model fm
```

`--data_dir` defaults to `./KuaiRand-Pure/data`; specify explicitly if data is elsewhere.

`--model` options: `fm` (official baseline) / `pop` (trivial baseline) / `random` (lower bound for self-checking evaluation code).
FM takes about 40 seconds (CPU, single core).

## Task Definition (specifications are hardcoded, do not modify)

| | |
|---|---|
| Task | **Intra-user ranking** — Each user only ranks their impressions in the evaluation set, no full-catalog retrieval |
| Relevance label | `long_view` (native column, 0/1) |
| Metrics | `GAUC`, `nDCG@5`; **primary score = average of both** |
| Data split | train `20220408–20220421` / valid `20220422–20220428` / test `20220429–20220508` |
| Zero-positive users | nDCG recorded as 0.0 and included in average; GAUC only counts users with `0 < positive count < impression count`, weighted by positive count |
| nDCG gain | `2^rel − 1` (equivalent to identity for binary labels) |

See implementation in `evaluate.py`, all conventions are documented in the file header comments.

## Baseline Ladder

Scores on the test set. **FM is the one to beat.**

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| random (lower bound, for self-checking) | 0.4996 | 0.4511 | 0.4753 |
| item popularity (trivial) | 0.6308 | 0.5121 | 0.5715 |
| **FM (official baseline)** | **0.6610** | **0.5282** | **0.5946** |

### ⚠️ Real metric range: nDCG@5 ceiling is 0.729, not 1.0

Among the 23,875 users in the test set:

| | Proportion | Impact on metrics |
|---|---|---|
| All-negative users (all impressions are non-long_view) | **27.1%** | nDCG always **0**, no model can help; not counted in GAUC |
| All-positive users | **9.2%** | nDCG always **1**; not counted in GAUC |
| Users with discriminative power | **63.7%** | Actual samples for GAUC |

So using true labels as prediction scores (oracle, perfect ranking) can only achieve:

| | random | FM baseline | **oracle upper bound** | FM's covered range |
|---|---|---|---|---|
| GAUC | 0.4996 | 0.6610 | **1.0000** | 32.3% |
| nDCG@5 | 0.4511 | 0.5282 | **0.7289** | 27.8% |
| **primary** | 0.4753 | **0.5946** | **0.8645** | **30.7%** |

**Assess progress using oracle as denominator.** Seeing 0.5946 and thinking "far from perfect 1.0" is a misjudgment—
the baseline has already consumed 30% of the usable range, remaining headroom is 0.27, not 0.41.

FM's std across 5 random seeds is **0.0008**. Based on this, convergence criteria: **ε = 0.002 (≈2.5σ), N = 3**:
If validation primary score improvement doesn't exceed 0.002 for 3 consecutive iterations, declare convergence.

> Self-check: If your evaluation code running `--model random` doesn't get primary ≈ 0.475 (±0.001), your harness has issues—fix it first.

## Submission Format

CSV with header, one row per evaluation set row:

```
row_id,user_id,video_id,score
0,0,7531,-3.34176
1,0,4214,-1.4955
...
```

| Field | Description |
|---|---|
| `row_id` | Starting from 0, continuous increment, corresponding to row index in `data.load()[split]` (deterministic: read `log_standard_4_08_to_4_21_pure.csv` first, then `log_standard_4_22_to_5_08_pure.csv`, maintain original file order after date filtering) |
| `user_id` / `video_id` | Redundant fields, only for alignment verification |
| `score` | Your model's score for this row, any real number, only relative order matters; NaN/Inf not allowed |

> **Why row_id is mandatory:** `(user_id, video_id)` is **not unique** in the evaluation set—
> test set has 3.06% duplicate pairs, with up to 12 repetitions. So it cannot serve as primary key.

Generation and validation:

```bash
python3 submit.py --make  --split test  submission.csv    # Generate example submission using official FM baseline
python3 submit.py --check --split test  submission.csv    # Validate format and alignment
python3 submit.py --score --split valid submission.csv    # Validate and score (local valid available)
```

`--check` will reject: incorrect headers, wrong row count, non-consecutive `row_id`, `user_id`/`video_id` misalignment with evaluation set,
non-numeric `score` or NaN/Inf. **Run `--check` yourself before submission.**

## Where to Start Improving

The following ranking is **empirically tested**, not guessed. Dead ends already tried by organizers are marked—don't repeat them.

### Tested: These two have no benefit, don't waste iterations

| Tried | Result |
|---|---|
| **Adding static features** — Include all 13 feature fields from CWM (+`music_id`/`video_type`/`upload_type` + 6 user-side coarse bins) | primary **0.5940** vs 5-field **0.5950**, no difference within noise, even slightly worse |
| **Adding model capacity** — embedding dimension k = 8 / 16 / 32 | 0.5895 / 0.5902 / 0.5887, virtually unchanged |

Reason: The `user_id × video_id` interaction has already captured most learnable signals. Coarse bins like `follow_user_num_range`
are redundant in the presence of `user_id`; and 1.14 million rows can't support larger capacity. **Bottleneck is not in features and capacity.**

⚠️ Also note: **First-order terms of pure user-side features contribute exactly 0 to scores.** Because ranking is done within-user, any term
constant within a user doesn't change intra-group order (empirically verified: `item_pop × user bias` and pure `item_pop` yield identical scores).
User-side features only work through **interaction terms with item-side features**.

### Unexplored: Headroom should be here

Ranked by our judgment of likelihood (**these haven't been tested by organizers, left for you**):

1. **Change loss function.** Currently pointwise logloss, but metrics (GAUC/nDCG) are **ranking metrics**.
   Switch to pairwise (BPR) or listwise (softmax over user's impressions)—aligning objective with evaluation,
   this is what we consider most likely effective.
2. **User history sequences.** Current features **don't use behavioral sequences at all**. KuaiRand has hundreds to thousands of interactions per user in train,
   interest modeling like DIN/SIM is completely unexplored territory.
3. **Multi-task learning.** Logs also have `is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`, `play_time_ms`,
   can do multi-task auxiliary learning for `long_view` main task.
4. **Watch time modeling.** [CWM](https://github.com/hyz20/CWM)'s contribution is exactly this: it does **censored regression** on watch time
   (when video plays to completion, true watch time is truncated, so use one-sided loss instead of squared error). This is a research-deep direction.
5. **Change model.** DeepFM / DCN / xDeepFM. Given that capacity empirically isn't the bottleneck, **prioritize after 1-4**.
6. **Temporal features and distribution drift.** `hourmin`, `date`, and drift between train and test.
7. **Unbiased validation (advanced).** `log_random_4_22_to_5_08_pure.csv` is random exposure log (1.18 million rows),
   can serve as additional unbiased validation set to check if model overfits only on biased traffic.

## Using Your Own Model (including CWM)

`evaluate.py` is completely decoupled from models, it only needs three equal-length arrays:

```python
from evaluate import evaluate
print(evaluate(user_ids, labels, scores))   # scores can come from any model
```

- `user_ids`: user_id for each row in evaluation set
- `labels`: `long_view` (0/1) for that row
- `scores`: Your model's score for that row (any real number, only relative order matters)

So you can completely skip `baseline.py`, switch to PyTorch, LightGBM or [CWM](https://github.com/hyz20/CWM)'s xDeepFM,
as long as you pass `scores` to `evaluate()` in the end. **Scoring criteria is uniquely determined by `evaluate.py`.**

> Note for using CWM: It depends on `torch==1.6.0` (2020 version, probably won't install on new GPUs),
> and its loss optimizes counterfactual watch time, evaluation label is its own reconstructed `long_view2`.
> It's research code for a watch-time debiasing paper, can be used as **advanced reference**, not recommended as starting point.

## Files

| | |
|---|---|
| `evaluate.py` | Metric implementation + all specification conventions. **Do not modify.** |
| `data.py` | Data loading, official splits, feature encoding. Modify here for adding features. |
| `baseline.py` | Three baselines. FM is the one to beat. |
| `baseline_scores.json` | Official published scores + seed variance + convergence parameters. |
| `submit.py` | Generate / validate submission files. |
| `ablation_features.py` | Feature ablation experiments, can reproduce "adding features has no benefit" numbers. |