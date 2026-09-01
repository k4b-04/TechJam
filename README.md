# Autonomous ML Research Agent for Recommender Systems

## Overview

Machine learning engineers often spend a large part of the model-development process repeating the same cycle: inspect the current solution, propose an improvement, modify the code, train the model, evaluate the result, and decide what to try next.

Our project automates this research loop.

We built an **Autonomous ML Research Agent** for the **KuaiRand-Pure recommender-system benchmark**. Starting from the organiser-provided Factorization Machine (FM) baseline, the agent can autonomously:

1. inspect the current best recommendation pipeline and previous experiment history;
2. propose one research hypothesis using an LLM;
3. generate a complete modified pipeline;
4. execute the candidate in an isolated working directory;
5. evaluate it using the official validation metrics;
6. keep or reject the change based on its score;
7. recover from failed candidates through a repair loop;
8. record the hypothesis, actual code diff, metrics, failures, resource usage and human interventions;
9. repeat until convergence, the iteration limit, or the wall-clock budget is reached.

The objective is therefore not only to produce a strong recommender model, but to demonstrate a **repeatable and robust autonomous ML experimentation process with minimal human intervention**.

---

## Challenge

The required benchmark is **KuaiRand-Pure**, derived from the KuaiRand short-video recommendation dataset.

This repository follows the task definition and evaluation implementation provided in the official starter kit:

| Item | Definition |
|---|---|
| Task | Within-user ranking over logged impressions |
| Target label | `long_view` (0/1) |
| Metrics | GAUC and nDCG@5 |
| Primary score | `(GAUC + nDCG@5) / 2` |
| Train split | 8 Apr 2022 – 21 Apr 2022 |
| Validation split | 22 Apr 2022 – 28 Apr 2022 |
| Test split | 29 Apr 2022 – 8 May 2022 |

During autonomous development, the agent is restricted to the **training and validation data only**. The candidate pipeline explicitly removes the test split from memory before training or evaluation.

### Official FM baseline

The organiser-provided Factorization Machine uses:

- `user_id`
- `video_id`
- `author_id`
- `tab`
- video-duration bucket

Reference scores:

| Split | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| Validation | 0.6674 | 0.5357 | **0.6016** |
| Published test baseline | 0.6610 | 0.5282 | **0.5946** |

All autonomous experiment decisions are made using the **validation primary score**, with `0.6016` used as the reference validation baseline.

---

## Our Approach

Our system separates the autonomous research workflow into several small modules with clearly defined interfaces.

                       ┌──────────────────────────┐
                       │         loop.py          │
                       │     Agent Orchestrator   │
                       └────────────┬─────────────┘
                                    │
                                    ▼
                       ┌──────────────────────────┐
                       │   llm.py + prompts.py    │
                       │ Hypothesis + code change │
                       └────────────┬─────────────┘
                                    │
                                    ▼
                          Candidate pipeline.py
                                    │
                                    ▼
                       ┌──────────────────────────┐
                       │        runner.py         │
                       │  Isolated execution +    │
                       │  timeout/error handling  │
                       └────────────┬─────────────┘
                                    │
                                    ▼
                       ┌──────────────────────────┐
                       │        metrics.py        │
                       │ Validate metrics.json    │
                       └────────────┬─────────────┘
                                    │
                              Keep / Reject
                                    │
                                    ▼
                       ┌──────────────────────────┐
                       │        logger.py         │
                       │ One JSONL entry / iter.  │
                       └────────────┬─────────────┘
                                    │
                                    ▼
                       ┌──────────────────────────┐
                       │      summarise.py        │
                       │ README-ready run report  │
                       └──────────────────────────┘


### 1. Research proposal — `agent/llm.py` and `agent/prompts.py`

The LLM receives:

- the current best pipeline;
- the best validation score so far;
- recent experiment history;
- previous hypotheses and results;
- the most recent traceback if the previous candidate failed.

It must return exactly one hypothesis and a complete modified Python pipeline.

The current implementation uses **Google Gemini through Google's OpenAI-compatible API endpoint**.

The system also tracks input and output token usage for every successful LLM call.

### 2. Orchestration — `agent/loop.py`

`loop.py` manages the entire autonomous experiment lifecycle.

For each iteration it:

1. requests one candidate from the LLM;
2. records the hypothesis and actual unified code diff;
3. sends the candidate to the runner;
4. reads its validation metrics;
5. compares the primary score against the current best;
6. keeps an improvement or reverts a worse candidate;
7. records the experiment in the run log;
8. adds the outcome to the history seen by future LLM calls.

A failed candidate never overwrites the current best pipeline.

### 3. Safe execution — `agent/runner.py`

Generated code is treated as untrusted input.

Before execution, the runner:

- checks the generated Python syntax;
- creates a separate working directory;
- writes the candidate to that directory;
- supplies the KuaiRand repository/data locations through environment variables;
- executes the candidate as a subprocess;
- captures stdout and stderr;
- enforces a timeout;
- converts syntax errors, crashes and timeouts into structured results instead of allowing them to crash the main agent.

### 4. Trusted evaluation — `agent/metrics.py`

Candidate pipelines must write:

```json
{
  "gauc": 0.0,
  "ndcg5": 0.0,
  "primary": 0.0
}
```

to `metrics.json`.

`metrics.py` is kept outside the LLM-editable candidate pipeline and validates the resulting values before they are trusted by the orchestrator.

This separation prevents a generated candidate from directly modifying the component responsible for deciding whether its score is accepted.

### 5. Logging — `agent/logger.py`

Every experiment appends one UTF-8 JSON object to:

```text
agent/runlog.jsonl
```

The run log captures fields including:

```text
timestamp
iteration
mode (proposal / repair)
hypothesis
actual code diff
GAUC
nDCG@5
primary score
delta against baseline
keep / reject decision
errors
recovery status
LLM input tokens
LLM output tokens
iteration duration
manual intervention
```

Logging is deliberately defensive: a logging failure should not terminate a long autonomous run.

### 6. Run reporting — `agent/summarise.py`

After a run, the log can be converted into a concise Markdown report:

```bash
python agent/summarise.py
```

The summary reports:

- total iterations logged;
- number of scored iterations;
- best validation primary;
- improvement over the official validation baseline;
- number of candidates retained;
- failures and their categories;
- successful recovery iterations;
- manual interventions;
- LLM input/output/total tokens;
- wall-clock runtime;
- GPU-hours.

Malformed JSONL lines are skipped rather than causing the reporting process to fail.

### 7. Final submission pipeline — `pipeline.py` and `make_submission.py` (repo root)

`agent/pipeline.py` is the file the autonomous loop is allowed to rewrite every iteration — it must stay reproducible against the organiser's 5-field, FM baseline (validation primary `0.6016`). The **final submission** is produced by a separate, frozen pair of files at the repo root, which are not touched by the agent loop:

- **`pipeline.py`** (root) — starts from the same FM baseline as `agent/pipeline.py`, plus additively-defined components that make up the submission architecture: out-of-fold CTR target encoding for `user_id`/`author_id` (`compute_target_encodings`), the CWM video columns (`music_id`, `video_type`, `upload_type`), and a hand-rolled **`DeepFM`** model — FM's order-1/order-2 terms plus a ReLU MLP tower over the same shared field embeddings, trained with the same from-scratch NumPy/Adam approach as the baseline `FM` class (no ML framework is used anywhere in this repo).
- **`make_submission.py`** (root) — run once, by hand, at the end. It is the one script explicitly allowed to touch the hidden test split: it retrains the submission architecture on train+valid, predicts on test, and writes `submission.csv` in the required `row_id,user_id,video_id,score` schema. It prefers a real `best_pipeline.py` if the agent loop ever produced one that beat the baseline (`agent/loop.py` writes this only when a kept candidate improves on `best_score`), and otherwise falls back to the DeepFM architecture in `pipeline.py`.

> Manually verified end-to-end against the real dataset: DeepFM reached **validation primary 0.6025** (vs. the `0.6016` FM baseline) before early-stopping at epoch 7, and `make_submission.py` produced a correctly-schemaed 170,588-row `submission.csv` in ~95 seconds. This was a **hand-designed architecture change**, not a result discovered by the autonomous loop — see [The autonomy trap](#the-autonomy-trap) below and the Results/Final hidden-test sections, which should report whatever the team's actual final run (autonomous loop and/or this script) produces.

---

## Autonomous Research Loop

At a high level:

```text
Current best pipeline
        │
        ▼
Read previous experiment history
        │
        ▼
LLM proposes ONE hypothesis
        │
        ▼
Generate complete candidate code
        │
        ▼
Syntax check + sandbox execution
        │
        ├──────────── failure ────────────┐
        │                                 │
        ▼                                 ▼
Read validation metrics             Record error
        │                                 │
        ▼                                 │
Compare with current best                 │
        │                                 │
   ┌────┴────┐                            │
   ▼         ▼                            │
 KEEP      REJECT                         │
   │         │                            │
   └────┬────┘                            │
        ▼                                 │
Log experiment                            │
        │                                 │
        └──── feed outcome / error ───────┘
                         │
                         ▼
                   Next iteration
```

If executable candidate code fails, the error is fed into the next LLM request and the prompt switches from **proposal mode** to **repair mode**.

The repair prompt asks the model to fix the failed implementation rather than introducing another unrelated research idea.

---

## Convergence

The organiser-defined convergence parameters are:

```text
epsilon = 0.002
N = 3
```

The current orchestration loop stops when the recent validation results do not produce a meaningful improvement above the earlier best score for the required number of iterations.

A run may also stop because:

- the maximum number of iterations is reached; or
- the wall-clock budget is exhausted.

---

# Setup

## Prerequisites

Recommended:

```text
Python 3.9+
Git
```

This project has been developed and tested across Windows (PowerShell), macOS, and Linux, all on Python 3.9+ — see the OS-specific commands throughout this section.

Clone the repository:

```bash
git clone https://github.com/k4b-04/TechJam.git
cd TechJam
```

Install the required Python dependencies:

```bash
python -m pip install numpy openai python-dotenv
```

No further dependencies are introduced anywhere in the frozen final pipeline (root `pipeline.py`, `make_submission.py`) — DeepFM and the out-of-fold target encoding are implemented from scratch with NumPy only, same as the rest of the repo.

---

## Dataset Setup

Download and extract **KuaiRand-Pure** into the repository root so the directory structure becomes:

```text
TechJam/
├── agent/
├── KuaiRand-Pure/
│   └── data/
│       ├── log_standard_4_08_to_4_21_pure.csv
│       ├── log_standard_4_22_to_5_08_pure.csv
│       ├── video_features_basic_pure.csv
│       └── ...
├── baseline.py
├── data.py
├── evaluate.py
└── ...
```

### Windows PowerShell

```powershell
curl.exe -L "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz" -o KuaiRand-Pure.tar.gz
tar -xzf KuaiRand-Pure.tar.gz
```

### macOS / Linux

```bash
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

Verify that this file exists:

```text
KuaiRand-Pure/data/video_features_basic_pure.csv
```

---

## Gemini API Setup

Create your own Gemini API key using Google AI Studio.

Then copy:

```text
agent/.env.example
```

to:

```text
agent/.env
```

### Windows PowerShell

```powershell
Copy-Item agent\.env.example agent\.env
```

### macOS / Linux

```bash
cp agent/.env.example agent/.env
```

For Gemini, configure:

```env
LLM_PROVIDER=openai
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
OPENAI_API_KEY=YOUR_OWN_GEMINI_API_KEY
LLM_MODEL=gemini-3.7-flash
```

If that model is temporarily unavailable, use another supported Gemini model listed in `agent/.env.example`.

### Important

Never commit `.env`.

Verify that Git ignores it:

```bash
git check-ignore agent/.env
```

The command should return:

```text
agent/.env
```

Do not print or share your API key.

---

# Reproducing the Official Baseline

Before running the autonomous agent, confirm that the starter-kit baseline works locally.

From the repository root:

```bash
python baseline.py --model fm
```

A correct setup should produce results near the organiser reference values:

```text
validation primary ≈ 0.6016
test primary       ≈ 0.5946
```

Small variation between seeds is expected.

---

# Running the Agent

## One-iteration smoke test

We recommend verifying the full system with one iteration first:

```bash
python agent/loop.py --max-iters 1 --budget-s 300
```

A successful run should perform:

```text
LLM proposal
→ candidate generation
→ candidate execution
→ validation scoring
→ keep/reject
→ logging
```

## Longer autonomous run

Example:

```bash
python agent/loop.py --max-iters 20 --budget-s 7200
```

Available arguments:

```text
--max-iters   maximum research iterations
--budget-s    overall wall-clock budget in seconds
--timeout-s   timeout for one candidate execution
```

Default values are currently:

```text
max iterations: 20
wall-clock budget: 7200 seconds
candidate timeout: 600 seconds
```

---

# Inspecting the Run Log

The raw experiment evidence is stored at:

```text
agent/runlog.jsonl
```

To inspect the latest iteration on Windows PowerShell without printing the full code diff:

```powershell
Get-Content agent\runlog.jsonl -Tail 1 |
ConvertFrom-Json |
Select-Object iter, hypothesis, primary, delta, kept, errors
```

For the complete aggregate report:

```bash
python agent/summarise.py
```

For a CPU-only run:

```bash
python agent/summarise.py --gpu-hours 0
```

---

# Results

### Final autonomous run

| Metric | Result |
|---|---:|
| Official validation FM baseline | **0.6016** |
| Best validation primary | **0.6037** |
| Delta vs validation baseline | **+0.0021** |
| Total iterations logged | **31** |
| Scored iterations | **19** |
| Candidate iterations kept | **6** |
| Failed iterations | **12** |
| Failures recovered on a later repair iteration | **0** |
| Manual interventions | **0** |
| LLM input tokens | **138,202** |
| LLM output tokens | **102,562** |
| Total LLM tokens | **240,764** |
| Total wall-clock time | **2h 45m 38s (9,938.3s)** |
| GPU-hours | **0.0000** |

Reproduce with:

```bash
python agent/summarise.py
```

### Failure / recovery breakdown

| Failure type | Count |
|---|---:|
| `llm` | 2 |
| `llm/rate_limit` | 4 |
| `run/nonzero_exit` | 2 |
| `run/timeout` | 4 |

Of the 31 logged iterations, 19 produced a scoreable candidate and 6 of those were kept as improvements (delta +0.0021 over baseline — modest, and worth reading against the measured seed-noise floor of 0.0008: this is a real but not large improvement). The remaining 12 failed outright rather than scoring; none were successfully repaired by a subsequent iteration in this run (see Robustness, below). Zero manual interventions occurred — every kept improvement came from the agent's own proposals.

The full per-iteration hypothesis/diff/score record for this run lives in `agent/runlog.jsonl`; that level of detail isn't reproduced here.

### Final hidden-test result

Pending the competition's official hidden-test evaluation — this table will be completed once the final submission is scored.

| Metric | Result |
|---|---:|
| GAUC | — |
| nDCG@5 | — |
| Primary | — |
| Delta vs published FM test baseline (0.5946) | — |

---

# Robustness

The system is designed around a **no-crash orchestration philosophy**.

Expected failures are treated as research events rather than fatal errors:

| Failure | Behaviour |
|---|---|
| LLM/API request failure | Retry with exponential backoff |
| Invalid generated Python | Reject before execution |
| Candidate runtime crash | Capture stderr and continue |
| Candidate timeout | Terminate candidate and log failure |
| Missing/malformed metrics | Reject candidate |
| Broken candidate implementation | Feed error into repair-mode prompt |
| Logging failure | Warn rather than terminate entire run |

During development, we observed real LLM service failures where Gemini returned temporary `503` availability errors. The LLM layer automatically retried those requests and was able to continue when the service recovered.

In the final run, 12 of 31 iterations failed (`llm`: 2, `llm/rate_limit`: 4, `run/nonzero_exit`: 2, `run/timeout`: 4) and the orchestration loop correctly kept `best_code` untouched and continued past every one of them without crashing — the no-crash design held for the entire 2h 45m run. However, none of those 12 failures were followed by a successful repair on the next iteration in this particular run (0 recoveries logged). The rate-limit and timeout failures in particular consumed iterations without the repair-mode prompt getting a chance to fix anything — see Known Limitations for what that implies about retry/backoff tuning and per-candidate runtime budgeting.

---

# Experiment Logging

Each logged iteration records:

```json
{
  "iter": 1,
  "mode": "propose",
  "hypothesis": "...",
  "code_diff": "...",
  "primary": 0.0,
  "gauc": 0.0,
  "ndcg5": 0.0,
  "delta": 0.0,
  "kept": false,
  "errors": [],
  "recovered": false,
  "tokens_in": 0,
  "tokens_out": 0,
  "duration_s": 0.0,
  "intervention": false
}
```

This log provides evidence of:

- what the agent decided to try;
- the actual code modification rather than only the stated intention;
- whether the experiment improved performance;
- whether the modification was kept or reverted;
- what errors occurred;
- whether the agent recovered;
- how much LLM compute was consumed;
- how much human intervention was required.

---

# Resource Usage

The competition evaluates not only model quality but also the cost of reaching the converged result.

We therefore measure:

- total LLM input tokens;
- total LLM output tokens;
- total LLM tokens;
- wall-clock runtime;
- GPU-hours.

Final run totals (same run as Results, above; source: `run_summary.json`):

| Resource | Total |
|---|---:|
| LLM input tokens | 138,202 |
| LLM output tokens | 102,562 |
| LLM total tokens | 240,764 |
| Wall-clock runtime | 2h 45m 38s (9,938.3s) |
| GPU-hours | 0.0000 (CPU-only throughout) |

---

# Repository Structure

```text
TechJam/
│
├── agent/
│   ├── .env.example
│   ├── contracts.py
│   ├── llm.py
│   ├── logger.py
│   ├── loop.py
│   ├── metrics.py
│   ├── pipeline.py
│   ├── prompts.py
│   ├── runner.py
│   ├── smoke_test.py
│   ├── summarise.py
│   ├── test_contracts.py
│   ├── test_llm_construction.py
│   ├── test_loop_dry_run.py
│   ├── test_parsing.py
│   └── test_runner.py
│
├── KuaiRand-Pure/
│   └── data/
│
├── ablation_features.py
├── baseline.py
├── baseline_scores.json
├── data.py
├── evaluate.py
├── submit.py
├── pipeline.py            # final-submission architecture (DeepFM + OOF target encoding)
├── make_submission.py     # run once, by hand: trains on train+valid, predicts test, writes submission.csv
├── submission.csv
├── README.md              # starter-kit spec (task definition, metrics, submission format — do not modify)
└── README_Owner_E.md      # this file — project overview, setup, results, contributions
```

### Core agent files

| File | Responsibility |
|---|---|
| `agent/loop.py` | Main autonomous orchestration loop |
| `agent/llm.py` | LLM provider interface, retries and token accounting |
| `agent/prompts.py` | Research and repair prompts |
| `agent/pipeline.py` | ML pipeline that the agent is allowed to modify |
| `agent/runner.py` | Safe candidate execution and timeouts |
| `agent/metrics.py` | Trusted validation-metric reader |
| `agent/logger.py` | Append one JSONL record per iteration |
| `agent/summarise.py` | Aggregate run evidence into Markdown |
| `agent/contracts.py` | Shared interfaces between modules |

### Starter-kit files

| File | Responsibility |
|---|---|
| `data.py` | Official dataset loading and split definitions |
| `evaluate.py` | Official evaluation implementation |
| `baseline.py` | Random, popularity and FM baselines |
| `baseline_scores.json` | Organiser reference scores |
| `submit.py` | Submission generation and validation |
| `ablation_features.py` | Organiser-provided feature-ablation experiment |

### Final submission files (repo root — separate from `agent/`, not agent-editable)

| File | Responsibility |
|---|---|
| `pipeline.py` | Frozen final-submission architecture: DeepFM (FM + deep MLP tower over shared embeddings) plus out-of-fold CTR target encoding and the CWM video columns. Additive on top of the same FM baseline `agent/pipeline.py` starts from — see "Final submission pipeline" under Our Approach, above. |
| `make_submission.py` | Run once, by hand, at the end. The only script allowed to touch the hidden test split: retrains on train+valid, predicts on test, writes `submission.csv` in the required schema. |
| `submission.csv` | Generated output — `row_id,user_id,video_id,score` for every test-split row. Regenerate with `python3 make_submission.py`. |

---

# Testing

Offline components can be tested without spending LLM tokens. Run these on a clean clone before final submission to catch any environment-specific assumption.

Examples:

```bash
python agent/test_contracts.py
python agent/test_parsing.py
python agent/test_runner.py
python agent/test_loop_dry_run.py
python agent/test_llm_construction.py
```

Live LLM connectivity can be tested with:

```bash
python agent/smoke_test.py
```

This makes a real API call and therefore requires a configured `.env`.

---

# Submission

The organiser requires:

```text
row_id,user_id,video_id,score
```

Before submitting the final CSV:

```bash
python submit.py --check --split test submission.csv
```

The submission checker verifies:

- the correct header;
- continuous `row_id`;
- correct number of rows;
- user/video alignment;
- numeric scores;
- absence of `NaN` and `Inf`.

The team's final submission is generated with:

```bash
python3 make_submission.py
python3 submit.py --check --split test submission.csv
```

`make_submission.py` (repo root) retrains the frozen DeepFM + out-of-fold target encoding architecture on train+valid and predicts on test — see Final submission pipeline, above.

---

# Known Limitations

Our current implementation has several limitations.

### 1. Dependence on an external LLM provider

Research proposals depend on API availability. During development we observed temporary Gemini `503` service-capacity failures. Retry logic allows transient failures to recover, but prolonged provider outages may consume a meaningful portion of the wall-clock budget.

**Future improvement:** automatically fall back to another configured model/provider after repeated service failures.

### 2. Experiment quality depends on prompt quality

The LLM can propose experiments that are technically valid but unpromising, redundant or already known to be weak.

We mitigate this through:

- explicit experiment history;
- organiser-known dead ends in the prompt;
- validation-based keep/reject decisions;
- repair mode after implementation failures.

**Future improvement:** introduce a structured experiment-planning stage that ranks several hypotheses before spending compute on one.

### 3. CPU-focused search space

The current system favours lightweight NumPy/CPU experiments to control resource consumption.

This makes the agent practical to run on a laptop, but restricts exploration of heavier neural recommender architectures.

**Future improvement:** selectively allow GPU-backed experiments when the expected gain justifies their resource cost.

### 4. Validation overfitting

Repeated autonomous experimentation against one fixed validation period risks adapting too closely to that split.

**Future improvement:** introduce additional validation strategies such as time-based sub-validation or the available random-exposure data while preserving the hidden-test boundary.

### 5. Repair mode did not recover any failure in the final run

Of the 12 failed iterations in the final run (`llm`: 2, `llm/rate_limit`: 4, `run/nonzero_exit`: 2, `run/timeout`: 4), none were followed by a successful repair on the next iteration (0 recoveries logged). Rate limits and timeouts together accounted for 8 of the 12 failures — both are plausibly better addressed by backoff/scheduling and stricter per-candidate runtime budgeting than by repair-mode prompting, since neither is a bug in the candidate code itself for the repair prompt to fix.

**Future improvement:** track failure type (`llm` vs `llm/rate_limit` vs `run/timeout` vs `run/nonzero_exit`) and only route genuine code failures to repair mode; back off and retry rate-limit/timeout failures instead of spending an LLM call on a repair prompt that can't address them.

### 6. The final run's seed pipeline was not the plain FM baseline

`agent/pipeline.py`'s own contract states the file must "keep the same 5 fields as the default — that's the acceptance test (must still reproduce valid primary 0.6016)." For this run, the seed file the loop started iterating from had already been replaced with the DeepFM + out-of-fold target encoding + CWM-column architecture (the same one documented under Final submission pipeline, above), not the plain 5-field FM baseline. That means the reported delta (+0.0021 over the 0.6016 FM baseline) reflects both this hand-designed starting point and whatever the agent changed on top of it across its 31 iterations — it is not a clean measurement of the agent's own autonomous contribution in isolation.

**Future improvement:** run the loop from the unmodified FM baseline (`agent/pipeline.py` as committed, 5 fields, `class FM`) to get a delta that isolates the agent's own discovery from any hand-designed starting architecture.

---

# Future Work

Potential extensions include:

- ranking-aligned objectives such as pairwise BPR or listwise losses — the agent proposed a BPR variant in this run, gated by prompts.py's sampled-negatives rule (rule 9) to keep it within the runtime budget;
- richer user-history / sequential-interest modelling;
- multi-task learning with auxiliary behavioural signals;
- watch-time modelling;
- temporal and distribution-shift-aware features;
- stronger architectures such as DCN or xDeepFM — DeepFM itself is already implemented as the frozen final-submission architecture (`pipeline.py`, root), separately from the autonomous loop;
- automatic LLM/provider fallback — not yet implemented; the final run's 4 `llm/rate_limit` failures are a concrete case this would have helped;
- smarter experiment prioritisation based on expected information gain;
- improved recovery classification and retry policies — see Known Limitations §5.

---

# Team Contributions

| Member | Contribution |
|---|---|
| **Owner A — Jadon Chan** | Main agent orchestration, experiment loop, convergence logic, keep/revert decisions and module integration (`agent/loop.py`, `agent/contracts.py`) |
| **Owner B — Zening Lim** | LLM interface, Gemini/API integration, prompt engineering, experiment-history formatting, repair prompting and token accounting (`agent/llm.py`, `agent/prompts.py`) |
| **Owner C — Kabir Durgani** | Candidate runner, isolated execution, timeout handling, syntax checks, failure capture, and prompt/target-encoding fixes (`agent/runner.py`, `agent/prompts.py`) |
| **Owner D — Dhevrath Malavar** | ML pipeline, feature encoding, validation-metric extraction, and the final submission pipeline — including the DeepFM + out-of-fold target encoding architecture and `make_submission.py` (`pipeline.py`, `make_submission.py`, `agent/pipeline.py`) |
| **Owner E — Neo Fu Jie** | Run-log schema and `logger.py`, run summarisation and resource reporting, README/Devpost documentation, clean-clone reproducibility testing and final demo preparation |

---

# Team Workflow — 72-Hour Execution Plan

To keep five people moving in parallel without blocking on each other, we planned the hackathon around hard **gates** rather than a soft schedule: each phase has one pass/fail check, and nobody moves to the next phase on hope alone.

## Day 0 — Get every unknown out of the way while it's free

**GATE:** every one of the five can run `python3 baseline.py --model fm` on their own laptop and see the baseline score.

| Owner | Task |
|---|---|
| All five | Attend the 28 Aug webinar with written questions ready. Download `kuairand-starter-kit.zip`; read `baseline.py`, `evaluate.py` and `submit.py` line by line together. |
| All five | Reproduce the baseline locally — everyone, individually. The single highest-value hour of the whole hackathon. |
| Kabir Durgani (C) | Get an LLM API key working and make one successful call from Python. That's the entire unknown for this workstream — kill it now, not on Day 1. |
| Jadon Chan (A) + Neo Fu Jie (E) | Agree the function signatures between the five modules and the JSON schema for one log entry. Write them as empty stubs and push, so nobody blocks. |
| Dhevrath Malavar (D) | Read the recsys primer in the appendix. Start `idea_library.md`: a list of things that plausibly improve an FM recommender, each with a one-line rationale. |
| Neo Fu Jie (E) | Set up the repo properly: branch protection off, everyone pushing to short-lived branches, one PR reviewer. Create the Devpost entry now, empty. |

## Hours 0–24 — End-to-end skeleton, however stupid

**GATE:** the agent completes one full iteration unattended — prompt → code → run → score → log. It does not need to improve anything.

| Owner | Task |
|---|---|
| Jadon Chan (A) | Loop skeleton with hardcoded stubs for every other module. Get the control flow right before anything real is plugged in. |
| Kabir Durgani (C) | First prompt template: problem summary + current code + "propose one change and return the full modified file." Code-block extraction. Token counter from the first call. |
| Zening Lim (B) | Subprocess runner with a hard timeout, stdout/stderr capture, and `ast.parse()` syntax check before execution. Never let bad model output crash the loop. |
| Dhevrath Malavar (D) | Wrap `evaluate.py` into a function returning a metrics dict. Confirm it matches the published validation numbers exactly before trusting anything downstream. |
| Neo Fu Jie (E) | Ship the logger: one JSON line per iteration with hypothesis, diff, metrics, errors, tokens. Then run the loop from a clean clone and file whatever breaks. |

> Do not chase score today. A team that has a working loop by hour 24 and a bad score is in a far better position than one with a great model and no agent.

## Hours 24–48 — Make it improve, and make it survive being left alone

**GATE:** an unattended 10-iteration run finishes without a human touching it, and the validation score beats the official baseline at least once.

| Owner | Task |
|---|---|
| Jadon Chan (A) | Keep-or-revert logic, best-checkpoint tracking, the convergence rule (ε = 0.002 over 3 iterations), and a wall-clock budget stop. Log why the run ended. |
| Zening Lim (B) | Rollback on failure, retry limits, and deliberate chaos testing: hand the runner deliberately broken code and confirm the loop recovers and logs the recovery event. |
| Kabir Durgani (C) | Feed history into the prompt so the agent stops re-proposing failed ideas — this is what separates a real agent from a random-code generator. Add the retry-with-traceback path. |
| Dhevrath Malavar (D) | Seed the idea library into the prompt as inspiration, not instruction. Independently re-score the agent's best checkpoint to confirm it isn't fooling itself. |
| Neo Fu Jie (E) | Draft the README and Devpost writeup now, from the logs already collected. Start the results table. Storyboard the video. |

### The autonomy trap

The moment you find yourself editing the model code by hand to push the score, stop. That is a manual intervention, it is counted, and it is worth more marks than the score gain. Put the idea into the agent's library and let the agent find it.

## Hours 48–72 — Freeze early, run long, spend the rest on evidence

**GATE:** code freeze at hour 54. Everything after that is the final run plus deliverables.

| Owner | Task |
|---|---|
| All five | Hour 48–54: last bug fixes only. No new features. Then freeze the branch. |
| Jadon Chan (A) + Zening Lim (B) | Hour 54: kick off the final long converged run and leave it alone. Every intervention from here gets written into the intervention count. |
| Dhevrath Malavar (D) | Validate the submission CSV with `python3 submit.py --check`. Do this early and repeatedly — a malformed header at hour 71 is how teams lose everything. |
| Neo Fu Jie (E) + Kabir Durgani (C) | Record and edit the 3-minute demo. Show a real iteration happening live, including a failure being recovered. Upload to YouTube as public and verify the link in an incognito window. |
| Neo Fu Jie (E) | Final README: overview, setup, reproduction steps, limitations, per-member contributions. Results table with the absolute delta over baseline, token total, and 0 GPU-hours. |
| All five | Hour 68: submit on Devpost. Not hour 71. Then use the remaining time to improve the writeup if the run is still going. |

---

# Development Tools, APIs and Libraries

### Development tools

- Git
- GitHub
- VS Code
- Claude Code (used for code review, debugging the final submission pipeline, and README preparation)

### API

- Google Gemini API through the OpenAI-compatible Gemini endpoint

### Python libraries

- NumPy
- `openai`
- `python-dotenv`
- Python standard library

This list is accurate as of the frozen final repository — no dependencies beyond these were introduced anywhere, including the DeepFM/target-encoding additions in the final submission pipeline.

### Dataset

- **KuaiRand-Pure**
- No external training data is used.

---

# Reproducibility Checklist

Before final submission, we verify the repository from a completely fresh clone:

- [ ] Clone repository into a new directory
- [ ] Install dependencies using README only
- [ ] Download/extract KuaiRand-Pure
- [ ] Reproduce official FM baseline
- [ ] Create local `.env`
- [ ] Confirm `.env` is ignored by Git
- [ ] Run offline tests
- [ ] Run LLM smoke test
- [ ] Complete one autonomous iteration
- [ ] Confirm `runlog.jsonl` is generated
- [ ] Confirm `summarise.py` works
- [ ] Reproduce final autonomous run using documented command
- [ ] Generate final submission
- [ ] Run `submit.py --check`
- [ ] Confirm final results/resource table matches the run logs

---

# Acknowledgements

This project builds on the organiser-provided KuaiRand-Pure starter kit and official evaluation implementation.

KuaiRand:
https://kuairand.com/

Methods referenced or implemented in this project:

- Guo, H., Tang, R., Ye, Y., Zhang, W., & He, X. (2017). *DeepFM: A Factorization-Machine based Neural Network for CTR Prediction.* IJCAI 2017. — architecture basis for the final submission's `DeepFM` class.
- Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). *BPR: Bayesian Personalized Ranking from Implicit Feedback.* UAI 2009. — pairwise ranking loss proposed and attempted by the autonomous agent during the final run.
