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

This project has been developed and tested on:

```text
[TODO: add OS/environment used for final demonstration]
```

Clone the repository:

```bash
git clone https://github.com/k4b-04/TechJam.git
cd TechJam
```

Install the required Python dependencies:

```bash
python -m pip install numpy openai python-dotenv
```

> [TODO: Confirm whether the final solution introduces any additional libraries and update this command accordingly.]

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

> **TODO: Replace this section using the FINAL CLEAN autonomous run only.**
>
> Do not use accumulated development/test logs.

### Final autonomous run

| Metric | Result |
|---|---:|
| Official validation FM baseline | **0.6016** |
| Best validation primary | **[TODO]** |
| Delta vs validation baseline | **[TODO]** |
| Total iterations logged | **[TODO]** |
| Scored iterations | **[TODO]** |
| Candidate iterations kept | **[TODO]** |
| Failed iterations | **[TODO]** |
| Failures recovered | **[TODO]** |
| Manual interventions | **[TODO]** |
| LLM input tokens | **[TODO]** |
| LLM output tokens | **[TODO]** |
| Total LLM tokens | **[TODO]** |
| Total wall-clock time | **[TODO]** |
| GPU-hours | **[TODO — 0.0000 if final run remains CPU-only]** |

Generate these values using:

```bash
python agent/summarise.py
```

### Experiment trajectory

> [TODO: Add a compact table of the final run's most important iterations.]

Example format:

| Iteration | Hypothesis | Primary | Delta | Decision |
|---:|---|---:|---:|---|
| 0 | Official FM baseline | 0.6016 | 0.0000 | Starting point |
| 1 | [TODO] | [TODO] | [TODO] | Keep / Reject |
| 2 | [TODO] | [TODO] | [TODO] | Keep / Reject |
| ... | ... | ... | ... | ... |
| Final | [TODO] | **[TODO]** | **[TODO]** | Final |

### Final hidden-test result

> [TODO: Fill only after the competition's final hidden-test evaluation/submission.]

| Metric | Result |
|---|---:|
| GAUC | [TODO] |
| nDCG@5 | [TODO] |
| Primary | [TODO] |
| Delta vs published FM test baseline (0.5946) | [TODO] |

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

> [TODO: Add the strongest failure → autonomous recovery example from the FINAL run, ideally including a candidate-code failure repaired by the next iteration.]

Example:

```text
Iteration X:
Generated candidate failed with [ERROR]

Iteration X+1:
Agent entered repair mode
→ fixed [CAUSE]
→ candidate executed successfully
→ validation primary = [SCORE]

Human intervention: 0
```

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

> [TODO: Insert `python agent/summarise.py` output from the final run here.]

If the final pipeline remains CPU-only:

```text
GPU-hours = 0
```

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
└── README.md
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

---

# Testing

Offline components can be tested without spending LLM tokens.

> [TODO: Verify all commands below on a clean clone before final submission.]

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

> [TODO: Update once the final model/submission pipeline is frozen.]

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

> [TODO: Add the exact command used to generate the team's FINAL submission from the selected best pipeline.]

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

### 5. [TODO: Add any limitation discovered during the final autonomous run]

---

# Future Work

Potential extensions include:

- ranking-aligned objectives such as pairwise BPR or listwise losses;
- richer user-history / sequential-interest modelling;
- multi-task learning with auxiliary behavioural signals;
- watch-time modelling;
- temporal and distribution-shift-aware features;
- stronger architectures such as DeepFM, DCN or xDeepFM;
- automatic LLM/provider fallback;
- smarter experiment prioritisation based on expected information gain;
- improved recovery classification and retry policies.

> [TODO: Remove items that the final agent already implemented successfully.]

---

# Team Contributions

> [TODO: Replace owner labels with names / GitHub handles before submission.]

| Member | Contribution |
|---|---|
| **Owner A — [NAME]** | Main agent orchestration, experiment loop, convergence logic, keep/revert decisions and module integration |
| **Owner B — [NAME]** | LLM interface, Gemini/API integration, prompt engineering, experiment-history formatting, repair prompting and token accounting |
| **Owner C — [NAME]** | Candidate runner, isolated execution, timeout handling, syntax checks and failure capture |
| **Owner D — [NAME]** | ML pipeline, feature encoding, validation-metric extraction and final submission pipeline |
| **Owner E — [NAME]** | Run-log schema and `logger.py`, run summarisation and resource reporting, README/Devpost documentation, clean-clone reproducibility testing and final demo preparation |

---

# Development Tools, APIs and Libraries

### Development tools

- Git
- GitHub
- VS Code
- [TODO: Add any other tools actually used in the final project]

### API

- Google Gemini API through the OpenAI-compatible Gemini endpoint

### Python libraries

- NumPy
- `openai`
- `python-dotenv`
- Python standard library

> [TODO: Add/remove dependencies based on the final frozen repository.]

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

[TODO: Add citations/links for any research papers, public implementations or external methods actually used by the final agent.]