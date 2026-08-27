# Owner B — Model Interface

Your module is the only part of the system that talks to an LLM. Everything
else in the agent is ordinary Python.

## Files

| File | What it is |
|---|---|
| `llm.py` | The module. Owner A imports `propose_change` from here. |
| `prompts.py` | The prompt templates. This is what you tune on Day 2. |
| `test_parsing.py` | Offline tests. No API key needed. Run these first. |
| `smoke_test.py` | One real API call. Costs a fraction of a cent. |
| `.env.example` | Copy to `.env` and fill in your key. |

---

## Step 1 — Get a key (20 minutes, free)

**Ask the organisers first.** Hackathons usually hand out credits, and the
problem statement already mentions a 7-day free Trae trial from ByteDance.
Take this to the 28 Aug webinar before setting anything else up.

Failing that, these are free and need no credit card:

| Option | Get a key | Notes |
|---|---|---|
| **Google AI Studio (Gemini)** | aistudio.google.com/apikey | Usually the most generous free quota. Check yours at aistudio.google.com/rate-limit |
| **Groq** | console.groq.com/keys | 30 req/min, 1000 req/day, 8K tokens/min, **200K tokens/day**. Very fast. |
| **Ollama (local)** | ollama.com | Runs on your laptop. No key, no quota, works offline. Weaker at code. |

Every one of them speaks the OpenAI protocol, so `llm.py` doesn't change —
you only edit `.env`. See `.env.example` for a ready block per provider.

### Living inside a free quota

Groq's 200K tokens/day is the binding constraint. At ~10K tokens per
iteration that's about 20 iterations a day. Three things keep you under it:

1. **Don't send the whole pipeline file** if only one function is being
   edited. Send that function plus a summary of the rest.
2. **`format_history` already caps at the last 12 entries.** Lower it to 6 if
   you're tight.
3. **Each teammate uses their own key while developing.** Reserve one key,
   untouched, for the final converged run on Day 3.

This constraint is a feature, not a bug. Feasibility is 15% of the grade and
scored on burning *fewer* tokens — a free-tier run that finishes is a better
submission than an expensive one that scores the same.

### The Ollama option is worth a serious look

Running `qwen2.5-coder:7b` locally means zero API spend, zero quota, no
network dependency, and a resource report that reads "0 GPU-hours, $0 API
spend, runs entirely on a laptop." That's a genuinely strong Feasibility
story. The trade-off is that a 7B model writes worse code than a frontier
model, so you'll see more failed iterations — which your repair path handles,
and which look good in the log under Robustness.

Worth having as a fallback even if you use a hosted API primarily: if a quota
runs dry at hour 60, you switch one line in `.env` and keep going.

## Step 2 — Never commit the key

This is the one mistake with real consequences. A key pushed to a public repo
gets scraped by bots within minutes and someone else spends your money.

```bash
cp .env.example .env          # put your real key in .env
echo ".env" >> .gitignore     # do this BEFORE your first commit
git check-ignore .env         # should print ".env"
```

If you ever do leak one: revoke it in the console immediately. Don't delete
the commit and hope — it's already scraped.

## Step 3 — Install and verify

```bash
pip install openai             # covers Gemini, Groq, Ollama, ModelArk
pip install python-dotenv      # optional, loads .env automatically

python3 test_parsing.py        # expect: 23 passed, 0 failed
```

(`pip install anthropic` only if someone chooses the paid Anthropic route.)

Those tests need no key and no network. They cover the ways a model reply
actually breaks in practice — extra prose around the code, two code blocks,
a missing hypothesis line, a refusal with no code at all.

## Step 4 — One real call

```bash
# fill in .env first, then:
export $(grep -v '^#' .env | xargs)
python3 smoke_test.py
```

You should see a hypothesis, a chunk of generated Python, and a token count.
**When this passes, your Day 0 job is done** — you've killed the team's
biggest unknown.

---

## The contract with Owner A

Owner A calls exactly one function. Agree this signature with them today and
neither of you is blocked:

```python
result = propose_change(
    current_code = "<the pipeline file as a string>",
    history      = [ {...}, {...} ],   # past iterations
    last_error   = None,               # or a traceback string
    best_score   = 0.6012,             # or None
)
```

Returns a dict that **never raises**:

```python
{
  "hypothesis": str,          # one-line rationale, goes in the run log
  "code":       str | None,   # complete python file, or None if unusable
  "tokens_in":  int,
  "tokens_out": int,
  "raw":        str,          # untouched reply, log this too
  "ok":         bool,         # check this, not exceptions
  "error":      str | None,   # why it failed, if it did
}
```

The no-raise rule matters. Robustness is graded on how the agent handles a
failure, so a bad LLM response must return `ok: False` and let the loop
continue, never crash it.

---

## What you own on each day

**Day 0** — key working, `smoke_test.py` passing, contract agreed with A.

**Day 1** — plugged into A's loop. Token counter reporting from the very
first call. Don't tune prompts yet; just make it run.

**Day 2** — this is your real day. Two things:

1. **History in the prompt.** Already implemented in `format_history()`, but
   tune it. Without history the agent re-proposes the same failed idea
   forever and looks like a random code generator. With good history it looks
   like a researcher. This is the difference between a mid and a strong score
   on Innovation.

2. **The repair path.** When Owner C's runner returns a traceback, you pass it
   as `last_error` and the prompt switches to repair mode — fix the bug, don't
   invent. Test it deliberately: feed the loop broken code and watch it
   recover. That recovery, logged, is worth real marks under Robustness.

**Day 3** — freeze. Then produce the resource report:

```python
from llm import METER
print(METER.report())
# {'llm_calls': 63, 'failed_calls': 2, 'tokens_in': 412_004,
#  'tokens_out': 98_311, 'tokens_total': 510_315}
```

That goes straight into the README's resource table, next to `0 GPU-hours`.
Feasibility is 15% of the grade and it's scored on being *cheap*.

---

## Things that will go wrong, and what to do

**The model returns prose with no code.** Handled — `ok: False`,
`error: "no_code_block_in_response"`. The loop should just try again.

**The model returns a snippet instead of the full file.** That's why the
system prompt says "the complete modified file" twice and why `extract_code`
takes the longest block. If it still happens, tighten the prompt wording.

**Rate limits / 529 overloaded.** Handled — four retries with exponential
backoff. If it's persistent, drop to a smaller model or add a sleep between
iterations.

**Token usage climbing every iteration.** The history grows. `format_history`
caps at the last 12 entries for exactly this reason. If you're still burning
too much, shrink the pipeline file you send or summarise older entries.

**The model tries to peek at the test split.** The system prompt forbids it,
but don't rely on that alone — ask Owner D to make the test split physically
unreachable from the agent's code path. Prompt rules are a suggestion; a
missing file is a guarantee.
