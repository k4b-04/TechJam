"""
Owner B — Model Interface
=========================
The one job of this module: turn "here is the current pipeline and everything
we've tried" into "here is one new idea plus the code that implements it".

Owner A only ever calls ONE function from here:

    from llm import propose_change

    result = propose_change(current_code, history, last_error=None)
    # result = {
    #     "hypothesis": "Add author-level historical CTR as a feature",
    #     "code":       "<full modified python file>",
    #     "tokens_in":  6120,
    #     "tokens_out": 2011,
    #     "raw":        "<the model's untouched reply, for the log>",
    #     "ok":         True,
    #     "error":      None,
    # }

Everything else in this file is plumbing that Owner A never needs to know about.

Setup
-----
    pip install anthropic          # or: pip install openai
    export LLM_PROVIDER=anthropic  # or: openai
    export ANTHROPIC_API_KEY=sk-ant-...

See README_owner_b.md for the full walkthrough.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

from prompts import SYSTEM_PROMPT, build_user_prompt

# Load .env automatically if python-dotenv is installed. This means you never
# have to set environment variables by hand, on any operating system.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass   # fine — we'll read whatever is already in the environment


# --------------------------------------------------------------------------
# 1. Token meter — Feasibility is 15% of our grade and it is scored on how
#    FEW tokens we burn. So we count every single one from the first call.
# --------------------------------------------------------------------------

@dataclass
class TokenMeter:
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    failures: int = 0

    def add(self, tin: int, tout: int) -> None:
        self.calls += 1
        self.tokens_in += tin
        self.tokens_out += tout

    @property
    def total(self) -> int:
        return self.tokens_in + self.tokens_out

    def report(self) -> dict:
        """Drop this straight into the README's resource table at the end."""
        return {
            "llm_calls": self.calls,
            "failed_calls": self.failures,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_total": self.total,
        }


METER = TokenMeter()   # module-level, so the whole run shares one counter


# --------------------------------------------------------------------------
# 2. Provider adapters — we support Anthropic and anything OpenAI-compatible
#    (that includes BytePlus ModelArk, which speaks the OpenAI protocol).
#    Both return the same shape so nothing downstream cares which we used.
# --------------------------------------------------------------------------

DEFAULTS = {
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-4.1-mini",
}


class LLMError(RuntimeError):
    """Raised when the provider fails and retries are exhausted."""


@dataclass
class LLMClient:
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "anthropic"))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", ""))
    max_tokens: int = 24000
    temperature: float = 1.0
    max_retries: int = 4
    timeout: float = 240.0        # seconds — a stalled call must fail, not hang
    _client: object = field(default=None, repr=False)

    def __post_init__(self):
        self.provider = self.provider.lower().strip()
        if not self.model:
            self.model = DEFAULTS.get(self.provider, "")
        if not self.model:
            raise LLMError(f"No model set for provider {self.provider!r}. Set LLM_MODEL.")

        if self.provider == "anthropic":
            from anthropic import Anthropic          # imported lazily so the
            self._client = Anthropic(timeout=self.timeout)   # other SDK isn't required
        elif self.provider == "openai":
            from openai import OpenAI
            # base_url lets us point at Gemini / ModelArk / any OpenAI-compatible endpoint
            self._client = OpenAI(
                base_url=os.getenv("LLM_BASE_URL") or None,
                timeout=self.timeout,
                max_retries=0,        # we do our own retries, with logging
            )
        else:
            raise LLMError(f"Unknown LLM_PROVIDER {self.provider!r} (use anthropic or openai)")

        # Print the config once so a misconfigured run is obvious immediately
        # rather than after a silent five-minute hang. Never prints the key.
        print(f"  [llm] provider={self.provider} model={self.model} "
              f"base_url={os.getenv('LLM_BASE_URL') or 'default'} timeout={self.timeout}s")

    # -- the single network call, with retries -----------------------------

    def complete(self, system: str, user: str) -> tuple[str, int, int]:
        """Returns (text, tokens_in, tokens_out). Retries on transient errors."""
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                if self.provider == "anthropic":
                    r = self._client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        system=system,
                        messages=[{"role": "user", "content": user}],
                    )
                    text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                    tin, tout = r.usage.input_tokens, r.usage.output_tokens
                else:
                    r = self._client.chat.completions.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                    )
                    text = r.choices[0].message.content or ""
                    tin, tout = r.usage.prompt_tokens, r.usage.completion_tokens

                METER.add(tin, tout)
                return text, tin, tout

            except Exception as exc:                      # noqa: BLE001
                last_exc = exc
                METER.failures += 1
                wait = 2 ** attempt                       # 1s, 2s, 4s, 8s
                print(f"  [llm] attempt {attempt + 1}/{self.max_retries} failed "
                      f"({type(exc).__name__}) — retrying in {wait}s")
                time.sleep(wait)

        raise LLMError(f"All {self.max_retries} attempts failed: {last_exc}")


_CLIENT: LLMClient | None = None


def get_client() -> LLMClient:
    """Lazy singleton so we only construct (and only need a key) on first use."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = LLMClient()
    return _CLIENT


# --------------------------------------------------------------------------
# 3. Parsing — the model replies with prose. We need structured fields out
#    of it. These are pure functions, which means they are TESTABLE WITHOUT
#    AN API KEY. Run test_parsing.py to check them.
# --------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_HYPOTHESIS = re.compile(r"^\s*HYPOTHESIS\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)


def extract_code(text: str) -> str | None:
    """Pull the python file out of a fenced code block.

    If the model emits several blocks we take the LONGEST one — that's
    reliably the full file rather than a snippet it quoted while explaining.
    """
    blocks = _FENCE.findall(text or "")
    if not blocks:
        return None
    return max(blocks, key=len).strip() or None


def extract_hypothesis(text: str) -> str | None:
    """Pull the one-line rationale. This goes straight into the run log and
    is read by judges under 'Innovation & Problem Insight'."""
    m = _HYPOTHESIS.search(text or "")
    if m:
        return m.group(1).strip()
    # fallback: first non-empty line that isn't a code fence
    for line in (text or "").splitlines():
        line = line.strip()
        if line and not line.startswith("```"):
            return line[:300]
    return None


# --------------------------------------------------------------------------
# 4. History formatting — this is the single biggest quality lever in the
#    whole module. Without it the agent re-proposes the same failed idea
#    forever and looks like a random code generator instead of a researcher.
# --------------------------------------------------------------------------

def format_history(history: list[dict], max_entries: int = 12) -> str:
    """Turn the run log into a compact briefing the model can actually use.

    We keep it SHORT on purpose: every token here is billed on every call,
    and token count is a scored category. Recent entries matter most.
    """
    if not history:
        return "(nothing tried yet — this is the first iteration)"

    lines = []
    for h in history[-max_entries:]:
        verdict = "KEPT" if h.get("kept") else "rejected"
        score = h.get("primary")
        score_s = f"{score:.4f}" if isinstance(score, (int, float)) else "failed"
        delta = h.get("delta")
        delta_s = f" ({delta:+.4f})" if isinstance(delta, (int, float)) else ""
        lines.append(
            f"  #{h.get('iter', '?')}: {h.get('hypothesis', '(no hypothesis)')}\n"
            f"      -> {score_s}{delta_s}  [{verdict}]"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 5. THE PUBLIC FUNCTION — this is the entire contract with Owner A.
# --------------------------------------------------------------------------

def propose_change(
    current_code: str,
    history: list[dict],
    last_error: str | None = None,
    best_score: float | None = None,
) -> dict:
    """Ask the model for one improvement to the pipeline.

    Args:
        current_code: full text of the pipeline file as it currently stands.
        history:      list of past iteration dicts (see format_history).
        last_error:   if the previous attempt crashed, the traceback goes here.
                      The model is then asked to FIX rather than to innovate.
        best_score:   our best validation primary so far, for context.

    Returns a dict that never raises — check result["ok"] instead. The loop
    must survive a bad LLM response; that's what Robustness is graded on.
    """
    user_prompt = build_user_prompt(
        current_code=current_code,
        history_text=format_history(history),
        last_error=last_error,
        best_score=best_score,
    )

    base = {"hypothesis": None, "code": None, "tokens_in": 0,
            "tokens_out": 0, "raw": "", "ok": False, "error": None}

    try:
        text, tin, tout = get_client().complete(SYSTEM_PROMPT, user_prompt)
    except LLMError as exc:
        base["error"] = f"llm_call_failed: {exc}"
        return base

    code = extract_code(text)
    hypothesis = extract_hypothesis(text)

    base.update(raw=text, tokens_in=tin, tokens_out=tout,
                hypothesis=hypothesis, code=code)

    if code is None:
        base["error"] = ("response_truncated — raise max_tokens"
                         if "```" in text else "no_code_block_in_response")
        return base
        
    if hypothesis is None:
        base["hypothesis"] = "(model gave no hypothesis)"

    base["ok"] = True
    return base


# --------------------------------------------------------------------------

if __name__ == "__main__":
    # Tiny manual check: python3 llm.py
    print("Provider:", os.getenv("LLM_PROVIDER", "anthropic"))
    print("Key present:", bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")))
    print("Run test_parsing.py for offline tests, smoke_test.py for a live call.")
