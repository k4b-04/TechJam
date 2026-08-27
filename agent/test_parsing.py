"""
Offline tests — NO API KEY NEEDED. Run these first:

    python3 test_parsing.py

Every one of these must pass before you plug llm.py into Owner A's loop.
They cover the parts that break in real runs: the model wrapping its answer
in prose, emitting two code blocks, forgetting the hypothesis line, or
returning nothing usable at all.
"""

from llm import extract_code, extract_hypothesis, format_history
from prompts import build_user_prompt

PASS, FAIL = 0, 0


def check(name, got, want):
    global PASS, FAIL
    if got == want:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}\n       got:  {got!r}\n       want: {want!r}")


def contains(name, haystack, needle):
    global PASS, FAIL
    if needle in haystack:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} — {needle!r} not found")


print("\nextract_code")
check("plain block",
      extract_code("```python\nx = 1\n```"), "x = 1")
check("no language tag",
      extract_code("```\nx = 1\n```"), "x = 1")
check("surrounded by prose",
      extract_code("Sure! Here you go:\n```python\nx = 1\n```\nHope that helps."), "x = 1")
check("picks the longest of two blocks",
      extract_code("```python\nold\n```\nand the full file:\n```python\nline1\nline2\nline3\n```"),
      "line1\nline2\nline3")
check("no block at all returns None",
      extract_code("I'm sorry, I can't help with that."), None)
check("empty block returns None",
      extract_code("```python\n\n```"), None)
check("handles None input",
      extract_code(None), None)

print("\nextract_hypothesis")
check("standard line",
      extract_hypothesis("HYPOTHESIS: add author CTR\n\n```python\nx=1\n```"),
      "add author CTR")
check("lowercase key",
      extract_hypothesis("hypothesis: add author CTR"), "add author CTR")
check("extra whitespace",
      extract_hypothesis("  HYPOTHESIS :   spaced out  \n"), "spaced out")
check("falls back to first prose line",
      extract_hypothesis("Adding a new feature field.\n```python\nx=1\n```"),
      "Adding a new feature field.")
check("handles None input",
      extract_hypothesis(None), None)

print("\nformat_history")
check("empty history",
      format_history([]), "(nothing tried yet — this is the first iteration)")
contains("shows kept verdict",
         format_history([{"iter": 1, "hypothesis": "k=32", "primary": 0.61, "kept": True}]),
         "KEPT")
contains("shows rejected verdict",
         format_history([{"iter": 1, "hypothesis": "k=32", "primary": 0.58, "kept": False}]),
         "rejected")
contains("crashed iteration doesn't explode",
         format_history([{"iter": 2, "hypothesis": "bad idea", "primary": None, "kept": False}]),
         "failed")
contains("shows delta when present",
         format_history([{"iter": 3, "hypothesis": "x", "primary": 0.61, "delta": 0.004, "kept": True}]),
         "+0.0040")
check("truncates to max_entries",
      len(format_history([{"iter": i, "hypothesis": "x", "primary": 0.6, "kept": True}
                          for i in range(50)], max_entries=3).splitlines()),
      6)  # 2 lines per entry

print("\nbuild_user_prompt")
contains("innovate mode includes history",
         build_user_prompt("CODE", "HISTORY"), "HISTORY")
contains("innovate mode includes best score",
         build_user_prompt("CODE", "HISTORY", best_score=0.6012), "0.6012")
contains("repair mode switches template",
         build_user_prompt("CODE", "HISTORY", last_error="NameError: x"), "repair only")
check("repair mode ignores history",
      "HISTORY" in build_user_prompt("CODE", "HISTORY", last_error="boom"), False)
contains("long traceback is truncated",
         build_user_prompt("CODE", "H", last_error="x" * 9000), "(truncated)")

print(f"\n{PASS} passed, {FAIL} failed\n")
raise SystemExit(1 if FAIL else 0)
