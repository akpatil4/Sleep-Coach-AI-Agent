# scripts/nightshift_probe.py
# Finds out why the draft call fails: tries prompts of growing length, then the exact
# prompt the Executor would send. Prints what worked and what failed.
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

from app import tools
from app.agents import ExecutorAgent, _clean
from app.data_source import load_daily_log
from app.experiments import rank_experiments
from app.llm import chat


def attempt(label, messages):
    chars = sum(len(m["content"]) for m in messages)
    start = time.time()
    try:
        reply = chat(messages)
        print(f"[ok]     {label}: {chars} chars, {time.time() - start:.0f}s, "
              f"model={reply['model']}, reply={reply['content'][:60]!r}")
    except Exception as exc:
        print(f"[FAILED] {label}: {chars} chars, {time.time() - start:.0f}s, "
              f"{type(exc).__name__} {getattr(exc, 'status_code', '')}: {str(exc)[:160]}")


attempt("tiny", [{"role": "user", "content": "Reply with exactly: OK"}])
for n in (1000, 3000, 6000, 12000):
    filler = ("Sleep data row. " * (n // 16)) + "\nReply with exactly: OK"
    attempt(f"padded ~{n}", [{"role": "user", "content": filler}])

# The exact messages the Executor would send for the first question.
df, source = load_daily_log()
evidence = _clean({"sleep_stats": tools.sleep_stats(df),
                   "sleep_debt": tools.sleep_debt(df),
                   "ranking": rank_experiments(df)})
captured = []


def capture(messages, temperature=0.2):
    captured.append(messages)
    return {"content": "x", "model": "capture"}


ExecutorAgent(capture, "probe-trace", set()).draft(
    "Why was my sleep bad this week?", {"memories": []}, evidence, {"focus": "probe"})
attempt("real draft prompt", captured[0])