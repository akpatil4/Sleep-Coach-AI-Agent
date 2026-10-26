# Tests the real draft prompt: format changes, then removing one line at a time.
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)
os.environ["SAMPLE_MODE"] = "true"

from app import tools
from app.agents import ExecutorAgent, _clean
from app.data_source import load_daily_log
from app.experiments import rank_experiments
from app.llm import _prompt_first, get_client

df, _ = load_daily_log()
evidence = _clean({"sleep_stats": tools.sleep_stats(df), "sleep_debt": tools.sleep_debt(df),
                   "ranking": rank_experiments(df)})
captured = []
ExecutorAgent(lambda m, temperature=0.2: captured.append(m) or {"content": "x", "model": "c"},
              "tp", set()).draft("Why was my sleep bad this week?", {"memories": []},
                                  evidence, {"focus": "x"})
text = _prompt_first(captured[0])[0]["content"]
units = text.split("\n")


def rejected(body):
    try:
        get_client().chat.completions.create(
            model=os.getenv("METERLESS_MODEL", "meterless/auto"),
            messages=[{"role": "user", "content": body}], temperature=0.2, max_tokens=4)
        return False
    except Exception as exc:
        return True if getattr(exc, "status_code", None) == 502 else None


def show(label, body):
    start = time.time()
    r = rejected(body)
    word = {True: "REJECTED", False: "accepted", None: "other error"}[r]
    print(f"{word:9} {label} ({len(body)} chars, {time.time() - start:.0f}s)", flush=True)
    return r


print("=== format changes ===")
show("as is (the real prompt)", text)
show("newlines -> spaces", " ".join(u for u in units if u.strip()))
show("bullets and newlines removed", " ".join(u.lstrip("- ").strip() for u in units if u.strip()))
show("all lowercase", text.lower())
show("EVIDENCE -> data", text.replace("EVIDENCE", "data"))
show("no 'Rules:' list (first sentence + question + data)",
     "\n".join([units[0]] + [u for u in units if u.startswith(("Question", "No relevant", "Nights", "Sleep debt", "Habit", "Suggested"))]))

print("\n=== remove one line at a time from the real prompt ===")
flips = []
for i, line in enumerate(units):
    if not line.strip():
        continue
    r = show(f"without line {i}: {line[:55]!r}", "\n".join(units[:i] + units[i + 1:]))
    if r is False:
        flips.append((i, line))

print("\n=== SUMMARY ===")
if flips:
    print("Removing any ONE of these lines makes the prompt accepted:")
    for i, line in flips:
        print(f"  line {i}: {line}")
else:
    print("No single line removal fixes it. See the format changes above.")
