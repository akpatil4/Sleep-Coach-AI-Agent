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
              "ddmin", set()).draft("Why was my sleep bad this week?", {"memories": []},
                                     evidence, {"focus": "find the habit most linked with worse sleep"})
text = _prompt_first(captured[0])[0]["content"]
units = [u for u in text.split("\n") if u.strip()]
cache, calls = {}, 0


def fails(items, sep):
    global calls
    if not items:
        return False
    key = (tuple(items), sep)
    if key in cache:
        return cache[key]
    body = sep.join(items)
    calls += 1
    start = time.time()
    try:
        get_client().chat.completions.create(
            model=os.getenv("METERLESS_MODEL", "meterless/auto"),
            messages=[{"role": "user", "content": body}], temperature=0.2, max_tokens=4)
        result = False
    except Exception as exc:
        if getattr(exc, "status_code", None) == 502:
            result = True
        else:
            print("Unexpected error:", type(exc).__name__, str(exc)[:150])
            raise
    print(f"call {calls}: {len(items)} pieces, {len(body)} chars -> "
          f"{'REJECTED' if result else 'accepted'} ({time.time() - start:.0f}s)", flush=True)
    cache[key] = result
    return result


def ddmin(items, sep):
    n = 2
    while len(items) >= 2:
        size = max(1, len(items) // n)
        parts = [items[i:i + size] for i in range(0, len(items), size)]
        reduced = False
        for part in parts:
            if fails(part, sep):
                items, n, reduced = part, 2, True
                break
        if not reduced:
            for i in range(len(parts)):
                rest = [x for j, p in enumerate(parts) if j != i for x in p]
                if rest and fails(rest, sep):
                    items, n, reduced = rest, max(n - 1, 2), True
                    break
        if not reduced:
            if n >= len(items):
                break
            n = min(len(items), n * 2)
    return items


print(f"Full prompt: {len(units)} lines, {len(text)} chars", flush=True)
if not fails(units, "\n"):
    sys.exit("The full prompt was ACCEPTED this time. Run real_demo.py again.")
core = ddmin(units, "\n")
print("\nSmallest set of lines that still gets rejected:")
for line in core:
    print("  >", line)
if len(core) == 1 and " " in core[0]:
    words = ddmin(core[0].split(" "), " ")
    print("\nSmallest set of WORDS that still gets rejected:")
    print("  >", " ".join(words))
