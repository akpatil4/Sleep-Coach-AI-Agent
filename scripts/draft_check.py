# Sends the Executor's reworded draft prompt to Nightshift (4-token replies only).
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
from app.evidence import before_after_with_ci
from app.experiments import rank_experiments
from app.llm import _prompt_first, get_client

df, _ = load_daily_log()
base = {"sleep_stats": tools.sleep_stats(df), "sleep_debt": tools.sleep_debt(df),
        "ranking": rank_experiments(df)}
with_exp = dict(base, experiment_check={
    "habit": "screen_min", "before_after": before_after_with_ci(df, "2026-09-29"),
    "period_means": tools.compare_periods(df, "2026-09-29")})
memory = {"memories": [{"content": "Proposed experiment on 2026-10-26: No screens in the last "
                                   "60 minutes before bed (habit screen_min, strong evidence "
                                   "of an association).", "confidence": 0.6}]}

cases = [("session 1 style", base, {"memories": []}, "find the habit most linked with worse sleep"),
         ("session 1, other focus", base, {"memories": []}, "analyse sleep duration and bedtime"),
         ("session 2 style (memory + experiment)", with_exp, memory, "check the experiment")]
for label, ev, mem, focus in cases:
    captured = []
    ExecutorAgent(lambda m, temperature=0.2: captured.append(m) or {"content": "x", "model": "c"},
                  "check", set()).draft("Why was my sleep bad this week?", mem, _clean(ev),
                                         {"focus": focus})
    text = _prompt_first(captured[0])[0]["content"]
    start = time.time()
    try:
        get_client().chat.completions.create(
            model=os.getenv("METERLESS_MODEL", "meterless/auto"),
            messages=[{"role": "user", "content": text}], temperature=0.2, max_tokens=4)
        print(f"[ok]     {label}: {len(text)} chars, {time.time() - start:.0f}s", flush=True)
    except Exception as exc:
        print(f"[FAILED] {label}: {len(text)} chars, {getattr(exc, 'status_code', '')}", flush=True)
