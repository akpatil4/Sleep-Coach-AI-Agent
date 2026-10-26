# scripts/field_probe.py
# Tests each piece of the ranking text on its own to find what Nightshift rejects.
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)
os.environ["SAMPLE_MODE"] = "true"            # skip Zetaris: this test is about Nightshift

from app.data_source import load_daily_log
from app.experiments import rank_experiments
from app.llm import get_client

df, _ = load_daily_log()
ranking = rank_experiments(df)
top = ranking["ranked"][0]


def attempt(label, text):
    start = time.time()
    try:
        get_client().chat.completions.create(
            model=os.getenv("METERLESS_MODEL", "meterless/auto"),
            messages=[{"role": "user", "content": text + "\n\nReply: OK"}],
            temperature=0.2, max_tokens=4)
        print(f"[ok]     {label}", flush=True)
    except Exception as exc:
        print(f"[FAILED] {label} ({getattr(exc, 'status_code', '')})", flush=True)


for key, value in top.items():
    attempt(f"{key} = {value}", f"{key} = {value}")

attempt("negative number: r = -0.611", "r = -0.611")
attempt("same number positive: r = 0.611", "r = 0.611")
attempt("small decimal: p = 0.0005", "p = 0.0005")
attempt("p = 0.05", "p = 0.05")
attempt("underscore words: last_caffeine_hour", "habit = last_caffeine_hour")
attempt("note, first sentence", ranking["note"].split(".")[0] + ".")
attempt("note, whole", ranking["note"])
attempt("top item as one plain sentence",
        f"The habit {top['habit']} has a correlation of {top['r']} with sleep quality "
        f"(p-value {top['p_value']}).")