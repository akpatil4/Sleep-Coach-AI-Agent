# scripts/real_demo.py
# Runs the real agents through Nightshift. No stand-in model. Memory is kept in scratch/.
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

from app.agents import run_agents
from app.data_source import load_daily_log
from app.engines.hmem import HMem


def memory(run_id):
    return HMem(store_path="scratch/real_memory.json",
                ledger_path="scratch/real_ledger.jsonl", run_id=run_id)


def show(title, out):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)
    print(out["result"]["report"])
    print("\nstatus:", out["status"], "| revisions:", out["collaboration"]["revision_count"])
    print("Evaluator findings:", out["collaboration"]["evaluator_findings"])
    print("memories recalled:", out["memory"]["retrieved"])
    print("models used:", out["models_used"])


for p in ["scratch/real_memory.json", "scratch/real_ledger.jsonl"]:
    if os.path.exists(p):
        os.remove(p)

df, source = load_daily_log()
print("Data source:", source, "|", len(df), "nights")

out1 = run_agents("Why was my sleep bad this week?", {}, df, memory("real-1"), "real-trace-1")
show("SESSION 1", out1)

ctx = {"experiment_start": "2026-09-29", "experiment_habit": "screen_min"}
out2 = run_agents("Did cutting screen time help?", ctx, df, memory("real-2"), "real-trace-2")
show("SESSION 2", out2)