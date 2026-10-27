# scripts/save_examples.py
# Runs 3 real requests and saves the inputs, the real outputs and sample logs.
import json
import os
import shutil
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
from app.tracing import new_trace_id

EXAMPLES = [
    {"q": "Which habit should I test next to sleep better?", "ctx": {}},
    {"q": "Did cutting screen time help, and what should I try next?",
     "ctx": {"experiment_start": "2026-09-29", "experiment_habit": "screen_min"}},
    {"q": "Which habit should I try next, after the screen time experiment?", "ctx": {}},
]

for p in ["scratch/examples_memory.json", "scratch/examples_ledger.jsonl"]:
    if os.path.exists(p):
        os.remove(p)
Path("input_examples").mkdir(exist_ok=True)
Path("output_examples").mkdir(exist_ok=True)

df, source = load_daily_log()
print("Data source:", source, "|", len(df), "nights")

for i, ex in enumerate(EXAMPLES, 1):
    request = {"request_id": f"example-{i}", "user_query": ex["q"]}
    if ex["ctx"]:
        request["context"] = ex["ctx"]
    Path(f"input_examples/example_{i}.json").write_text(json.dumps(request, indent=2))

    trace_id = new_trace_id()
    memory = HMem(store_path="scratch/examples_memory.json",
                  ledger_path="scratch/examples_ledger.jsonl", run_id=trace_id)
    out = run_agents(ex["q"], ex["ctx"], df, memory, trace_id)
    result = {"request_id": request["request_id"], "trace_id": trace_id,
              "data_source": source, **out}
    Path(f"output_examples/example_{i}.json").write_text(json.dumps(result, indent=2, default=str))
    shutil.copy(f"logs/{trace_id}.jsonl", f"output_examples/example_{i}_trace.jsonl")
    print(f"example {i}: status={out['status']} revisions={out['collaboration']['revision_count']} "
          f"memories_recalled={out['memory']['retrieved']} models={out['models_used']}")

if os.path.exists("scratch/examples_ledger.jsonl"):
    shutil.copy("scratch/examples_ledger.jsonl", "output_examples/hmem_ledger_sample.jsonl")
print("saved to input_examples/ and output_examples/")