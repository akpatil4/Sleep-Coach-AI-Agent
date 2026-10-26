# scratch/demo_offline.py
# Runs your real pipeline and prints each step.
# The model here is an OFFLINE STAND-IN, not Nightshift, so don't present its text as model output.
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)                      # run from the project root, wherever you start this
sys.path.insert(0, str(ROOT))

import pandas as pd
from app import tools
from app.agents import run_agents
from app.engines.hmem import HMem
from app.evidence import before_after_with_ci
from app.experiments import rank_experiments
from app.safety import check_red_flags


def section(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


def show(obj):
    print(json.dumps(obj, indent=2, default=str))


class StandInModel:
    """Fake model for offline demos. It writes a template report from the evidence it is
    given. With inject_error=True its first draft includes an invented number (93.7),
    so you can watch the Evaluator catch it."""

    def __init__(self, inject_error=False):
        self.inject_error = inject_error

    def __call__(self, messages, temperature=0.2):
        system, user = messages[0]["content"], messages[1]["content"]
        if "Planner" in system:
            plan = {"tools": ["sleep_stats", "sleep_debt", "rank_experiments"],
                    "focus": "find the habit most linked with worse sleep"}
            return {"content": json.dumps(plan), "model": "stand-in (not a real model)"}
        ev = json.loads(user.split("EVIDENCE:\n")[1])
        parts = []
        s = ev.get("sleep_stats")
        if s:
            parts.append(f"Over {s['nights']} nights, average sleep was {s['avg_duration_h']} "
                         f"hours and average quality was {s['avg_quality']}.")
        chk = ev.get("experiment_check")
        if chk:
            ba = chk["before_after"]
            parts.append(f"In the experiment, quality changed by {ba['change']} "
                         f"(confidence interval {ba['ci95'][0]} to {ba['ci95'][1]}): "
                         f"{ba['label']}.")
        ranked = ev.get("ranking", {}).get("ranked", [])
        if ranked:
            t = ranked[0]
            parts.append(f"The habit most linked with worse sleep is {t['habit']} "
                         f"(r = {t['r']}, p = {t['p_value']}), which is {t['label']}.")
            parts.append(f"Suggested experiment: {t['experiment']}. "
                         "This is an association, not proof of cause.")
        text = " ".join(parts)
        if self.inject_error and "Fix these issues" not in user:
            text += " Average quality across all nights was 93.7."   # invented on purpose
        return {"content": text, "model": "stand-in (not a real model)"}


def memory(run_id):
    return HMem(store_path="scratch/demo_memory.json",
                ledger_path="scratch/demo_ledger.jsonl", run_id=run_id)


def main():
    for p in ["scratch/demo_memory.json", "scratch/demo_ledger.jsonl"]:
        if os.path.exists(p):
            os.remove(p)                      # every demo starts with empty memory

    df = tools.prepare(pd.read_csv("data/daily_log.csv"))

    section("1. INPUT DATA: last 8 of the synthetic nights (the last 7 simulate a "
            "'no screens' experiment)")
    cols = ["date", "duration_h", "sleep_quality", "caffeine_cups", "screen_min", "mood"]
    print(df[cols].tail(8).round(2).to_string(index=False))

    section("2. TOOLS: plain pandas maths, no AI involved")
    print("sleep_stats:");  show(tools.sleep_stats(df))
    print("sleep_debt:");   show(tools.sleep_debt(df))

    section("3. WHICH HABIT TO TEST NEXT? (correlation + permutation-test p-value)")
    show(rank_experiments(df))

    section("4. DID THE EXPERIMENT HELP? (before vs after, bootstrap 95% interval)")
    show(before_after_with_ci(df, "2026-09-29"))

    section("5. SAFETY GATE: runs before any coaching")
    for q in ["Why was my sleep bad this week?", "I get chest pain at night",
              "what melatonin dose should I take"]:
        print(f"  {q!r:45} -> escalate = {check_red_flags(q)['escalate']}")

    section("6. SESSION 1: Planner -> Executor -> Evaluator (memory empty)")
    q1 = "Why was my sleep bad this week?"
    out1 = run_agents(q1, {}, df, memory("demo-1"), "demo-trace-1",
                      llm=StandInModel(inject_error=True))
    print(out1["result"]["report"])
    print("\nstatus:", out1["status"])
    print("revisions:", out1["collaboration"]["revision_count"])
    print("Evaluator findings:")
    for f in out1["collaboration"]["evaluator_findings"]:
        print("   -", f)
    print("memories recalled:", out1["memory"]["retrieved"])

    section("7. WHAT H-MEM NOW HOLDS (saved to disk)")
    for rec in memory("viewer").memories.values():
        print(f"  [{rec['layer']}] {rec['content']}")

    section("8. SESSION 2: the experiment ran. Does the agent remember?")
    q2 = "Did cutting screen time help?"
    ctx = {"experiment_start": "2026-09-29", "experiment_habit": "screen_min"}
    mem2 = memory("demo-2")
    print("Memories the Planner will recall for this question:")
    for m in mem2.retrieve(q2)["memories"]:
        print(f"  (score {m['score']}) {m['content']}")
    out2 = run_agents(q2, ctx, df, mem2, "demo-trace-2", llm=StandInModel())
    print("\n" + out2["result"]["report"])

    section("9. SESSION 3: what should I try next? (screen_min is now 'tested')")
    out3 = run_agents("What should I try next?", {}, df, memory("demo-3"),
                      "demo-trace-3", llm=StandInModel())
    ranked = out3["result"]["evidence"]["ranking"]["ranked"]
    print("Next experiment:", ranked[0]["experiment"])
    print("Skipped:", out3["result"]["evidence"]["ranking"]["skipped"])

    section("10. AUDIT TRAIL for session 1 (logs/demo-trace-1.jsonl)")
    for line in open("logs/demo-trace-1.jsonl"):
        e = json.loads(line)
        print(f"  {e['agent_name']:<15} {e['action']:<18} {e['status']:<19} "
              f"{e['output_summary'][:70]}")


if __name__ == "__main__":
    main()