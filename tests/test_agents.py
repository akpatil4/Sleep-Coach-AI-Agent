# Tests the agent loop with a FAKE model. This fake exists only in tests; the app
# itself always calls the real model through app/llm.py.
import json
import re
import numpy as np
import pandas as pd
from app.agents import run_agents
from app.engines.hmem import HMem

GOOD = "Screen time is linked with worse sleep. Experiment: {top}. This is an association, not proof."
BAD_NUM = "Average quality was 93.7. Experiment: {top}."
BAD_CAUSAL = "Screen time causes poor sleep. Experiment: {top}."


def make_df(n=90):
    rng = np.random.default_rng(1)
    screen = rng.integers(0, 150, n).astype(float)
    return pd.DataFrame({
        "date": pd.date_range("2026-07-01", periods=n).strftime("%Y-%m-%d"),
        "screen_min": screen,
        "caffeine_cups": rng.integers(0, 5, n).astype(float),
        "last_caffeine_hour": rng.integers(8, 20, n).astype(float),
        "bed_hour": 23 + screen / 150 + rng.normal(0, 0.4, n),
        "sleep_quality": 8 - 0.03 * screen + rng.normal(0, 0.5, n),
        "duration_h": 7 + rng.normal(0, 0.5, n),
        "mood": 5 + rng.normal(0, 1, n)})


class FakeLLM:
    def __init__(self, drafts):
        self.drafts = list(drafts)

    def __call__(self, messages, temperature=0.2):
        system, user = messages[0]["content"], messages[-1]["content"]
        if "Planner" in system:
            return {"content": json.dumps({"tools": ["sleep_stats", "sleep_debt",
                    "rank_experiments"], "focus": "test"}), "model": "fake-model"}
        body = user.split("EVIDENCE:\n")[-1]
        found = re.search(r"Suggested experiment \(quote exactly\): (.+)", body)
        top = found.group(1).strip() if found else ""
        text = (self.drafts.pop(0) if self.drafts else GOOD).replace("{top}", top)
        return {"content": text, "model": "fake-model"}


def make_hmem(tmp_path, run="r1"):
    return HMem(store_path=tmp_path / "store.json",
                ledger_path=tmp_path / "ledger.jsonl", run_id=run)


def test_revision_loop_fixes_unsupported_number(tmp_path):
    out = run_agents("Why was my sleep bad?", {}, make_df(), make_hmem(tmp_path), "t-1",
                     llm=FakeLLM([BAD_NUM, GOOD]))
    assert out["status"] == "success"
    assert out["collaboration"]["revision_count"] == 1
    assert any("93.7" in f for f in out["collaboration"]["evaluator_findings"])
    assert "not medical advice" in out["result"]["report"].lower()
    assert out["models_used"] == ["fake-model"]


def test_causal_wording_is_rejected(tmp_path):
    out = run_agents("Why was my sleep bad?", {}, make_df(), make_hmem(tmp_path), "t-2",
                     llm=FakeLLM([BAD_CAUSAL, GOOD]))
    assert out["status"] == "success"
    assert any("causes" in f for f in out["collaboration"]["evaluator_findings"])


def test_unapproved_draft_is_flagged_and_not_remembered(tmp_path):
    hmem = make_hmem(tmp_path)
    out = run_agents("Why was my sleep bad?", {}, make_df(), hmem, "t-3",
                     llm=FakeLLM([BAD_NUM] * 5))
    assert out["status"] == "needs_review"
    assert out["collaboration"]["revision_count"] == 2
    assert hmem.memories == {}


def test_proposal_is_remembered(tmp_path):
    hmem = make_hmem(tmp_path)
    run_agents("Why was my sleep bad?", {}, make_df(), hmem, "t-4", llm=FakeLLM([GOOD]))
    assert any("Proposed experiment" in r["content"] for r in hmem.memories.values())


def test_session_two_records_result_and_skips_tested_habit(tmp_path):
    ctx = {"experiment_start": "2026-09-20", "experiment_habit": "screen_min"}
    out = run_agents("Did cutting screens help?", ctx, make_df(),
                     make_hmem(tmp_path, "r1"), "t-5", llm=FakeLLM([GOOD]))
    assert out["status"] == "success"
    assert "experiment_check" in out["result"]["evidence"]
    again = run_agents("What should I try next?", {}, make_df(),
                       make_hmem(tmp_path, "r2"), "t-6", llm=FakeLLM([GOOD]))
    ranked = again["result"]["evidence"]["ranking"]["ranked"]
    assert all(row["habit"] != "screen_min" for row in ranked)

class Reject502(Exception):
    status_code = 502


def test_draft_retries_with_less_evidence_when_nightshift_rejects(tmp_path):
    inner = FakeLLM([GOOD])

    def picky(messages, temperature=0.2):
        if len(messages) == 2 and "Planner" not in messages[0]["content"] and "Nights analysed" in messages[-1]["content"]:
            raise Reject502("rejected")
        return inner(messages, temperature)

    out = run_agents("Why was my sleep bad?", {}, make_df(), make_hmem(tmp_path), "t-7", llm=picky)
    assert out["status"] == "success"
