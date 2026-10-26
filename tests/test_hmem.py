import json
from app.engines.hmem import HMem


def make(tmp_path, run_id="r1"):
    return HMem(store_path=tmp_path / "store.json",
                ledger_path=tmp_path / "ledger.jsonl", run_id=run_id)


def test_retrieval_ranks_relevant_first(tmp_path):
    m = make(tmp_path)
    m.add("Experiment: no screens after 10pm for 7 nights raised sleep quality",
          layer="working", tags=["experiment", "screen"], entities=["screen"])
    m.add("Caffeine cap of two cups was tested last month",
          layer="working", tags=["experiment", "caffeine"], entities=["caffeine"])
    out = m.retrieve("did cutting screen time before bed help my sleep quality")
    assert out["memories"][0]["content"].startswith("Experiment: no screens")


def test_unrelated_query_returns_nothing(tmp_path):
    m = make(tmp_path)
    m.add("No screens after 10pm", layer="working", tags=["screen"])
    assert m.retrieve("quarterly tax filing deadlines")["memories"] == []


def test_warm_start_after_end_run(tmp_path):
    m1 = make(tmp_path, "r1")
    m1.add("No caffeine after 2pm experiment started", tags=["experiment", "caffeine"])
    assert m1.end_run() == 1
    out = make(tmp_path, "r2").retrieve("caffeine experiment")
    assert len(out["memories"]) == 1


def test_short_term_not_persisted(tmp_path):
    make(tmp_path).add("temporary note about sleep")
    assert make(tmp_path, "r2").memories == {}


def test_superseded_ranks_lower(tmp_path):
    m = make(tmp_path)
    old = m.add("Screen cutoff experiment result: no effect on sleep",
                layer="working", tags=["screen"])
    m.add("Screen cutoff experiment result: improved sleep quality",
          layer="working", tags=["screen"], supersedes=old["id"])
    out = m.retrieve("screen cutoff experiment result sleep")
    assert "improved" in out["memories"][0]["content"]


def test_feedback_adjusts_confidence_and_logs(tmp_path):
    m = make(tmp_path)
    r = m.add("Late caffeine hurts sleep", layer="working", confidence=0.6)
    m.feedback(r["id"], "wrong")
    assert m.memories[r["id"]]["confidence"] < 0.5
    assert "needs-review" in m.memories[r["id"]]["tags"]
    assert "feedback_wrong" in (tmp_path / "ledger.jsonl").read_text()


def test_ledger_records_create_and_read(tmp_path):
    m = make(tmp_path)
    m.add("Screen time experiment", layer="working", tags=["screen"])
    m.retrieve("screen experiment")
    lines = (tmp_path / "ledger.jsonl").read_text().splitlines()
    actions = [json.loads(line)["action"] for line in lines]
    assert "create" in actions and "read" in actions
