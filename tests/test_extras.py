import numpy as np
import pandas as pd
from app.evidence import corr_with_uncertainty, before_after_with_ci
from app.experiments import rank_experiments
from app.safety import check_red_flags


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
    })


def test_strong_signal_is_found():
    out = corr_with_uncertainty(make_df(), "screen_min")
    assert out["r"] < -0.5 and out["p_value"] < 0.01


def test_unrelated_habit_is_weak():
    out = corr_with_uncertainty(make_df(), "caffeine_cups")
    assert abs(out["r"]) < 0.4


def test_before_after_detects_a_real_shift():
    df = make_df()
    df.loc[df["date"] >= "2026-09-24", "sleep_quality"] += 2.0
    out = before_after_with_ci(df, "2026-09-24")
    assert out["ci95"][0] > 0 and "unlikely" in out["label"]


def test_ranker_skips_tested_habits():
    out = rank_experiments(make_df(), tested=["screen_min"])
    assert all(row["habit"] != "screen_min" for row in out["ranked"])


def test_safety_gate():
    assert check_red_flags("I get chest pain at night")["escalate"]
    assert check_red_flags("what melatonin dose should I take")["escalate"]
    assert not check_red_flags("why was my sleep bad this week")["escalate"]
