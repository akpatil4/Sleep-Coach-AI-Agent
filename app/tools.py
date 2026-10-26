# The Executor's toolbox: plain pandas functions that return numbers.
import pandas as pd


def prepare(df):
    df = df.copy()
    df["bedtime"] = pd.to_datetime(df["bedtime"])
    df["wake_time"] = pd.to_datetime(df["wake_time"])
    df["duration_h"] = (df["wake_time"] - df["bedtime"]).dt.total_seconds() / 3600
    h = df["bedtime"].dt.hour + df["bedtime"].dt.minute / 60
    df["bed_hour"] = ((h - 12) % 24) + 12      # noon-based clock: 00:30 -> 24.5
    return df.sort_values("date").reset_index(drop=True)


def sleep_stats(df):
    return {"nights": len(df),
            "avg_duration_h": round(df["duration_h"].mean(), 2),
            "bedtime_std_min": round(df["bed_hour"].std() * 60),
            "avg_quality": round(df["sleep_quality"].mean(), 2),
            "nights_under_7h": int((df["duration_h"] < 7).sum())}


def sleep_debt(df, target_h=8.0, last_n=7):
    recent = df.tail(last_n)
    debt = (target_h - recent["duration_h"]).clip(lower=0).sum()
    return {"target_h": target_h, "window_nights": len(recent),
            "debt_h": round(debt, 2)}


def correlations(df, outcome="sleep_quality"):
    cols = ["caffeine_cups", "last_caffeine_hour", "screen_min",
            "duration_h", "bed_hour", "mood", "sleep_quality"]
    c = df[cols].corr()[outcome].drop(outcome).dropna()
    c = c.reindex(c.abs().sort_values(ascending=False).index)
    return {"outcome": outcome, "correlations": c.round(2).to_dict(),
            "note": "Correlation is not causation."}


def compare_periods(df, split_date):
    before = df[df["date"] < split_date]
    after = df[df["date"] >= split_date]
    out = {"split_date": split_date, "n_before": len(before),
           "n_after": len(after), "metrics": {}}
    for col in ["sleep_quality", "duration_h", "screen_min", "mood"]:
        out["metrics"][col] = {
            "before": round(before[col].mean(), 2),
            "after": round(after[col].mean(), 2),
            "change": round(after[col].mean() - before[col].mean(), 2)}
    out["caution"] = ("Fewer than 14 nights after the change: weak evidence."
                      if len(after) < 14 else "Reasonable sample size.")
    return out
