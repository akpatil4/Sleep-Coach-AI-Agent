# Ranks candidate experiments by evidence strength, skipping habits already tested.
from app.evidence import corr_with_uncertainty

CANDIDATES = {
    "screen_min": "No screens in the last 60 minutes before bed",
    "last_caffeine_hour": "No caffeine after 2pm",
    "caffeine_cups": "Cap caffeine at 2 cups a day",
    "bed_hour": "Go to bed 30 minutes earlier",
}


def rank_experiments(df, tested=()):
    ranked, skipped = [], []
    for habit, experiment in CANDIDATES.items():
        if habit in tested:
            skipped.append({"habit": habit, "reason": "already tested"})
            continue
        ev = corr_with_uncertainty(df, habit)
        if ev["r"] is None:
            skipped.append({"habit": habit, "reason": "insufficient data"})
            continue
        if ev["r"] >= 0:
            skipped.append({"habit": habit,
                            "reason": "no sign that less of this hurts sleep"})
            continue
        score = abs(ev["r"]) * (1 - ev["p_value"])
        ranked.append({"habit": habit, "experiment": experiment, "r": ev["r"],
                       "p_value": ev["p_value"], "label": ev["label"],
                       "score": round(score, 3)})
    ranked.sort(key=lambda row: row["score"], reverse=True)
    return {"ranked": ranked, "skipped": skipped,
            "note": ("We test several habits, so one 'moderate' result could be luck. "
                     "Habits can also be linked (e.g. screen time and bedtime), so "
                     "the top habit may be standing in for another.")}
