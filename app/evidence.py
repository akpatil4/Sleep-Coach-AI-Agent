# Puts a number on "how sure are we?" so the agent never overclaims.
import numpy as np

MIN_NIGHTS = 14


def corr_with_uncertainty(df, habit, outcome="sleep_quality", n_perm=2000, seed=0):
    d = df[[habit, outcome]].dropna()
    x = d[habit].to_numpy(dtype=float)
    y = d[outcome].to_numpy(dtype=float)
    if len(d) < MIN_NIGHTS or x.std() == 0 or y.std() == 0:
        return {"habit": habit, "outcome": outcome, "n": len(d),
                "r": None, "p_value": None, "label": "insufficient data"}
    r = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    perm_r = np.array([np.corrcoef(rng.permutation(x), y)[0, 1]
                       for _ in range(n_perm)])
    p = float((np.sum(np.abs(perm_r) >= abs(r)) + 1) / (n_perm + 1))
    if p < 0.01:
        label = "strong evidence of an association"
    elif p < 0.05:
        label = "moderate evidence of an association"
    else:
        label = "could be noise"
    return {"habit": habit, "outcome": outcome, "n": len(d), "r": round(r, 3),
            "p_value": round(p, 4), "label": label,
            "note": "Association only. Not proof of cause."}


def before_after_with_ci(df, split_date, metric="sleep_quality", n_boot=2000, seed=0):
    before = df.loc[df["date"] < split_date, metric].dropna().to_numpy(dtype=float)
    after = df.loc[df["date"] >= split_date, metric].dropna().to_numpy(dtype=float)
    if len(before) < 7 or len(after) < 3:
        return {"metric": metric, "label": "insufficient data",
                "n_before": len(before), "n_after": len(after)}
    rng = np.random.default_rng(seed)
    diffs = np.array([rng.choice(after, len(after)).mean()
                      - rng.choice(before, len(before)).mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    change = float(after.mean() - before.mean())
    if lo > 0 or hi < 0:
        label = "change unlikely to be noise alone (still not proof the habit caused it)"
    else:
        label = "inconclusive: the interval includes zero"
    return {"metric": metric, "n_before": len(before), "n_after": len(after),
            "change": round(change, 3),
            "ci95": [round(float(lo), 3), round(float(hi), 3)], "label": label,
            "caution": ("Fewer than 14 nights after the change: weak evidence."
                        if len(after) < 14 else "Reasonable sample size."),
            "note": "Uncontrolled before/after; other things changed too."}
