# Creates ONE table of 97 nights of synthetic sleep data (input data, not answers).
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
n = 97
dates = pd.date_range("2026-07-01", periods=n)

screen = rng.integers(0, 150, n)
screen[-7:] = rng.integers(0, 30, 7)      # last 7 nights: a "no screens" experiment

cups = rng.integers(0, 5, n)
last_caf = np.where(cups > 0, rng.integers(8, 20, n), np.nan)
late_caf = np.nan_to_num(last_caf, nan=0) >= 15

bed_hour = 23 + screen / 150 + rng.normal(0, 0.4, n)
weekend = np.asarray(dates.dayofweek >= 4)
wake_hour = 7 + weekend * 1.0 + rng.normal(0, 0.5, n)
bedtime = dates + pd.to_timedelta(bed_hour, unit="h")
wake_time = dates + pd.Timedelta(days=1) + pd.to_timedelta(wake_hour, unit="h")
duration = np.asarray((wake_time - bedtime).total_seconds()) / 3600

quality = np.clip(8 - 0.015 * screen - 1.0 * late_caf + rng.normal(0, 0.7, n), 1, 10)
mood = np.clip(5 + 0.5 * (duration - 7) + rng.normal(0, 1, n), 1, 10)

Path("data").mkdir(exist_ok=True)
pd.DataFrame({
    "date": dates.strftime("%Y-%m-%d"),
    "bedtime": bedtime, "wake_time": wake_time,
    "sleep_quality": quality.round(1),
    "caffeine_cups": cups, "last_caffeine_hour": last_caf,
    "screen_min": screen, "mood": mood.round(1),
}).to_csv("data/daily_log.csv", index=False)
print("wrote data/daily_log.csv")
