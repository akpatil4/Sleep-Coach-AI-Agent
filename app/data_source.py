# app/data_source.py
# Where the sleep table comes from.
# SAMPLE_MODE=true reads the local CSV (input data only, never answers).
# Otherwise the table comes from Zetaris.
import os
import pandas as pd
from app import tools


def load_daily_log():
    if os.getenv("SAMPLE_MODE", "true").lower() == "true":
        raw = pd.read_csv("data/daily_log.csv")
        source = "local_csv_sample_mode"
    else:
        from app.adapters.zetaris import fetch_daily_log
        raw = fetch_daily_log()
        source = "zetaris"
    return tools.prepare(raw), source