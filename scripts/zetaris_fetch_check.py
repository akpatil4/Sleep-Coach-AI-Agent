# scripts/zetaris_fetch_check.py
# Prints what the agent will receive from Zetaris.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

from app.adapters.zetaris import fetch_daily_log

df = fetch_daily_log()
print(len(df), "rows from Zetaris")
print(df.head().to_string())
print(df.dtypes)