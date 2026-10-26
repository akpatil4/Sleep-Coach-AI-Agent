# scripts/nightshift_check.py
# Sends one real request through Nightshift and prints which model answered.
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)   # the .env file wins over shell variables

from app.llm import chat, get_client

tok = os.getenv("METERLESS_RUNTIME_TOKEN") or ""
print("Base URL:", os.getenv("METERLESS_BASE_URL"))
print("Model setting:", os.getenv("METERLESS_MODEL"))
print("Token length:", len(tok), "| last 4 characters:", tok[-4:] if tok else "(empty)")
print("Token contains a space, quote or ellipsis:", any(c in tok for c in " \"'…"))

try:
    ids = [m.id for m in get_client().models.list().data]
    print("Models offered:", ids[:15])
except Exception as exc:
    print("Could not list models:", type(exc).__name__, "|", getattr(exc, "status_code", ""),
          "|", str(exc)[:300])

try:
    reply = chat([{"role": "user", "content": "Reply with exactly: Nightshift OK"}])
    print("Reply:", reply["content"])
    print("Model that handled it:", reply["model"])
except Exception as exc:
    print("CHAT FAILED:", type(exc).__name__, "| status:", getattr(exc, "status_code", ""))
    print("Message:", str(exc)[:500])