# scripts/quick_probe.py
# Fast test: which variant of the Executor's draft prompt does Nightshift accept?
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)
os.environ["SAMPLE_MODE"] = "true"            # skip Zetaris: this test is about Nightshift

from app import tools
from app.agents import ExecutorAgent, _clean
from app.data_source import load_daily_log
from app.experiments import rank_experiments
from app.llm import _prompt_first, get_client

df, _ = load_daily_log()
evidence = _clean({"sleep_stats": tools.sleep_stats(df), "sleep_debt": tools.sleep_debt(df),
                   "ranking": rank_experiments(df)})
captured = []
ExecutorAgent(lambda m, temperature=0.2: captured.append(m) or {"content": "x", "model": "c"},
              "quick", set()).draft("Why was my sleep bad this week?", {"memories": []},
                                     evidence, {"focus": "probe"})
messages = captured[0]
folded = _prompt_first(messages)
rules, user = messages[0]["content"], messages[1]["content"]
evidence_json = user.split("EVIDENCE:\n")[1]
default_model = os.getenv("METERLESS_MODEL", "meterless/auto")


def attempt(label, msgs, model=None, max_tokens=8):
    start = time.time()
    try:
        kwargs = {"max_tokens": max_tokens} if max_tokens else {}
        r = get_client().chat.completions.create(
            model=model or default_model, messages=msgs, temperature=0.2, **kwargs)
        print(f"[ok]     {label}: {time.time() - start:.0f}s model={r.model} "
              f"reply={(r.choices[0].message.content or '')[:30]!r}", flush=True)
    except Exception as exc:
        print(f"[FAILED] {label}: {time.time() - start:.0f}s "
              f"{getattr(exc, 'status_code', '')} {str(exc)[:90]}", flush=True)


attempt("as sent, auto, no limit (the failing case)", folded, max_tokens=None)
attempt("as sent, auto, 8 tokens", folded)
for model in ["meterless/text", "meterless/local", "meterless/fast", "meterless/text.local"]:
    attempt(f"as sent, {model}", folded, model=model)
attempt("rules only", [{"role": "user", "content": rules + "\n\nReply: OK"}])
attempt("evidence only", [{"role": "user", "content": "Summarise this data in one line:\n" + evidence_json}])
attempt("short instruction + evidence",
        [{"role": "user", "content": "Write two sentences using only these numbers:\n" + evidence_json}])