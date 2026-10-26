import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path("logs")

def new_trace_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"trace-{stamp}-{uuid.uuid4().hex[:6]}"

def log_event(trace_id, agent_name, action, input_summary="",
              output_summary="", model=None, retry_count=0, status="success"):
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "agent_name": agent_name,
        "action": action,
        "input_summary": input_summary,
        "output_summary": output_summary,
        "model": model,
        "retry_count": retry_count,
        "status": status,
    }
    LOG_DIR.mkdir(exist_ok=True)
    with open(LOG_DIR / f"{trace_id}.jsonl", "a") as f:
        f.write(json.dumps(event) + "\n")
    print(json.dumps(event))
    return event
