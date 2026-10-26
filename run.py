from dotenv import load_dotenv
load_dotenv()

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.agents import run_agents
from app.data_source import load_daily_log
from app.engines.hmem import HMem
from app.safety import check_red_flags
from app.tracing import new_trace_id, log_event

app = FastAPI(title="Sleep Experiment Coach")


class RunRequest(BaseModel):
    request_id: str
    user_query: str
    context: dict | None = None   # e.g. {"experiment_start": "2026-09-20", "experiment_habit": "screen_min"}


@app.post("/run")
def run(req: RunRequest):
    trace_id = new_trace_id()
    log_path = f"logs/{trace_id}.jsonl"
    flag = check_red_flags(req.user_query)
    if flag["escalate"]:
        log_event(trace_id, "SafetyGate", "escalated",
                  input_summary=f"category={flag['category']}", status="escalated")
        return {"request_id": req.request_id, "trace_id": trace_id,
                "status": "escalated", "message": flag["message"], "log_path": log_path}
    log_event(trace_id, "System", "received_request", input_summary=req.user_query[:100])
    try:
        df, source = load_daily_log()
        log_event(trace_id, "System", "load_data",
                  output_summary=f"{len(df)} nights from {source}")
        result = run_agents(req.user_query, req.context or {}, df,
                            HMem(run_id=trace_id), trace_id)
    except Exception as exc:
        log_event(trace_id, "System", "error",
                  output_summary=f"{type(exc).__name__}: {str(exc)[:200]}", status="error")
        return JSONResponse(status_code=502, content={
            "request_id": req.request_id, "trace_id": trace_id, "status": "error",
            "error": f"{type(exc).__name__}: {exc}", "log_path": log_path})
    return {"request_id": req.request_id, "trace_id": trace_id,
            "data_source": source, "log_path": log_path, **result}


if __name__ == "__main__":
    uvicorn.run("run:app", host="0.0.0.0", port=8000)