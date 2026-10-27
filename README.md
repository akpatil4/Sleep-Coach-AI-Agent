# Sleep Experiment Coach

A sleep-habit coach that helps one person decide **what to try next**, and checks its own numbers before it answers.

Most sleep trackers show log charts but don't say which habit is worth changing. You ask this agent a question, such as "Which habit should I test next to sleep better?". It then:

1. **Analyses your sleep log:** average sleep, sleep debt, and how strongly each habit (screen time, caffeine, bedtime) goes with how well you slept, including whether that pattern could be chance or a cause.
2. **Ranks one habit to test next,** skipping habits you've already tested.
3. **Checks the result of an experiment,** for example "did cutting screen time help?", with a confidence range and a plain statement of how weak or strong the evidence is.
4. **Checks itself.** The statistics are done by ordinary code, and the AI model only plans and writes up the findings. A separate code-based Evaluator rejects any draft with a number that isn't in the real results, or with wording that claims one thing causes another.
5. **Remembers** what you've tested, so the next answer changes.

It gives suggestions for experiments, not medical advice, and every report says so.

Built solo for the Open Agent Hackathon 2026 (Track 2). All data in this repository is synthetic.

**All data is synthetic.** This is not a medical tool.

## What it does
Ask "Which habit should I test next to sleep better?" and the agent returns a plain-language report: sleep stats, habit-by-habit correlations with p-values, before/after checks with confidence intervals, and one ranked experiment. It remembers what you've already tested.

## Architecture
Safety gate (plain code) → Planner → Executor → Evaluator, with up to 2 revisions.
- **Planner**: recalls H-MEM memory, asks the model for a tool plan, always adds `rank_experiments`.
- **Executor**: runs the statistics tools on the data, drafts the report through Meterless Nightshift.
- **Evaluator**: code checks. Every number must appear in the evidence, question, context or recalled memory. Causal wording is rejected unless negated. The top-ranked experiment must be quoted.
- Every step writes a JSONL trace line to `logs/`.

## Sponsor technology
- **Zetaris** (local Docker): the data layer. Table `SLEEPDB.daily_log`, read over JDBC (`localhost:10000`) when `SAMPLE_MODE=false`.
- **Meterless Nightshift**: OpenAI-compatible endpoint at `http://127.0.0.1:8088/v1`, routing to `gemini-3.1-flash-lite`.
- **Meterless H-MEM**: a minimal Python implementation (`app/engines/hmem.py`): short-term/working/long-term layers, TF-IDF retrieval, JSONL audit ledger, feedback, supersede. 7 tests.
- **Cursor**: coding editor. **NVIDIA**: not used.

## Setup
1. Python 3.11+, then `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
2. `cp .env.example .env` and fill in your own values.
3. **Quick start (no Docker, no Nightshift needed for the data part):** set `SAMPLE_MODE=true` to read `data/daily_log.csv`.
4. **Full setup:** run Zetaris in Docker, load the table with `python scripts/zetaris_load.py`, install Java 17 and set `JAVA_HOME`, start Meterless Nightshift with a Gemini key in its Providers tab, and set `METERLESS_BASE_URL`, `METERLESS_RUNTIME_TOKEN`, `METERLESS_MODEL`, `ZETARIS_USER`, `ZETARIS_PASSWORD`, `ZETARIS_TABLE`, `SAMPLE_MODE=false`.
5. The model calls need Nightshift running in either mode.

## Usage
- Run the samples: `python scripts/save_examples.py` (writes `output_examples/`).
- API: `uvicorn run:app --port 8000`, then `POST /run` with `{"request_id": "1", "user_query": "Which habit should I test next to sleep better?"}`.
- Tests: `pytest` (18 passing).

## Sample inputs and outputs
`input_examples/` and `output_examples/` hold three real runs, with traces (`*_trace.jsonl`) and `hmem_ledger_sample.jsonl`. Example 3 recalls 4 memories and proposes `bed_hour` after screen time was recorded as tested.

## Dependencies
FastAPI, uvicorn, scikit-learn, JayDeBeApi, JPype1, openai client (see `requirements.txt`).

## Limitations
- Synthetic data with planted patterns; the experiment in the demo is simulated.
- Sleep quality is self-reported.
- H-MEM is minimal: no dreaming, sleep cycle, conflict scanning or embeddings.
- Zetaris holds a single table, so its federation features aren't exercised.
- The first model lane (local Llama 1B) returned 502 errors; request-rewriting workarounds (including lowercasing) remain in `app/llm.py`.
- The safety gate is keyword-based.
- `demo/demo_offline.py` is an outdated stand-in-model demo.
