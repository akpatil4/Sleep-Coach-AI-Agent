# Planner -> Executor -> Evaluator, with a revision loop and H-MEM memory.
import json
import re
from pathlib import Path
from datetime import date

from app import tools
from app.evidence import before_after_with_ci
from app.experiments import rank_experiments
from app.llm import chat as default_chat
from app.safety import DISCLAIMER
from app.tracing import log_event

MAX_REVISIONS = 2
TOOL_NAMES = ["sleep_stats", "sleep_debt", "rank_experiments", "compare_periods"]
DEFAULT_TOOLS = ["sleep_stats", "sleep_debt", "rank_experiments"]

ISO = re.compile(r"\d{4}-\d{2}-\d{2}")
NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
CAUSAL = re.compile(
    r"\b(caus(?:e|es|ed|ing)|proves?|proven|because of|due to|leads? to|results? in)\b",
    re.I)
NEGATION = re.compile(r"\b(not|no|never|cannot|can't|isn't|doesn't|don't)\b", re.I)


def _numbers(text):
    text = ISO.sub(" ", text)                           # ignore dates
    text = re.sub(r"(?m)^\s*\d+[.)]\s+", "", text)      # ignore list numbering
    return NUM.findall(text)


def _supported(tok, allowed):
    v = abs(float(tok))
    tol = 0.0101 if "." in tok else 0.0                 # decimals: allow rounding
    return any(abs(v - abs(a)) <= tol for a in allowed)


def _parse_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _clean(o):
    """Make evidence JSON-safe: numpy -> python, NaN -> None."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and o != o:
        return None
    if hasattr(o, "item"):
        return _clean(o.item())
    return o


def memory_block(payload):
    if payload["memories"]:
        lines = [f"- {m['content']} (confidence {m['confidence']})"
                 for m in payload["memories"]]
        return "Relevant past experiments and findings:\n" + "\n".join(lines)
    return "No relevant past experiments were found."


def evidence_text(evidence):
    """Plain sentences instead of JSON. Nightshift rejected the JSON ranking block."""
    lines = []
    s = evidence.get("sleep_stats")
    if s:
        lines.append(
            f"Nights analysed: {s['nights']}. Average sleep: {s['avg_duration_h']} hours. "
            f"Average quality rating: {s['avg_quality']}. "
            f"Number of short nights: {s['nights_under_7h']}. "
            f"Bedtime varies by {s['bedtime_std_min']} minutes.")
    d = evidence.get("sleep_debt")
    if d:
        lines.append(f"Sleep debt over the last {d['window_nights']} nights against a "
                     f"{d['target_h']} hour target: {d['debt_h']} hours.")
    ranked = evidence.get("ranking", {}).get("ranked", [])
    if ranked:
        t = ranked[0]
        lines.append(f"Habit most linked with worse sleep: {t['habit']}, correlation {t['r']}, "
                     f"p-value {t['p_value']}, which is {t['label']}.")
        lines.append(f"Suggested experiment (quote exactly): {t['experiment']}")
    chk = evidence.get("experiment_check")
    if chk:
        ba = chk["before_after"]
        if "change" in ba:
            lines.append(
                f"Experiment on {chk.get('habit')}: nights before {ba['n_before']}, "
                f"nights after {ba['n_after']}. Sleep quality change {ba['change']}, "
                f"confidence interval {ba['ci95'][0]} to {ba['ci95'][1]}. "
                f"{ba['label']}. {ba['caution']}")
        else:
            lines.append(f"Experiment check: {ba.get('label')}.")
    return "\n".join(lines)



def evidence_levels(evidence):
    """Evidence text at decreasing size, so a rejected prompt can retry with less."""
    lines = evidence_text(evidence).split("\n")
    keep = ("Habit most linked", "Suggested experiment", "Experiment on", "Experiment check")
    core = [ln for ln in lines if ln.startswith(keep)]
    levels = ["\n".join(lines),
              "\n".join(ln for ln in lines if not ln.startswith("Sleep debt")),
              "\n".join(core)]
    unique = []
    for lv in levels:
        if lv and lv not in unique:
            unique.append(lv)
    return unique


def tested_habits(hmem):
    out = set()
    for rec in hmem.memories.values():
        if "experiment_result" in rec["tags"]:
            out.update(t[6:] for t in rec["tags"] if t.startswith("habit:"))
    return out


class PlannerAgent:
    name = "PlannerAgent"

    def __init__(self, llm, hmem, trace_id, models):
        self.llm, self.hmem, self.trace_id, self.models = llm, hmem, trace_id, models

    def plan(self, question, context):
        mem = self.hmem.retrieve(question, agent_id=self.name)
        log_event(self.trace_id, self.name, "recall_memory", input_summary=question[:80],
                  output_summary=f"{len(mem['memories'])} memories: {mem['retrieval_reason']}")
        system = ("You are the Planner of a sleep experiment coach. Decide which analysis "
                  "tools are needed to answer the question. Tools: sleep_stats, sleep_debt, "
                  "rank_experiments, compare_periods (only if an experiment start date is "
                  "given). Reply with JSON only: {\"tools\": [...], \"focus\": \"one sentence\"}")
        user = (f"Question: {question}\n\n{memory_block(mem)}\n\n"
                f"Experiment start date given: {bool(context.get('experiment_start'))}")
        reply = self.llm([{"role": "system", "content": system},
                          {"role": "user", "content": user}])
        self.models.add(reply["model"])
        parsed = _parse_json(reply["content"]) or {}
        chosen = [t for t in parsed.get("tools", []) if t in TOOL_NAMES]
        fallback = not chosen
        if fallback:
            chosen = list(DEFAULT_TOOLS)
        if context.get("experiment_start") and "compare_periods" not in chosen:
            chosen.append("compare_periods")
        if not context.get("experiment_start"):
            chosen = [t for t in chosen if t != "compare_periods"]
        plan = {"tools": chosen, "focus": str(parsed.get("focus", ""))[:200],
                "fallback_used": fallback}
        log_event(self.trace_id, self.name, "make_plan", model=reply["model"],
                  input_summary=question[:80],
                  output_summary=f"tools={chosen} fallback={fallback}")
        return plan, mem

    def finalise(self, text, evidence, context, approved, revisions):
        final = text.strip() + "\n\n" + DISCLAIMER
        if not approved:
            log_event(self.trace_id, self.name, "memory_write_skipped",
                      output_summary="draft not approved, nothing stored", status="needs_review")
            return final
        today = date.today().isoformat()
        check = evidence.get("experiment_check")
        if check and context.get("experiment_habit"):
            habit = context["experiment_habit"]
            ba = check["before_after"]
            self.hmem.add(
                f"Experiment result for {habit} starting {context['experiment_start']}: "
                f"sleep quality changed by {ba.get('change')} (95% CI {ba.get('ci95')}), "
                f"{ba.get('label')}, {ba.get('n_after')} nights after.",
                type="procedural", layer="working", confidence=0.6, agent_id=self.name,
                tags=["experiment", "experiment_result", f"habit:{habit}"], entities=[habit])
        ranked = evidence.get("ranking", {}).get("ranked", [])
        if ranked:
            top = ranked[0]
            self.hmem.add(
                f"Proposed experiment on {today}: {top['experiment']} "
                f"(habit {top['habit']}, {top['label']}).",
                type="procedural", layer="working", confidence=0.6, agent_id=self.name,
                tags=["experiment", "proposed", f"habit:{top['habit']}"],
                entities=[top["habit"]])
        self.hmem.mine_event("run_outcome", f"Run approved after {revisions} revision(s).",
                             agent_id=self.name)
        moved = self.hmem.end_run()
        log_event(self.trace_id, self.name, "memory_write",
                  output_summary=f"experiment memory saved, {moved} run notes kept")
        return final


class ExecutorAgent:
    name = "ExecutorAgent"

    def __init__(self, llm, trace_id, models):
        self.llm, self.trace_id, self.models = llm, trace_id, models

    def run_tools(self, plan, df, context, tested):
        evidence = {}
        if "sleep_stats" in plan["tools"]:
            evidence["sleep_stats"] = tools.sleep_stats(df)
        if "sleep_debt" in plan["tools"]:
            evidence["sleep_debt"] = tools.sleep_debt(df)
        if "rank_experiments" in plan["tools"]:
            skip = set(tested)
            if context.get("experiment_habit"):
                skip.add(context["experiment_habit"])
            evidence["ranking"] = rank_experiments(df, tested=skip)
        if "compare_periods" in plan["tools"] and context.get("experiment_start"):
            split = context["experiment_start"]
            evidence["experiment_check"] = {
                "habit": context.get("experiment_habit"),
                "before_after": before_after_with_ci(df, split),
                "period_means": tools.compare_periods(df, split)}
        evidence = _clean(evidence)
        log_event(self.trace_id, self.name, "run_tools",
                  output_summary=f"evidence keys: {sorted(evidence)}")
        return evidence

    def draft(self, question, mem, evidence, plan, issues=None, previous=None, revision=0):
        system = (
            "You are the Executor of a sleep experiment coach. Write a short report "
            "(under 180 words) answering the user's question using ONLY the EVIDENCE.\n"
            "Rules:\n"
            "- Use only numbers that appear in the EVIDENCE. Never invent, convert or "
            "recompute numbers, and do not use percentages.\n"
            "- State how strong the evidence is, using the labels in the EVIDENCE "
            "(for example 'could be noise').\n"
            "- Say that a habit is linked with worse sleep. Do not say it is the reason.\n"
            "- Copy the suggested experiment sentence from the EVIDENCE word for word.\n"
            "- If the EVIDENCE has experiment_check, report what happened and say it is "
            "not proof of cause.\n"
            "- If relevant past experiments are listed, refer to them.\n"
            "- Give no medical advice or diagnoses. Do not number your points.")
        user = f"Question: {question}\n\n{memory_block(mem)}"
        if issues:
            user += ("\n\nYour previous draft:\n" + (previous or "") +
                     "\n\nFix these issues and rewrite:\n- " + "\n- ".join(issues))
        base_user = user
        reply, last_exc = None, None
        for level, ev in enumerate(evidence_levels(evidence)):
            user = base_user + "\n\nEVIDENCE:\n" + ev
            try:
                reply = self.llm([{"role": "system", "content": system},
                                  {"role": "user", "content": user}])
                break
            except Exception as exc:
                if getattr(exc, "status_code", None) != 502:
                    raise
                last_exc = exc
                Path("logs").mkdir(exist_ok=True)
                Path("logs/rejected_prompt.txt").write_text(system + "\n\n" + user)
                log_event(self.trace_id, self.name, "draft_prompt_rejected",
                          input_summary=f"evidence level {level}, {len(user)} chars",
                          output_summary="Nightshift returned 502, retrying with less evidence",
                          retry_count=revision, status="retry")
        if reply is None:
            raise last_exc
        self.models.add(reply["model"])
        log_event(self.trace_id, self.name, "revise_draft" if issues else "write_draft",
                  model=reply["model"], retry_count=revision,
                  output_summary=reply["content"][:120])
        return reply["content"]


class EvaluatorAgent:
    name = "EvaluatorAgent"

    def __init__(self, trace_id):
        self.trace_id = trace_id

    def review(self, draft, evidence, question, context, revision=0):
        issues = []
        blob = json.dumps(evidence, default=str) + " " + question + " " + \
            json.dumps(context, default=str)
        allowed = [float(x) for x in _numbers(blob)]
        for tok in dict.fromkeys(_numbers(draft)):
            if not _supported(tok, allowed):
                issues.append(f"The number {tok} does not appear in the evidence. "
                              "Remove it or use a number from the evidence.")
        seen = set()
        for m in CAUSAL.finditer(draft):
            window = draft[max(0, m.start() - 30):m.start()]
            word = m.group(0).lower()
            if NEGATION.search(window) or word in seen:
                continue
            seen.add(word)
            issues.append(f"Causal wording '{m.group(0)}' is not allowed. "
                          "Describe an association instead.")
        ranked = evidence.get("ranking", {}).get("ranked", [])
        if ranked and ranked[0]["experiment"].lower() not in draft.lower():
            issues.append("Propose the top-ranked experiment and quote it exactly: "
                          f"'{ranked[0]['experiment']}'.")
        result = {"requires_revision": bool(issues), "issues": issues}
        log_event(self.trace_id, self.name, "review_draft", retry_count=revision,
                  output_summary=("revision requested: " + "; ".join(issues)[:200])
                  if issues else "approved",
                  status="revision_requested" if issues else "success")
        return result


def run_agents(question, context, df, hmem, trace_id, llm=default_chat):
    models = set()
    planner = PlannerAgent(llm, hmem, trace_id, models)
    executor = ExecutorAgent(llm, trace_id, models)
    evaluator = EvaluatorAgent(trace_id)

    plan, mem = planner.plan(question, context)
    evidence = executor.run_tools(plan, df, context, tested_habits(hmem))
    draft = executor.draft(question, mem, evidence, plan)
    review = evaluator.review(draft, evidence, question, context)
    findings, revisions = [], 0
    while review["requires_revision"] and revisions < MAX_REVISIONS:
        findings.extend(review["issues"])
        revisions += 1
        draft = executor.draft(question, mem, evidence, plan, issues=review["issues"],
                               previous=draft, revision=revisions)
        review = evaluator.review(draft, evidence, question, context, revision=revisions)
    approved = not review["requires_revision"]
    report = planner.finalise(draft, evidence, context, approved, revisions)
    log_event(trace_id, planner.name, "final_response",
              output_summary=f"approved={approved} revisions={revisions}",
              status="success" if approved else "needs_review")
    return {
        "status": "success" if approved else "needs_review",
        "agents_used": [planner.name, executor.name, evaluator.name],
        "result": {"report": report, "evidence": evidence, "plan": plan},
        "collaboration": {"feedback_loop_completed": True, "revision_count": revisions,
                          "evaluator_findings": findings,
                          "final_issues": [] if approved else review["issues"]},
        "memory": {"retrieved": len(mem["memories"]),
                   "retrieval_reason": mem["retrieval_reason"]},
        "models_used": sorted(models),
    }