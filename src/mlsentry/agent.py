"""
ML Sentry: the severity agent, built with LangGraph.

When a run breaks a limit, the agent walks through three steps (nodes):

  gather  -> collect context: recent history of each broken metric and how many
             runs in a row have been bad (the "streak")
  judge   -> decide severity (minor / moderate / critical), why, and what to do.
             Asks Groq first; if Groq is unavailable asks a local Ollama model;
             if neither answers properly, falls back to the simple rules in checker.py
  write   -> turn the decision into a short alert message

The LLM only sees metric names and numbers, never the model or the data,
so the same agent works for any model.
"""

import json
import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .checker import SEVERITIES, rule_severity

log = logging.getLogger("mlsentry.agent")

SYSTEM_PROMPT = """You are ML Sentry, an MLOps monitoring agent watching a machine learning model in production.
You receive the latest evaluation metrics that broke their allowed limits, their recent history,
and how many evaluation runs in a row have been bad.

Judge how serious the problem is:
- minor: slightly outside the limit, could be noise; keep watching
- moderate: clearly outside the limit; someone should investigate soon
- critical: far outside the limit, or a problem that keeps persisting; act now (e.g. retrain or roll back)

Use the metric names to explain the likely cause in plain language, for example:
error metrics (rmse, mae, r2, accuracy, f1) = the model's predictions got worse;
drift metrics (psi, drift) = incoming data no longer looks like the data the model learned from;
gap metrics = the model got much worse for one group than another (a fairness problem).

Reply with JSON only, exactly these keys:
{"severity": "minor|moderate|critical", "reason": "one or two sentences", "action": "one sentence"}"""


class AgentState(TypedDict, total=False):
    run: dict            # the MLflow run being judged
    breaches: list       # metrics outside their limits
    missing: list        # watched metrics the run did not log
    history: dict        # metric -> recent values, oldest first
    streak: int          # bad runs in a row, including this one
    rule_severity: str   # what the simple rules say
    severity: str
    reason: str
    action: str
    engine: str          # groq, ollama or rules
    message: str


def _parse(text: str) -> dict:
    """Read the LLM's JSON answer and make sure it is usable."""
    data = json.loads(text)
    severity = str(data.get("severity", "")).strip().lower()
    if severity not in SEVERITIES:
        raise ValueError(f"unknown severity '{severity}'")
    return {"severity": severity,
            "reason": str(data.get("reason", "")).strip(),
            "action": str(data.get("action", "")).strip()}


def _ask_groq(cfg: dict, messages: list) -> dict:
    from groq import Groq

    client = Groq(api_key=cfg["groq_key"], timeout=30)
    reply = client.chat.completions.create(
        model=cfg["groq_model"], messages=messages,
        response_format={"type": "json_object"}, temperature=0,
    )
    return _parse(reply.choices[0].message.content)


def _ask_ollama(cfg: dict, messages: list) -> dict:
    from ollama import Client

    client = Client(host=cfg["ollama_host"], timeout=180)
    reply = client.chat(model=cfg["ollama_model"], messages=messages,
                        format="json", options={"temperature": 0})
    return _parse(reply["message"]["content"])


def build_agent(cfg: dict, storage):
    """Create the LangGraph agent. `storage` gives it access to the metric history."""

    def gather(state: AgentState) -> AgentState:
        history = {b["metric"]: storage.history(b["metric"]) for b in state["breaches"]}
        streak = storage.breach_streak() + 1  # earlier bad runs + this one
        return {"history": history, "streak": streak,
                "rule_severity": rule_severity(state["breaches"], streak)}

    def judge(state: AgentState) -> AgentState:
        facts = {
            "run": state["run"]["run_name"],
            "breaches": [{k: b[k] for k in ("metric", "value", "limit", "side")} for b in state["breaches"]],
            "recent_history_oldest_first": state["history"],
            "bad_runs_in_a_row": state["streak"],
            "missing_metrics": state.get("missing", []),
            "rule_based_suggestion": state["rule_severity"],
        }
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(facts, indent=2)}]

        brains = []
        if cfg["groq_key"]:
            brains.append(("groq", _ask_groq))
        if cfg["ollama_model"]:
            brains.append(("ollama", _ask_ollama))
        for name, ask in brains:
            try:
                answer = ask(cfg, messages)
                return {**answer, "engine": name}
            except Exception as error:  # any failure: try the next brain
                log.warning("%s could not judge this run (%s), trying next option", name, error)

        worst = max(state["breaches"], key=lambda b: b["excess"])
        return {
            "severity": state["rule_severity"],
            "reason": f"{worst['metric']} is {worst['value']} against a limit of {worst['limit']} "
                      f"({worst['excess']:.0%} past it); {state['streak']} bad run(s) in a row.",
            "action": "Check the latest data and model, and retrain if the problem persists.",
            "engine": "rules",
        }

    def write(state: AgentState) -> AgentState:
        icon = {"minor": "🟡", "moderate": "🟠", "critical": "🔴"}[state["severity"]]
        lines = [f"{icon} {state['severity'].upper()}: run '{state['run']['run_name']}' is out of bounds"]
        for b in state["breaches"]:
            word = "above max" if b["side"] == "max" else "below min"
            lines.append(f"  • {b['metric']} = {b['value']} ({word} {b['limit']})")
        if state.get("missing"):
            lines.append(f"  • not logged: {', '.join(state['missing'])}")
        lines.append(f"Why: {state['reason']}")
        lines.append(f"Do: {state['action']}")
        lines.append(f"(judged by {state['engine']}, {state['streak']} bad run(s) in a row)")
        return {"message": "\n".join(lines)}

    graph = StateGraph(AgentState)
    graph.add_node("gather", gather)
    graph.add_node("judge", judge)
    graph.add_node("write", write)
    graph.add_edge(START, "gather")
    graph.add_edge("gather", "judge")
    graph.add_edge("judge", "write")
    graph.add_edge("write", END)
    return graph.compile()
