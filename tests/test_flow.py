"""End-to-end checks of one polling pass, with a fake MLflow and no LLM (rules only)."""

import json

import pytest

from mlsentry.__main__ import poll
from mlsentry.agent import _parse, build_agent
from mlsentry.storage import Storage

CFG = {
    "limits": {"rmse": {"max": 0.6}, "r2": {"min": 0.7}},
    "groq_key": "",
    "ollama_model": "",       # no LLM: the agent must fall back to rules
    "alert_channels": ["console"],
}


class FakeReader:
    """Stands in for MLflowReader: returns whatever runs we give it."""

    def __init__(self):
        self.runs = []

    def add(self, name, rmse, r2):
        self.runs.append({"run_id": name, "run_name": name, "start_time": len(self.runs) + 1,
                          "metrics": {"rmse": rmse, "r2": r2}})

    def finished_runs(self, since_ms=0):
        return [r for r in self.runs if r["start_time"] >= since_ms]


@pytest.fixture
def setup(tmp_path):
    store = Storage(tmp_path / "test.db")
    return FakeReader(), store, build_agent(CFG, store)


def incidents(store):
    return store.db.execute("SELECT kind, severity, engine FROM incidents ORDER BY id").fetchall()


def test_first_start_saves_old_runs_without_alerting(setup):
    reader, store, agent = setup
    reader.add("old-bad", rmse=0.9, r2=0.5)
    assert poll(CFG, reader, store, agent, alert_existing=False) == 1
    assert incidents(store) == []


def test_breach_then_recovery(setup):
    reader, store, agent = setup
    reader.add("old", rmse=0.5, r2=0.8)
    poll(CFG, reader, store, agent, alert_existing=False)  # first start: history only

    reader.add("bad-1", rmse=0.9, r2=0.6)
    reader.add("bad-2", rmse=0.95, r2=0.6)
    reader.add("bad-3", rmse=0.66, r2=0.75)
    reader.add("fixed", rmse=0.5, r2=0.82)
    assert poll(CFG, reader, store, agent, alert_existing=False) == 4

    assert incidents(store) == [
        ("breach", "moderate", "rules"),
        ("breach", "moderate", "rules"),
        ("breach", "moderate", "rules"),   # only 10% past the limit, but 3 bad runs in a row
        ("recovered", "", ""),
    ]
    assert poll(CFG, reader, store, agent, alert_existing=False) == 0  # nothing is checked twice


def test_llm_answer_parsing():
    assert _parse(json.dumps({"severity": "Critical", "reason": "r", "action": "a"}))["severity"] == "critical"
    with pytest.raises(ValueError):
        _parse(json.dumps({"severity": "catastrophic"}))
