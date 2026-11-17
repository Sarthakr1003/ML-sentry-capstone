"""Checks of the setup wizard's live checks, with fake services (no network, no real keys)."""

import groq
import httpx
import pytest
import requests

from mlsentry import checks
from mlsentry.mlflow_reader import MLflowError, MLflowReader


# ---------- Groq ----------

def _status_error(cls, code):
    response = httpx.Response(code, request=httpx.Request("GET", "https://api.groq.com"))
    return cls("error", response=response, body=None)


def _fake_groq(monkeypatch, error=None, models=("openai/gpt-oss-120b",)):
    class Models:
        def list(self):
            if error:
                raise error
            return type("Page", (), {"data": [type("M", (), {"id": m}) for m in models]})

    class FakeGroq:
        def __init__(self, **kwargs):
            self.models = Models()

    monkeypatch.setattr(groq, "Groq", FakeGroq)


@pytest.mark.parametrize("error, words", [
    (_status_error(groq.AuthenticationError, 401), "401"),
    (_status_error(groq.RateLimitError, 429), "429"),
    (groq.APITimeoutError(request=httpx.Request("GET", "https://api.groq.com")), "in time"),
    (groq.APIConnectionError(request=httpx.Request("GET", "https://api.groq.com")), "internet"),
])
def test_groq_problems_are_explained(monkeypatch, error, words):
    _fake_groq(monkeypatch, error)
    ok, message = checks.check_groq("gsk_fake")
    assert not ok and words in message
    assert "gsk_fake" not in message                        # the key is never shown


def test_groq_key_works_and_unknown_model_is_flagged(monkeypatch):
    _fake_groq(monkeypatch)
    assert checks.check_groq("gsk_fake", "openai/gpt-oss-120b") == (True, "The Groq key works.")
    ok, message = checks.check_groq("gsk_fake", "llama-3.3-70b-versatile")
    assert ok and "no model called" in message


# ---------- MLflow ----------

def test_mlflow_experiments_and_logged_metrics(monkeypatch):
    def fake_call(self, method, path, **kwargs):
        if path == "/experiments/search":
            return {"experiments": [{"name": "Default"}, {"name": "house-prices"},
                                    {"name": "old", "lifecycle_stage": "deleted"}]}
        if path == "/experiments/get-by-name":
            return {"experiment": {"experiment_id": "1"}}
        return {"runs": [  # newest first
            {"info": {"run_id": "b", "start_time": 2}, "data": {"metrics": [{"key": "rmse", "value": 0.9}]}},
            {"info": {"run_id": "a", "start_time": 1},
             "data": {"metrics": [{"key": "rmse", "value": 0.5}, {"key": "r2", "value": 0.8}]}},
        ]}

    monkeypatch.setattr(MLflowReader, "_call", fake_call)
    ok, _, names = checks.check_mlflow("http://127.0.0.1:5000")
    assert ok and names == ["Default", "house-prices"]
    assert checks.logged_metrics("http://127.0.0.1:5000", "house-prices") == {"rmse": 0.9, "r2": 0.8}


def test_mlflow_not_running_is_explained(monkeypatch):
    def refuse(self, method, path, **kwargs):
        raise MLflowError("Cannot reach MLflow at ...: Connection refused")

    monkeypatch.setattr(MLflowReader, "_call", refuse)
    monkeypatch.setattr(checks, "in_docker", lambda: True)
    ok, message, names = checks.check_mlflow("http://host.docker.internal:5000")
    assert not ok and names == [] and "--host 0.0.0.0" in message


def test_mlflow_url_inside_and_outside_docker():
    assert checks.mlflow_url(8080) == "http://127.0.0.1:8080"
    assert checks.mlflow_url(8080, docker=True) == "http://host.docker.internal:8080"


# ---------- Slack and Telegram ----------

class FakeResponse:
    def __init__(self, code, body=None):
        self.status_code, self._body = code, body or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.HTTPError(response=self)


def test_slack_webhook_is_checked(monkeypatch):
    assert not checks.check_slack("https://example.com/hook")[0]           # wrong format, nothing sent
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(404))
    ok, message = checks.check_slack("https://hooks.slack.com/services/X")
    assert not ok and "404" in message
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(200))
    assert checks.check_slack("https://hooks.slack.com/services/X")[0]


def test_telegram_token_chat_id_and_webhook_conflict(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse(401))
    ok, message = checks.check_telegram_token("123:secret")
    assert not ok and "secret" not in message

    updates = {"result": [{"message": {"chat": {"id": 111, "first_name": "Old"}}},
                          {"message": {"chat": {"id": 987654321, "first_name": "Sarthak"}}}]}
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse(200, updates))
    assert checks.find_telegram_chat("123:secret") == (True, "Found the chat with Sarthak.", "987654321")

    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse(200, {"result": []}))
    ok, message, chat = checks.find_telegram_chat("123:secret")
    assert not ok and chat == "" and "send it any message" in message

    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse(409))
    assert "409" in checks.find_telegram_chat("123:secret")[1]
