"""
ML Sentry: live checks used by the setup wizard.

Before saving an answer, the wizard tries it out for real: is the Groq key
accepted, is MLflow answering on that port, do the Slack and Telegram details
deliver a message? Each check returns (ok, message) with a plain-language
message the wizard can show as it is.

Messages never contain the key, webhook or bot token themselves.
"""

from pathlib import Path

import requests

from .alerts import _slack, _telegram
from .mlflow_reader import MLflowError, MLflowReader

TIMEOUT = 15  # seconds


def in_docker() -> bool:
    """True when ML Sentry runs inside a Docker container."""
    return Path("/.dockerenv").exists()


# ---------- Groq ----------

def check_groq(key: str, model: str = "") -> tuple:
    """Try the key with one small, free request (the list of Groq's models)."""
    import groq

    try:
        client = groq.Groq(api_key=key, timeout=TIMEOUT, max_retries=0)
        available = [m.id for m in client.models.list().data]
    except groq.AuthenticationError:
        return False, "Groq rejected this key (error 401): it is wrong or has expired. Create a new one at https://console.groq.com/keys"
    except groq.PermissionDeniedError:
        return False, "Groq says this key is not allowed to do this (error 403). Check your Groq account."
    except groq.RateLimitError:
        return False, "Groq is limiting requests for this key right now (error 429). Wait a minute and try again."
    except groq.APITimeoutError:
        return False, "Groq did not answer in time. Check your internet connection and try again."
    except groq.APIConnectionError:
        return False, "Cannot reach Groq. Check your internet connection and try again."
    except groq.APIStatusError as error:
        return False, f"Groq answered with an error (HTTP {error.status_code}). Try again later."
    if model and model not in available:
        return True, (f"The key works, but Groq has no model called '{model}'. "
                      "ML Sentry will use Ollama or the built-in rules until you pick another model.")
    return True, "The Groq key works."


# ---------- MLflow ----------

def mlflow_url(port: int, docker: bool = False) -> str:
    """Where MLflow is, seen from where ML Sentry runs (inside a container 'localhost' is the container)."""
    host = "host.docker.internal" if docker else "127.0.0.1"
    return f"http://{host}:{port}"


def check_mlflow(url: str) -> tuple:
    """Is MLflow answering? Returns (ok, message, experiment names)."""
    try:
        names = MLflowReader(url, "").experiments()
    except MLflowError as error:
        message = f"Cannot reach MLflow at {url}. Is MLflow running, and on this port?"
        if "Connection refused" in str(error) and in_docker():
            message += (" ML Sentry runs in Docker, so MLflow must accept outside connections: "
                        "start it with --host 0.0.0.0 instead of --host 127.0.0.1.")
        elif "answered" in str(error):
            message = f"Something answered at {url}, but it does not look like MLflow: {error}"
        return False, message, []
    return True, f"Connected to MLflow ({len(names)} experiment(s) found).", names


def logged_metrics(url: str, experiment: str, runs: int = 20) -> dict:
    """
    Metric names the model really logs, with their newest value, from the latest runs:
    {"rmse": 0.48, "r2": 0.83, ...}. Empty if the experiment has no finished runs yet.
    """
    latest = {}
    for run in MLflowReader(url, experiment).recent_runs(runs):  # newest first
        for name, value in run["metrics"].items():
            latest.setdefault(name, value)
    return latest


# ---------- Slack ----------

def check_slack(webhook: str) -> tuple:
    """Check the webhook looks right, then send a test message."""
    if not webhook.startswith("https://hooks.slack.com/"):
        return False, "A Slack webhook starts with https://hooks.slack.com/ . Copy it again from your Slack app's Incoming Webhooks page."
    try:
        _slack({"slack_webhook": webhook}, "✅ ML Sentry test message: Slack alerts are working.")
    except requests.HTTPError as error:
        return False, f"Slack rejected the webhook (HTTP {error.response.status_code}). It may be wrong or deleted; create a new one."
    except requests.RequestException:
        return False, "Cannot reach Slack. Check your internet connection and try again."
    return True, "Test message sent to Slack. Check your channel."


# ---------- Telegram ----------

def _telegram_api(token: str, method: str) -> dict:
    response = requests.get(f"https://api.telegram.org/bot{token}/{method}", timeout=TIMEOUT)
    if response.status_code != 200:
        raise requests.HTTPError(response=response)
    return response.json()


def check_telegram_token(token: str) -> tuple:
    """Ask Telegram who this bot is. Returns (ok, message)."""
    try:
        bot = _telegram_api(token, "getMe")["result"]
    except requests.HTTPError as error:
        if error.response.status_code in (401, 404):
            return False, "Telegram rejected this bot token. Copy it again from @BotFather."
        return False, f"Telegram answered with an error (HTTP {error.response.status_code}). Try again later."
    except requests.RequestException:
        return False, "Cannot reach Telegram. Check your internet connection and try again."
    return True, f"Bot token works: @{bot.get('username', '?')}"


def find_telegram_chat(token: str) -> tuple:
    """
    Find the chat ID from the latest message someone sent to the bot.
    Returns (ok, message, chat_id).
    """
    try:
        updates = _telegram_api(token, "getUpdates")["result"]
    except requests.HTTPError as error:
        if error.response.status_code == 409:
            return False, ("This bot is connected to a webhook, so its messages cannot be read here (error 409). "
                           "Create a new bot with @BotFather, or remove the webhook."), ""
        return False, f"Telegram answered with an error (HTTP {error.response.status_code}).", ""
    except requests.RequestException:
        return False, "Cannot reach Telegram. Check your internet connection and try again.", ""
    for update in reversed(updates):  # newest message last
        message = update.get("message") or update.get("channel_post") or update.get("my_chat_member") or {}
        chat = message.get("chat")
        if chat:
            name = chat.get("title") or chat.get("first_name") or chat.get("username") or "your chat"
            return True, f"Found the chat with {name}.", str(chat["id"])
    return False, "No message found yet. Open your bot in Telegram, press Start, send it any message, then try again.", ""


def check_telegram(token: str, chat_id: str) -> tuple:
    """Send a test message to the chat."""
    try:
        _telegram({"telegram_token": token, "telegram_chat_id": chat_id},
                  "✅ ML Sentry test message: Telegram alerts are working.")
    except requests.HTTPError as error:
        return False, f"Telegram could not deliver the test message (HTTP {error.response.status_code}). Check the chat ID."
    except requests.RequestException:
        return False, "Cannot reach Telegram. Check your internet connection and try again."
    return True, "Test message sent to Telegram. Check your phone."
