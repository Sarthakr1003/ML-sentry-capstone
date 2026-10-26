"""
ML Sentry: settings loader.
Reads every setting from the .env file, so no value is hardcoded in the code.

Settings already present in the environment (for example set by Docker)
win over the .env file. MLSENTRY_ENV_FILE can point to a different .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Project root = three levels up from this file (src/mlsentry/config.py)
ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(os.getenv("MLSENTRY_ENV_FILE", ROOT / ".env"))


def _require(name: str) -> str:
    """Return a setting's value, or stop with a clear message if it is missing."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise ValueError(f"Missing required setting {name} in .env (see .env.example)")
    return value.strip()


def _optional(name: str, default: str = "") -> str:
    """Return a setting's value, or the default. Placeholder values like 'your-...' count as empty."""
    value = (os.getenv(name) or "").strip()
    if value.lower().startswith("your-"):
        return default
    return value or default


def _number(name: str, kind=float):
    """Read a required setting and convert it to a number."""
    raw = _require(name)
    try:
        return kind(raw)
    except ValueError:
        raise ValueError(f"Setting {name} must be a number, got '{raw}'")


def _limits(metric: str) -> dict:
    """
    Read the allowed range for one metric.
      RMSE_MAX=0.6  -> alert when rmse goes ABOVE 0.6 (lower is better)
      R2_MIN=0.75   -> alert when r2 goes BELOW 0.75 (higher is better)
    A metric needs at least one of the two.
    """
    limits = {}
    for side in ("min", "max"):
        name = f"{metric.upper()}_{side.upper()}"
        if os.getenv(name, "").strip():
            limits[side] = _number(name)
    if not limits:
        raise ValueError(
            f"Metric '{metric}' has no limit. Add {metric.upper()}_MAX and/or {metric.upper()}_MIN to .env"
        )
    return limits


def load_config() -> dict:
    metrics = [m.strip() for m in _require("METRICS").split(",") if m.strip()]
    channels = [c.strip().lower() for c in _optional("ALERT_CHANNEL", "console").split(",") if c.strip()]

    sqlite_path = Path(_optional("SQLITE_PATH", "state/mlsentry.db"))
    if not sqlite_path.is_absolute():
        sqlite_path = ROOT / sqlite_path

    return {
        "mlflow_uri": _require("MLFLOW_TRACKING_URI").rstrip("/"),
        "experiment": _require("MLFLOW_EXPERIMENT"),
        "metrics": metrics,
        "limits": {m: _limits(m) for m in metrics},
        "poll_interval": int(_optional("POLL_INTERVAL_SECONDS", "30")),
        "sqlite_path": sqlite_path,
        "groq_key": _optional("GROQ_API_KEY"),
        "groq_model": _optional("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "ollama_host": _optional("OLLAMA_HOST", "http://localhost:11434"),
        "ollama_model": _optional("OLLAMA_MODEL", "llama3.1:8b"),
        "alert_channels": channels,
        "slack_webhook": _optional("SLACK_WEBHOOK_URL"),
        "telegram_token": _optional("TELEGRAM_BOT_TOKEN"),
        "telegram_chat_id": _optional("TELEGRAM_CHAT_ID"),
    }


SECRETS = ("groq_key", "slack_webhook", "telegram_token", "telegram_chat_id")

if __name__ == "__main__":
    cfg = load_config()
    # Never print secrets
    for secret in SECRETS:
        cfg[secret] = "(set)" if cfg[secret] else "(not set)"
    for key, value in cfg.items():
        print(f"{key}: {value}")
