"""
ML Sentry: settings loader.
Reads every setting from the .env file, so no value is hardcoded in the code.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Project root = three levels up from this file (src/mlsentry/config.py)
ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")


def _require(name: str) -> str:
    """Return a setting's value, or stop with a clear message if it is missing."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise ValueError(f"Missing required setting {name} in .env (see .env.example)")
    return value.strip()


def _number(name: str, kind=float):
    """Read a required setting and convert it to a number."""
    raw = _require(name)
    try:
        return kind(raw)
    except ValueError:
        raise ValueError(f"Setting {name} must be a number, got '{raw}'")


def load_config() -> dict:
    metrics = [m.strip() for m in _require("METRICS").split(",") if m.strip()]
    # One threshold per metric, e.g. accuracy -> ACCURACY_THRESHOLD
    thresholds = {m: _number(f"{m.upper()}_THRESHOLD") for m in metrics}

    return {
        "mlflow_uri": _require("MLFLOW_TRACKING_URI"),
        "experiment": _require("MLFLOW_EXPERIMENT"),
        "metrics": metrics,
        "thresholds": thresholds,
        "poll_interval": int(os.getenv("POLL_INTERVAL_SECONDS", "30")),
        "groq_key": os.getenv("GROQ_API_KEY", ""),
        "alert_channel": os.getenv("ALERT_CHANNEL", "slack"),
        "slack_webhook": os.getenv("SLACK_WEBHOOK_URL", ""),
    }


if __name__ == "__main__":
    cfg = load_config()
    # Never print secrets
    for secret in ("groq_key", "slack_webhook"):
        cfg[secret] = "(set)" if cfg[secret] else "(not set)"
    for key, value in cfg.items():
        print(f"{key}: {value}")