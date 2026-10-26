"""
ML Sentry: sending alerts.

Every alert is printed to the console (so it shows in the Docker logs too),
then sent to each channel listed in ALERT_CHANNEL: slack and/or telegram.
A failed channel is logged but never stops ML Sentry.
"""

import logging

import requests

log = logging.getLogger("mlsentry.alerts")


def _slack(cfg: dict, text: str) -> None:
    if not cfg["slack_webhook"]:
        raise ValueError("SLACK_WEBHOOK_URL is not set in .env")
    response = requests.post(cfg["slack_webhook"], json={"text": text}, timeout=15)
    response.raise_for_status()


def _telegram(cfg: dict, text: str) -> None:
    if not cfg["telegram_token"] or not cfg["telegram_chat_id"]:
        raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must both be set in .env")
    url = f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage"
    response = requests.post(url, json={"chat_id": cfg["telegram_chat_id"], "text": text}, timeout=15)
    response.raise_for_status()


SENDERS = {"slack": _slack, "telegram": _telegram}


def send_alert(cfg: dict, text: str) -> list:
    """Send one alert everywhere it should go. Returns the channels it reached."""
    print("\n" + "=" * 60 + "\n" + text + "\n" + "=" * 60, flush=True)
    reached = ["console"]
    for channel in cfg["alert_channels"]:
        if channel == "console":
            continue
        sender = SENDERS.get(channel)
        if sender is None:
            log.error("Unknown ALERT_CHANNEL '%s' (use slack, telegram or console)", channel)
            continue
        try:
            sender(cfg, text)
            reached.append(channel)
        except ValueError as error:  # our own "setting missing" messages
            log.error("Could not send the alert to %s: %s", channel, error)
        except requests.HTTPError as error:
            log.error("Could not send the alert to %s: the service answered HTTP %s "
                      "(check the webhook/token in .env)", channel, error.response.status_code)
        except requests.RequestException as error:
            # Only the error type: the full text could contain the webhook URL or bot token
            log.error("Could not send the alert to %s: network problem (%s)", channel, type(error).__name__)
    return reached
