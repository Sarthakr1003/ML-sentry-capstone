"""
ML Sentry: the setup questionnaire and the settings menu.

Instead of editing .env by hand, the user answers questions in the terminal:

  first time (`mlsentry` or `mlsentry setup`):
    Groq key -> model type -> MLflow port -> experiment -> metrics -> thresholds
    -> alert channels -> advanced settings -> summary -> save
  later (`mlsentry` or `mlsentry menu`):
    a menu to change any of these, except the model type and the experiment
    (one ML Sentry watches one model).

Every answer is tried out first (see checks.py) and only written to .env when the
user confirms. Ctrl+C at any point leaves .env exactly as it was.
"""

import os
import sys
from pathlib import Path

import questionary
from questionary import Choice, Separator

from . import checks
from .config import ROOT
from .envfile import EnvFile
from .metrics_catalog import CATALOG, MODEL_TYPES, known_metric, parse_threshold, suggest_threshold
from .mlflow_reader import MLflowError

ENV_PATH = Path(os.getenv("MLSENTRY_ENV_FILE", ROOT / ".env"))

SAVED, NOT_SAVED = 0, 2   # exit codes; the Docker launchers restart ML Sentry only after SAVED
DEFAULTS = {
    "GROQ_MODEL": "openai/gpt-oss-120b",
    "OLLAMA_HOST": "http://localhost:11434",
    "OLLAMA_MODEL": "llama3.1:8b",
    "POLL_INTERVAL_SECONDS": "10",
    "SQLITE_PATH": "state/mlsentry.db",
}


class Cancelled(Exception):
    """The user pressed Ctrl+C."""


def _ask(question):
    """Show one question; Ctrl+C anywhere stops the whole questionnaire."""
    answer = question.ask()
    if answer is None:
        raise Cancelled
    return answer


def _good(text: str) -> None:
    questionary.print(f"  ✔ {text}", style="fg:ansigreen")


def _bad(text: str) -> None:
    questionary.print(f"  ✖ {text}", style="fg:ansired")


def _note(text: str) -> None:
    questionary.print(f"  {text}", style="fg:ansiyellow")


def _real(value: str) -> str:
    """The value, or '' for an empty value or a placeholder copied from .env.example (your-...)."""
    return "" if value.lower().startswith("your-") else value


def _mask(value: str) -> str:
    return "set ✔" if _real(value) else "not set"


def _port_of(url: str) -> str:
    return url.rsplit(":", 1)[-1].strip("/") if url.count(":") == 2 else "5000"


# ---------- one function per question (the menu reuses them) ----------

def ask_groq(env: EnvFile, first_time: bool) -> None:
    current = _real(env.get("GROQ_API_KEY"))
    hint = "press Enter to keep the current one" if current else "press Enter to skip"
    model = env.get("GROQ_MODEL", DEFAULTS["GROQ_MODEL"])
    while True:
        key = _ask(questionary.password(f"Groq API key ({hint}):")).strip()
        if not key:
            if not current:
                _note("No Groq key: ML Sentry will judge alerts with local Ollama, or its built-in rules.")
                env.set("GROQ_API_KEY", "")
            return
        ok, message = checks.check_groq(key, model)
        if ok:
            _good(message)
            env.set("GROQ_API_KEY", key)
            return
        _bad(message)
        choice = _ask(questionary.select("What now?", choices=[
            "Type the key again", "Save it anyway (I will fix it later)", "Continue without Groq"]))
        if choice.startswith("Save"):
            env.set("GROQ_API_KEY", key)
            return
        if choice.startswith("Continue"):
            env.set("GROQ_API_KEY", "")
            return


def ask_model_type(env: EnvFile) -> str:
    model_type = _ask(questionary.select(
        "What type of model is ML Sentry watching?",
        choices=[Choice(label, value=key) for key, label in MODEL_TYPES.items()]))
    env.set("MODEL_TYPE", model_type)
    return model_type


def _valid_port(text: str):
    return True if text.strip().isdigit() and 1 <= int(text) <= 65535 else "Type a port number between 1 and 65535, e.g. 5000"


def ask_mlflow_port(env: EnvFile) -> tuple:
    """Returns (url ML Sentry uses to reach MLflow, experiment names, reachable?)."""
    current = _port_of(env.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    while True:
        port = int(_ask(questionary.text("Which port is your MLflow running on?", default=current,
                                         validate=_valid_port)))
        url = checks.mlflow_url(port, docker=checks.in_docker())
        ok, message, names = checks.check_mlflow(url)
        if ok:
            _good(message)
        else:
            _bad(message)
            choice = _ask(questionary.select("What now?", choices=["Try again", "Use this port anyway (MLflow is not running yet)"]))
            if choice == "Try again":
                current = str(port)
                continue
        # The address as seen from this computer, and from inside the ML Sentry container
        env.set("MLFLOW_TRACKING_URI", checks.mlflow_url(port))
        env.set("DOCKER_MLFLOW_TRACKING_URI", checks.mlflow_url(port, docker=True))
        return url, names, ok


def ask_experiment(env: EnvFile, names: list) -> str:
    other = "(type another name)"
    if names:
        experiment = _ask(questionary.select("Which MLflow experiment is your model logging to?",
                                             choices=names + [other]))
    else:
        experiment = other
    if experiment == other:
        experiment = _ask(questionary.text("MLflow experiment name:",
                                           validate=lambda t: True if t.strip() else "The experiment name is needed")).strip()
    env.set("MLFLOW_EXPERIMENT", experiment)
    return experiment


def _current_limits(env: EnvFile) -> dict:
    limits = {}
    for metric in [m.strip() for m in env.get("METRICS").split(",") if m.strip()]:
        for side in ("max", "min"):
            value = env.get(f"{metric.upper()}_{side.upper()}")
            if value:
                limits.setdefault(metric, {})[side] = float(value)
    return limits


def ask_metrics(env: EnvFile, model_type: str, url: str, experiment: str, reachable: bool) -> None:
    """Questions 5 and 6: which metrics to watch, and a limit for each."""
    current = _current_limits(env)
    logged = {}
    if reachable:
        try:
            logged = checks.logged_metrics(url, experiment)
        except MLflowError:
            logged = {}

    def better(direction):
        return "lower is better" if direction == "max" else "higher is better"

    choices = []
    if logged:
        known = {name: known_metric(model_type, name) for name in logged}
        recommended = [n for n in logged if known[n]]
        others = [n for n in logged if not known[n]]
        found = {known[n].name for n in recommended}
        missing = [m for m in CATALOG[model_type] if m.name not in found]
        if recommended:
            choices.append(Separator(f"── Recommended for {model_type} (found in your MLflow) ──"))
            choices += [Choice(f"{n:<20} {known[n].label}, {better(known[n].direction)} (latest {logged[n]:.4g})",
                               value=n, checked=(n in current) if current else True) for n in sorted(recommended)]
        if others:
            choices.append(Separator("── Other metrics your model logs ──"))
            choices += [Choice(f"{n:<20} (latest {logged[n]:.4g})", value=n, checked=n in current)
                        for n in sorted(others)]
        if missing:
            choices.append(Separator(f"── Standard {model_type} metrics NOT logged by your model ──"))
            choices += [Choice(m.name, disabled="not in MLflow: log it from your model first") for m in missing]
    else:
        _note("No finished runs found in MLflow yet, so these are the standard names. "
              "Make sure your model logs exactly these names.")
        choices = [Choice(f"{m.name:<20} {m.label}, {better(m.direction)}", value=m.name,
                          checked=m.name in current) for m in CATALOG[model_type]]

    picked = _ask(questionary.checkbox(
        "Which metrics should ML Sentry watch? (Space to tick, Enter to confirm)", choices=choices,
        validate=lambda selected: True if selected else "Pick at least one metric"))

    limits = {}
    for name in picked:
        metric = known_metric(model_type, name)
        if metric:
            direction, low, high, default = metric.direction, metric.low, metric.high, metric.default
        else:
            old_side = next(iter(current.get(name, {})), None)
            direction = _ask(questionary.select(
                f'For "{name}", which is better?',
                choices=[Choice("Lower is better (alert when it goes ABOVE the limit)", value="max"),
                         Choice("Higher is better (alert when it goes BELOW the limit)", value="min")],
                default=old_side))
            low = high = default = None
        latest = logged.get(name)
        suggestion = current.get(name, {}).get(direction, suggest_threshold(direction, latest, default))
        word = "ABOVE" if direction == "max" else "BELOW"
        hint = f" (latest {latest:.4g})" if latest is not None else ""
        text = _ask(questionary.text(
            f"Alert if {name} goes {word}{hint}:", default="" if suggestion is None else f"{suggestion:g}",
            validate=lambda t, low=low, high=high: parse_threshold(t, low, high)[1] or True))
        limits[name] = {direction: parse_threshold(text, low, high)[0]}
    env.set_metrics(limits)


def _ask_slack(env: EnvFile) -> bool:
    current = _real(env.get("SLACK_WEBHOOK_URL"))
    keep = "press Enter to keep the current one" if current else "from your Slack app's Incoming Webhooks page"
    while True:
        webhook = _ask(questionary.password(f"Slack webhook URL ({keep}):")).strip() or current
        ok, message = checks.check_slack(webhook)
        if ok:
            _good(message)
            env.set("SLACK_WEBHOOK_URL", webhook)
            return True
        _bad(message)
        if _ask(questionary.select("What now?", choices=["Type the webhook again", "Skip Slack"])) == "Skip Slack":
            return False


def _ask_telegram(env: EnvFile) -> bool:
    current = _real(env.get("TELEGRAM_BOT_TOKEN"))
    keep = "press Enter to keep the current one" if current else "from @BotFather"
    while True:
        token = _ask(questionary.password(f"Telegram bot token ({keep}):")).strip() or current
        ok, message = checks.check_telegram_token(token)
        if ok:
            _good(message)
            break
        _bad(message)
        if _ask(questionary.select("What now?", choices=["Type the token again", "Skip Telegram"])) == "Skip Telegram":
            return False

    bot = message.split("@")[-1]
    while True:
        _ask(questionary.text(f"Open @{bot} in Telegram, press Start and send it any message. Then press Enter here."))
        ok, message, chat_id = checks.find_telegram_chat(token)
        if ok:
            _good(message)
            break
        _bad(message)
        choice = _ask(questionary.select("What now?", choices=["Try again", "Type the chat ID myself", "Skip Telegram"]))
        if choice == "Skip Telegram":
            return False
        if choice == "Type the chat ID myself":
            chat_id = _ask(questionary.text("Telegram chat ID:", validate=lambda t: True if t.strip().lstrip("-").isdigit()
                                            else "A chat ID is a number, e.g. 987654321")).strip()
            break

    ok, message = checks.check_telegram(token, chat_id)
    (_good if ok else _bad)(message)
    env.set("TELEGRAM_BOT_TOKEN", token)
    env.set("TELEGRAM_CHAT_ID", chat_id)
    return True


def ask_alerts(env: EnvFile) -> None:
    current = [c.strip() for c in env.get("ALERT_CHANNEL").split(",")]
    picked = _ask(questionary.checkbox(
        "Where should alerts go? (Space to tick, Enter to confirm; none = only this terminal)",
        choices=[Choice("Slack", value="slack", checked="slack" in current),
                 Choice("Telegram", value="telegram", checked="telegram" in current)]))
    channels = []
    if "slack" in picked and _ask_slack(env):
        channels.append("slack")
    if "telegram" in picked and _ask_telegram(env):
        channels.append("telegram")
    env.set("ALERT_CHANNEL", ",".join(channels) or "console")


def ask_advanced(env: EnvFile, first_time: bool) -> None:
    if first_time and not _ask(questionary.confirm("Change advanced settings? (how often to check, Ollama)", default=False)):
        return
    env.set("POLL_INTERVAL_SECONDS", _ask(questionary.text(
        "Check MLflow every how many seconds?", default=env.get("POLL_INTERVAL_SECONDS", DEFAULTS["POLL_INTERVAL_SECONDS"]),
        validate=lambda t: True if t.strip().isdigit() and int(t) >= 1 else "Type a whole number of seconds, e.g. 10")).strip())
    env.set("OLLAMA_MODEL", _ask(questionary.text(
        "Local Ollama model (backup when Groq is unavailable):",
        default=env.get("OLLAMA_MODEL", DEFAULTS["OLLAMA_MODEL"]))).strip())
    env.set("OLLAMA_HOST", _ask(questionary.text(
        "Ollama address:", default=env.get("OLLAMA_HOST", DEFAULTS["OLLAMA_HOST"]))).strip())


# ---------- summary, first-time setup and menu ----------

def summary(env: EnvFile) -> str:
    limits = _current_limits(env)
    watching = ", ".join(f"{m} {'>' if 'max' in b else '<'} {next(iter(b.values())):g}" for m, b in limits.items())
    return "\n".join([
        f"  Model type:   {env.get('MODEL_TYPE')}   (locked)",
        f"  Experiment:   {env.get('MLFLOW_EXPERIMENT')}   (locked)",
        f"  MLflow:       port {_port_of(env.get('MLFLOW_TRACKING_URI'))}",
        f"  Watching:     {watching or 'nothing yet'}",
        f"  Groq key:     {_mask(env.get('GROQ_API_KEY'))}",
        f"  Alerts:       {env.get('ALERT_CHANNEL', 'console')}",
        f"  Slack:        {_mask(env.get('SLACK_WEBHOOK_URL'))}   Telegram: {_mask(env.get('TELEGRAM_BOT_TOKEN'))}",
        f"  Checks every: {env.get('POLL_INTERVAL_SECONDS', DEFAULTS['POLL_INTERVAL_SECONDS'])} s   "
        f"Ollama: {env.get('OLLAMA_MODEL', DEFAULTS['OLLAMA_MODEL'])}",
    ])


def first_time_setup(env: EnvFile) -> bool:
    questionary.print("\nML Sentry setup: a few questions, then ML Sentry starts watching your model.\n"
                      "Press Ctrl+C at any time to stop; nothing is saved until the end.\n", style="bold")
    for key, value in DEFAULTS.items():
        if not env.get(key):
            env.set(key, value)
    ask_groq(env, first_time=True)
    model_type = ask_model_type(env)
    url, names, reachable = ask_mlflow_port(env)
    experiment = ask_experiment(env, names)
    ask_metrics(env, model_type, url, experiment, reachable)
    ask_alerts(env)
    ask_advanced(env, first_time=True)
    while True:
        questionary.print("\nYour settings:\n" + summary(env), style="bold")
        choice = _ask(questionary.select("Save these settings?", choices=[
            "Yes, save", "Change the metrics or thresholds", "Change the alert channels", "Cancel without saving"]))
        if choice == "Yes, save":
            env.save()
            return True
        if choice.startswith("Cancel"):
            return False
        if "metrics" in choice:
            ask_metrics(env, model_type, url, experiment, reachable)
        else:
            ask_alerts(env)


def menu(env: EnvFile) -> bool:
    changed = False
    model_type = env.get("MODEL_TYPE")
    while True:
        limits = _current_limits(env)
        choice = _ask(questionary.select("What do you want to change?" + (" (unsaved changes)" if changed else ""), choices=[
            Choice(f"Groq API key            [{_mask(env.get('GROQ_API_KEY'))}]", value="groq"),
            Choice(f"Metrics and thresholds  [{', '.join(limits) or 'none'}]", value="metrics"),
            Choice(f"MLflow port             [{_port_of(env.get('MLFLOW_TRACKING_URI'))}]", value="port"),
            Choice(f"Alert channels          [{env.get('ALERT_CHANNEL', 'console')}]", value="alerts"),
            Choice("Advanced settings", value="advanced"),
            Choice("Send a test alert", value="test"),
            Choice("Show current settings", value="show"),
            Separator(),
            Choice("Save and exit", value="save"),
            Choice("Exit without saving", value="quit"),
            Separator("── locked: one ML Sentry watches one model ──"),
            Choice(f"Model type: {model_type}", disabled="locked"),
            Choice(f"Experiment: {env.get('MLFLOW_EXPERIMENT')}", disabled="locked"),
        ]))
        if choice == "save":
            env.save()
            return True
        if choice == "quit":
            if not changed or _ask(questionary.confirm("Exit and lose your unsaved changes?", default=False)):
                return False
            continue
        if choice == "show":
            questionary.print(summary(env))
            continue
        if choice == "test":
            from .alerts import send_alert
            cfg = {"alert_channels": [c.strip() for c in env.get("ALERT_CHANNEL", "console").split(",")],
                   "slack_webhook": env.get("SLACK_WEBHOOK_URL"), "telegram_token": env.get("TELEGRAM_BOT_TOKEN"),
                   "telegram_chat_id": env.get("TELEGRAM_CHAT_ID")}
            _good(f"Test alert reached: {', '.join(send_alert(cfg, '🧪 ML Sentry test alert from the settings menu.'))}")
            continue
        if choice == "groq":
            ask_groq(env, first_time=False)
        elif choice == "metrics":
            url = checks.mlflow_url(int(_port_of(env.get("MLFLOW_TRACKING_URI"))), docker=checks.in_docker())
            ok, _, _ = checks.check_mlflow(url)
            if not ok:
                _note("MLflow is not reachable right now, so the standard metric names are shown.")
            ask_metrics(env, model_type, url, env.get("MLFLOW_EXPERIMENT"), ok)
        elif choice == "port":
            ask_mlflow_port(env)
        elif choice == "alerts":
            ask_alerts(env)
        elif choice == "advanced":
            ask_advanced(env, first_time=False)
        changed = True


def run(command: str | None) -> tuple:
    """
    Run the questionnaire or the menu. Returns (exit code, start watching now?).
    `command` is "setup", "menu" or None (choose by whether ML Sentry is set up yet).
    """
    if not sys.stdin.isatty():
        print("The ML Sentry setup needs a terminal to ask its questions. Run `mlsentry` in a terminal "
              "(with Docker: the mlsentry.cmd / mlsentry.sh launcher, or `docker compose run --rm mlsentry`).")
        return NOT_SAVED, False

    env = EnvFile(ENV_PATH)
    set_up = bool(env.get("MODEL_TYPE"))
    try:
        if set_up:
            if command == "setup":
                _note("ML Sentry is already set up for a "
                      f"{env.get('MODEL_TYPE')} model, so here is the settings menu instead.")
            saved = menu(env)
        else:
            saved = first_time_setup(env)
    except (Cancelled, KeyboardInterrupt):
        _note("Stopped. Nothing was saved.")
        return NOT_SAVED, False

    if not saved:
        _note("Nothing was saved.")
        return NOT_SAVED, False
    _good(f"Settings saved to {ENV_PATH.name}.")
    if checks.in_docker():
        _good("ML Sentry will now (re)start in the background with these settings.")
        return SAVED, False
    print("  To change settings later, run: mlsentry")
    try:
        start = questionary.confirm("Start watching your model now?", default=True).ask()
    except KeyboardInterrupt:
        start = False
    return SAVED, bool(start)
