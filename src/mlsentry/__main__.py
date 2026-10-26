"""
ML Sentry: the main loop.

Every POLL_INTERVAL_SECONDS:
  1. ask MLflow for evaluation runs ML Sentry has not seen yet
  2. save each run's metrics in SQLite
  3. compare them with the limits from .env
  4. out of bounds -> the agent judges severity and an alert is sent
     back within bounds after a problem -> a "recovered" message is sent
     (this is the re-check after the user retrains the model)

Run it with:  python -m mlsentry          (keeps running, Ctrl+C to stop)
              python -m mlsentry --once   (one check, then exit)
"""

import argparse
import logging
import sys
import time

from .agent import build_agent
from .alerts import send_alert
from .checker import check
from .config import load_config
from .mlflow_reader import MLflowError, MLflowReader
from .storage import Storage

log = logging.getLogger("mlsentry")

# Look this far back for runs that started earlier but finished after the last poll
LOOKBACK_MS = 60 * 60 * 1000


def handle_run(run: dict, cfg: dict, store: Storage, agent) -> None:
    """Check one new evaluation run and alert if needed."""
    breaches, missing = check(run["metrics"], cfg["limits"])
    previous = store.last_status()

    if breaches:
        result = agent.invoke({"run": run, "breaches": breaches, "missing": missing})
        sent_to = send_alert(cfg, result["message"])
        store.save_incident(run["run_id"], "breach", result["severity"], result["engine"],
                            breaches, result["message"], sent_to)
        store.save_run(run, "breach")
        log.info("%s: %s (%s)", run["run_name"], result["severity"], result["engine"])
        return

    if missing:
        log.warning("%s did not log: %s", run["run_name"], ", ".join(missing))
    if previous == "breach":
        message = (f"🟢 RECOVERED: run '{run['run_name']}' is back within all limits "
                   f"({', '.join(f'{m}={run['metrics'][m]:.4g}' for m in cfg['limits'] if m in run['metrics'])})")
        sent_to = send_alert(cfg, message)
        store.save_incident(run["run_id"], "recovered", "", "", [], message, sent_to)
    store.save_run(run, "ok")
    log.info("%s: ok", run["run_name"])


def poll(cfg: dict, reader: MLflowReader, store: Storage, agent, alert_existing: bool) -> int:
    """One pass: fetch and handle new runs. Returns how many were new."""
    first_start = store.is_empty()
    since = max(store.last_start_time() - LOOKBACK_MS, 0)
    new_runs = [r for r in reader.finished_runs(since) if not store.seen(r["run_id"])]

    if first_start and new_runs and not alert_existing:
        # Old results are saved as history, so we don't flood the user with alerts about the past
        for run in new_runs:
            store.save_run(run, "history")
        log.info("First start: saved %d existing run(s) as history without alerting. "
                 "New runs from now on will be checked.", len(new_runs))
        return len(new_runs)

    for run in new_runs:
        handle_run(run, cfg, store, agent)
    return len(new_runs)


def main() -> None:
    parser = argparse.ArgumentParser(prog="mlsentry", description="Watch a model's MLflow metrics and alert")
    parser.add_argument("--once", action="store_true", help="check once and exit")
    parser.add_argument("--alert-existing", action="store_true",
                        help="on first start, also check runs that already exist in MLflow")
    args = parser.parse_args()

    # Alerts contain emoji; the default Windows console encoding cannot show them
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # hide one line per LLM request

    try:
        cfg = load_config()
    except ValueError as error:
        log.error("Settings problem: %s", error)
        sys.exit(1)

    reader = MLflowReader(cfg["mlflow_uri"], cfg["experiment"])
    store = Storage(cfg["sqlite_path"])
    agent = build_agent(cfg, store)
    brains = (["groq"] if cfg["groq_key"] else []) + (["ollama"] if cfg["ollama_model"] else []) + ["rules"]
    log.info("ML Sentry watching experiment '%s' at %s, metrics %s, every %ss. Agent order: %s. Alerts: %s",
             cfg["experiment"], cfg["mlflow_uri"], ", ".join(cfg["metrics"]), cfg["poll_interval"],
             " -> ".join(brains), ", ".join(["console"] + [c for c in cfg["alert_channels"] if c != "console"]))

    while True:
        try:
            poll(cfg, reader, store, agent, args.alert_existing)
        except MLflowError as error:
            log.error("%s. Will try again in %ss.", error, cfg["poll_interval"])
        if args.once:
            break
        try:
            time.sleep(cfg["poll_interval"])
        except KeyboardInterrupt:
            log.info("Stopped.")
            break


if __name__ == "__main__":
    main()
