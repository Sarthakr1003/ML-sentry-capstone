# Troubleshooting: errors and what they mean

Each entry: the message you see, what went wrong, and what to check.

## Settings

**`Settings problem: Missing required setting X in .env (see .env.example)`**
The setting is not in `.env`, or is empty. Copy the line from `.env.example` into `.env` and fill it in.
Check you are running from the repo folder, and that the file is called exactly `.env` (Windows sometimes saves it as `.env.txt`).

**`Metric 'x' has no limit. Add X_MAX and/or X_MIN to .env`**
Every name in `METRICS` needs a limit. Older versions used `X_THRESHOLD`; rename it to `X_MIN` or `X_MAX`.

**`Setting X must be a number, got '...'`**
A limit or interval has text in it. Use plain numbers like `0.6` (no `%`, no quotes).

Run `python -m mlsentry.config` to see what ML Sentry actually loaded.

## MLflow

**`Cannot reach MLflow at http://...`**
MLflow is not running, or the address is wrong. Open the address in a browser. If it does not load, start MLflow.
In Docker, `localhost` means the container itself: check `DOCKER_MLFLOW_TRACKING_URI` (see configuration.md).
ML Sentry keeps retrying every poll, so it recovers by itself once MLflow is back.

**`MLflow has no experiment named 'x' yet`**
`MLFLOW_EXPERIMENT` does not match any experiment. Names are case sensitive. In the demo the experiment
is created by the first `simulate.py` run.

**`MLflow answered 403 ... Invalid Host header` (or a DNS rebinding message)**
Newer MLflow servers only accept requests addressed to hosts they trust. When ML Sentry runs in Docker it calls
`host.docker.internal`, so start MLflow with `--allowed-hosts "localhost:*,127.0.0.1:*,host.docker.internal:*"` (see running.md).

**`First start: saved N existing run(s) as history without alerting`**
Not an error. See scenarios.md, section 7.

## The agent

**`groq could not judge this run (...), trying next option`**
Groq failed: wrong or missing key (401), no internet, rate limit (429), or the model name in `GROQ_MODEL` was retired.
ML Sentry moves on to Ollama, so you still get an alert.

**`ollama could not judge this run (...)`**
Ollama is not running (`ollama serve`, or open the Ollama app), the model is not pulled (`ollama pull llama3.1:8b`),
or the model gave an answer that was not valid JSON. The first call after a while can be slow because the model is loading.
The built-in rules take over; the alert says "judged by rules".

## Alerts

**`Could not send the alert to slack: the service answered HTTP 403/404`**
The webhook URL is wrong or was deleted. Create a new incoming webhook and paste it into `SLACK_WEBHOOK_URL`.

**`Could not send the alert to telegram: the service answered HTTP 401`** (bad token) **or 400** (bad chat id)
Check `TELEGRAM_BOT_TOKEN`, and that you have sent your bot at least one message so the chat exists.

**`Unknown ALERT_CHANNEL 'x'`**
Use `slack`, `telegram` or `console`.

The alert is always printed to the console and stored in SQLite even when a channel fails.

## Docker

**`failed to connect to the docker API` / `cannot find the file specified ... dockerDesktopLinuxEngine`**
Docker Desktop is not running. Start it and wait until it says "Engine running".

**The container keeps restarting**
Run `docker compose logs mlsentry`. Usually a settings problem (see above).

## The demo

**`UnicodeEncodeError: 'charmap' codec can't encode character`**
Old Windows consoles cannot print emoji. The scripts switch their output to UTF-8 to avoid this; if you see it in your own code,
run `$env:PYTHONIOENCODING="utf-8"` in PowerShell first.

**`FileNotFoundError: ... data/processed/...` or `models/baseline.joblib`**
Run `data_prep.py` and `train_baseline.py` first (see running.md). These files are generated, not stored in Git.

**`Ignoring invalid distribution ~lotly` during pip install**
Harmless leftover from an interrupted install. It can be ignored.
