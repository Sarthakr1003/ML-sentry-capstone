# Troubleshooting: errors and what they mean

Each entry: the message you see, what went wrong, and what to check.

## Settings

Messages shown while answering the setup questions are explained in [setup_wizard.md](setup_wizard.md#messages-and-what-they-mean).

**`Settings problem: Missing required setting X in .env (see .env.example). Run `mlsentry` to answer the setup questions.`**
ML Sentry has not been set up yet, or the setting is empty. Run `mlsentry` (Docker: `.\mlsentry.cmd` / `./mlsentry.sh`),
or copy the line from `.env.example` into `.env` and fill it in.
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
If the name is right and the MLflow page shows the experiment, ML Sentry is probably talking to a *different*
MLflow on the same port. Example: a Docker container (`docker ps` shows one on port 5000) listens on IPv6 as well,
and on Windows `localhost` tries IPv6 first, so it reaches the container instead of your server on `127.0.0.1`.
Stop the other one (`docker stop <name>`) and use `http://127.0.0.1:5000` instead of `localhost` in `.env`.

**MLflow says `Uvicorn running on http://0.0.0.0:5000`, but that link does not open**
Not an error. `--host 0.0.0.0` means "listen on every address of this computer" (needed when ML Sentry runs in Docker);
it is a setting, not an address a browser can visit. Type `http://127.0.0.1:5000` in the browser instead.
If the MLflow page shows no runs, click your experiment (e.g. `california-housing-demo`) in the list: MLflow often opens on the empty `Default` experiment.

**`MLflow answered 403 ... Invalid Host header` (or a DNS rebinding message)**
Newer MLflow servers only accept requests addressed to hosts they trust. When ML Sentry runs in Docker it calls
`host.docker.internal`, so start MLflow with `--allowed-hosts "localhost:*,127.0.0.1:*,host.docker.internal:*"` (see running.md).

**`First start: saved N existing run(s) as history without alerting`**
Not an error. See scenarios.md, section 7.

## The agent

**`groq could not judge this run (...), trying next option`**
Groq failed: wrong or missing key (401), no internet, rate limit (429), or the model name in `GROQ_MODEL` was retired.
ML Sentry moves on to Ollama, so you still get an alert.
`404 ... model_not_found` means Groq has retired that model (this happened to `llama-3.3-70b-versatile` in Oct 2026):
set `GROQ_MODEL` to a current one from console.groq.com/docs/models, e.g. `openai/gpt-oss-120b`.

**`ollama could not judge this run (...)`**
Ollama is not running (`ollama serve`, or open the Ollama app), the model is not pulled (`ollama pull llama3.1:8b`),
or the model gave an answer that was not valid JSON. The first call after a while can be slow because the model is loading.
The built-in rules take over; the alert says "judged by rules".

**`ollama could not judge this run (llama-server process has terminated: exit status 0xc0000409 ... CUDA error ...)`**
Ollama itself crashed on the graphics card while loading the model, usually on the first request or when another
program is using the GPU. Nothing is wrong with ML Sentry; the rules handle that run and the next run normally works.
If it keeps happening, update Ollama, close other GPU-heavy programs, or use a smaller model in `OLLAMA_MODEL`.

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

**Docker Desktop shows an error and quits: `initializing Inference manager ... dockerInference: The file cannot be accessed by the system`**
A leftover file from an earlier Docker session is locked. Run `wsl --shutdown`, delete
`%LOCALAPPDATA%\Docker\run\dockerInference`, then start Docker Desktop again. Updating Docker Desktop also helps.

**A folder called `.env` appeared, or `.env` is "not a file" / "is a directory"**
Docker creates a folder when the mounted `.env` file does not exist yet. Delete the `.env` folder and use the launcher
(`.\mlsentry.cmd` / `./mlsentry.sh`), which creates the file first.

**`unrecognized arguments: --once` (or `--alert-existing`)**
These options now belong to the watch command: `mlsentry watch --once`.

**The container keeps restarting**
Run `docker compose logs mlsentry`. Usually a settings problem (see above).

**Port 5000 or 11434 is already in use / ML Sentry reads the wrong MLflow or Ollama**
Another program or container is using the same port (check with `docker ps`). Either stop it while you use ML Sentry
(`docker stop <name>`), or run your MLflow on another port and change `MLFLOW_TRACKING_URI` and
`DOCKER_MLFLOW_TRACKING_URI` to match, e.g. `http://localhost:5001` and `http://host.docker.internal:5001`.

## The demo

**`UnicodeEncodeError: 'charmap' codec can't encode character`**
Old Windows consoles cannot print emoji. The scripts switch their output to UTF-8 to avoid this; if you see it in your own code,
run `$env:PYTHONIOENCODING="utf-8"` in PowerShell first.

**`FileNotFoundError: ... data/processed/...` or `models/baseline.joblib`**
Run `data_prep.py` and `train_baseline.py` first (see running.md). These files are generated, not stored in Git.

**`Ignoring invalid distribution ~lotly` during pip install**
Harmless leftover from an interrupted install. It can be ignored.
