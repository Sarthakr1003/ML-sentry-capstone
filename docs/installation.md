# Installation

ML Sentry runs on your own computer or server, next to the model you already have.
Your model never leaves where it runs: ML Sentry only reads its evaluation numbers from MLflow.

## What you need

| Need | Why | Check with |
|---|---|---|
| Git | to download the repo | `git --version` |
| Docker Desktop (or Docker Engine + Compose) | to run ML Sentry as a container | `docker compose version` |
| An MLflow tracking server that your model logs its evaluation metrics to | ML Sentry reads the metrics from there | open its address in a browser |
| A Groq API key (free at console.groq.com), optional | the agent's main LLM | |
| Ollama with a model pulled, optional | the agent's local fallback LLM | `ollama list` |
| A Slack incoming webhook or a Telegram bot, optional | where alerts are sent | |

Without Groq or Ollama, ML Sentry still works: it falls back to simple built-in severity rules.
Without Slack or Telegram, alerts are printed to the console (`docker compose logs`).

## Install with Docker (normal use)

```bash
git clone https://github.com/Sarthakr1003/ML-sentry-capstone.git
cd ML-sentry-capstone
```

Copy the settings template, then fill it in (see [configuration.md](configuration.md)):

```bash
cp .env.example .env
```

On Windows PowerShell use `copy .env.example .env`. Then see [running.md](running.md).

## Install without Docker (development, or to run the demo)

Needs Python 3.11 or newer.

```bash
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

On macOS/Linux activate with `source venv/bin/activate`.

- `requirements.txt` installs everything, including the demo's libraries (scikit-learn, Evidently, MLflow, pytest).
- `pip install -e .` installs ML Sentry itself, so `python -m mlsentry` works from any folder.

Check the install:

```bash
python -m pytest -q
```

All tests should pass.
