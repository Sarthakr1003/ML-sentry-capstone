# ML Sentry

**An open-source, model-agnostic agentic governance layer for machine learning models in production.**

ML Sentry runs in Docker next to a model that is already live. It reads the model's evaluation metrics from
your own MLflow, keeps their history, and when a metric goes out of bounds an LLM agent judges how serious it is
and alerts you on Slack or Telegram. After you retrain, it checks again and tells you whether the problem is fixed.

Your model is never uploaded, moved or even loaded: ML Sentry only reads numbers, so it works with any model,
classifier or regressor, in any framework.

```
your model ──logs metrics──> your MLflow <──reads (HTTP)── ML Sentry (Docker)
                                                              │
                                    SQLite history ◄──────────┤
                                    limits from .env ─────────┤ check
                                                              ▼
                                         LangGraph agent: gather → judge → write
                                         (Groq → local Ollama → built-in rules)
                                                              ▼
                                               Slack / Telegram / console alert
```

## What it checks

Whatever metrics your model logs, each with a `_MAX` and/or `_MIN` limit in `.env`. The included demo watches:

- **performance decay**: prediction error (RMSE) and R²
- **data drift**: Population Stability Index (PSI) of the input features
- **bias**: the error gap between two groups (coastal vs inland districts)

Severity is **minor**, **moderate** or **critical**; a problem that lasts 3 evaluations in a row is raised one level.

## Quick start

```bash
git clone https://github.com/Sarthakr1003/ML-sentry-capstone.git
cd ML-sentry-capstone
cp .env.example .env        # then edit .env
docker compose up -d --build
docker compose logs -f
```

## Documentation

| Doc | What it covers |
|---|---|
| [docs/installation.md](docs/installation.md) | What you need and how to install |
| [docs/configuration.md](docs/configuration.md) | Every setting in `.env` |
| [docs/running.md](docs/running.md) | Running ML Sentry, and the full demo step by step |
| [docs/scenarios.md](docs/scenarios.md) | What happens with drift, decay, bias, recovery and outages |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Every error message, what it means and what to check |

## Repository layout

```
src/mlsentry/              ML Sentry itself (this is what runs in Docker)
  config.py                reads .env
  mlflow_reader.py         fetches finished runs from MLflow's REST API
  storage.py               SQLite history of runs, metrics and incidents
  checker.py               compares metrics with their limits; simple severity rules
  agent.py                 LangGraph agent that judges severity and writes the alert
  alerts.py                Slack, Telegram and console
  __main__.py              the polling loop (python -m mlsentry)
examples/california_housing/   the demo: data prep, baseline model, fault injection, simulator, retraining
tests/                     automated tests (python -m pytest -q)
docs/                      documentation
```

## Status

Phase 1 (alert system) for the B.Tech capstone at Dayananda Sagar University, 2026.
Phase 2 (next semester): the agent applies safe fixes itself and reports what it did, and Model Card / compliance reporting.

Team: Sarthak Ramachandra, Rishabh S, Saumyaa, Sonika. Supervisor: Dr. Abdul Haq Nalband.
