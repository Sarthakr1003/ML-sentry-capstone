# Configuration

Everything you can change is in **one file, `.env`**, in the repo folder.
Copy `.env.example` to `.env` and edit it. `.env` holds your secrets (API keys, webhooks),
is listed in `.gitignore`, and must never be committed.

No column name, metric name, threshold or path is written inside the code. That is what makes
ML Sentry work with any model: you tell it, in `.env`, what to watch.

## Where your metrics live

| Setting | Example | Meaning |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | Address of your MLflow server |
| `MLFLOW_EXPERIMENT` | `california-housing-demo` | The MLflow experiment your model logs its evaluations to |

ML Sentry treats every **finished run** in that experiment as one evaluation result.

## What to watch

| Setting | Example | Meaning |
|---|---|---|
| `METRICS` | `rmse,r2,rmse_gap,max_psi` | Metric names exactly as your model logs them, comma separated |
| `<NAME>_MAX` | `RMSE_MAX=0.60` | Alert when the metric goes **above** this (for "lower is better" metrics like errors and drift) |
| `<NAME>_MIN` | `R2_MIN=0.70` | Alert when the metric goes **below** this (for "higher is better" metrics like accuracy, R², F1) |
| `POLL_INTERVAL_SECONDS` | `10` | How often ML Sentry asks MLflow for new results |
| `SQLITE_PATH` | `state/mlsentry.db` | Where the history is kept (relative to the repo folder) |

Every metric in `METRICS` needs at least one `_MAX` or `_MIN`. The name is the metric name in capitals.
For a classifier you might use:

```
METRICS=accuracy,f1
ACCURACY_MIN=0.90
F1_MIN=0.85
```

### How to pick the limits
Look at your model's normal numbers first, then set each limit a little outside the normal range.
For the demo the healthy batches had RMSE 0.44-0.50, R² 0.84-0.85, group gap up to 0.09 and PSI up to 0.05,
so the limits are RMSE 0.60, R² 0.70, gap 0.20 and PSI 0.20. A PSI above 0.2 is the common rule of thumb for "significant drift".

## The agent

| Setting | Example | Meaning |
|---|---|---|
| `GROQ_API_KEY` | (your key) | Groq is asked first. Leave the placeholder to skip Groq |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Which Groq model to use. Groq retires models over time: if you get "model_not_found", pick a current one from console.groq.com/docs/models |
| `OLLAMA_HOST` | `http://localhost:11434` | Your local Ollama, the fallback |
| `OLLAMA_MODEL` | `llama3.1:8b` | A model you have pulled (`ollama list`). Empty = skip Ollama |

If both fail (no key, no internet, Ollama not running), simple rules decide severity, so you always get an alert.

## Alerts

| Setting | Example | Meaning |
|---|---|---|
| `ALERT_CHANNEL` | `slack` | `slack`, `telegram` or `console`; several with commas, e.g. `slack,telegram` |
| `SLACK_WEBHOOK_URL` | `https://hooks.slack.com/services/...` | Slack: Apps > Incoming Webhooks > Add to a channel |
| `TELEGRAM_BOT_TOKEN` | `123456:ABC...` | Telegram: create a bot with @BotFather |
| `TELEGRAM_CHAT_ID` | `123456789` | Your chat with the bot (send it a message, then open `https://api.telegram.org/bot<token>/getUpdates`) |

Alerts are always printed to the console as well.

## Docker only

| Setting | Default | Meaning |
|---|---|---|
| `DOCKER_MLFLOW_TRACKING_URI` | `http://host.docker.internal:5000` | MLflow address as seen from inside the container |
| `DOCKER_OLLAMA_HOST` | `http://host.docker.internal:11434` | Ollama address as seen from inside the container |

Inside a container, `localhost` means the container itself, not your computer.
`host.docker.internal` is Docker's name for your computer. If your MLflow runs on another machine,
put its real address in both `MLFLOW_TRACKING_URI` and `DOCKER_MLFLOW_TRACKING_URI`.

## Demo only

`DEMO_TARGET_COL`, `DEMO_GROUP_COL` and `DEMO_BATCH_SIZE` are read only by the scripts in
`examples/california_housing/`. ML Sentry itself ignores them.

## Check your settings

```bash
python -m mlsentry.config
```

This prints every setting (secrets show only as "(set)" or "(not set)") or tells you exactly which one is missing.
