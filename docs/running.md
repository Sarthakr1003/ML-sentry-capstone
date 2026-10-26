# Running ML Sentry

## A. Normal use: watch your own model (Docker)

1. Your model logs its evaluation metrics to your MLflow (one run per evaluation). Nothing else about your model changes.
2. Fill in `.env` (see [configuration.md](configuration.md)).
3. Start ML Sentry:

```bash
docker compose up -d --build
```

4. Watch what it does:

```bash
docker compose logs -f
```

5. Stop it:

```bash
docker compose down
```

The history stays in `state/mlsentry.db` on your computer, so a restart carries on where it stopped.
`restart: unless-stopped` brings it back automatically after a reboot while Docker is running.

## B. Without Docker

```bash
python -m mlsentry
```

Keeps running and checks every `POLL_INTERVAL_SECONDS`; Ctrl+C stops it.
`python -m mlsentry --once` checks once and exits (useful for testing or a scheduled task).

## C. The full demo (California Housing)

The demo plays the part of "your model". Use the venv install from [installation.md](installation.md).
Each step goes in its own PowerShell window, from the repo folder, with the venv active.

**1. Prepare the data and train the baseline model (once):**

```bash
python examples/california_housing/data_prep.py
python examples/california_housing/train_baseline.py
```

Expected: RMSE about 0.48, R² about 0.83.

**2. Start MLflow (window 1, leave it open):**

```bash
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root ./mlruns/artifacts --allowed-hosts "localhost:*,127.0.0.1:*,host.docker.internal:*"
```

Open http://localhost:5000 to see the MLflow page.

**3. Start ML Sentry (window 2):** `docker compose up -d --build` then `docker compose logs -f`, or without Docker: `python -m mlsentry`.

**4. Play a scenario (window 3):**

```bash
python examples/california_housing/simulate.py --scenario drift
```

Scenarios: `healthy`, `drift`, `decay`, `bias`. The first 3 batches are healthy, then the fault starts.
Watch window 2: ML Sentry stays quiet for the healthy batches, then alerts, getting more serious while the problem lasts.
See [scenarios.md](scenarios.md) for what each one shows.

**5. Retrain and re-check:**

```bash
python examples/california_housing/retrain.py --scenario drift
python examples/california_housing/simulate.py --scenario drift --healthy-batches 0 --model models/retrained.joblib --label after-retrain
```

ML Sentry checks the new model's results: a **RECOVERED** message if they are back within limits, or another alert if not.

**Useful simulator options:** `--delay 3` seconds between batches (default 3), `--healthy-batches 3`,
fault strength `--drift-shift 2.0`, `--decay-factor 1.3`, `--bias-factor 1.4`.

## D. Start fresh

Stop ML Sentry and MLflow, then delete the `state/` and `mlruns/` folders. Both are generated and not stored in Git.

## Run the tests

```bash
python -m pytest -q
```
