# Scenarios (what ML Sentry does in each situation)

The demo in `examples/california_housing/` creates each situation on purpose, so you can see ML Sentry react.
The model predicts house prices (regression). Every batch of 500 rows logged to MLflow is one evaluation.

| Metric | What it measures | Limit in the demo |
|---|---|---|
| `rmse` | Average size of the prediction error (in $100k) | above 0.60 = alert |
| `r2` | How much of the price variation the model explains (1 = perfect) | below 0.70 = alert |
| `rmse_gap` | Coastal error minus inland error: is the model much worse for one group? (our bias check for regression) | above 0.20 = alert |
| `max_psi` | Population Stability Index of the most-changed input feature: does the incoming data still look like the reference data? | above 0.20 = alert |

## 1. Healthy

All metrics stay inside their limits. ML Sentry saves them and stays quiet.
Demo numbers: RMSE 0.44-0.50, R² 0.84-0.85, gap 0.01-0.09, PSI under 0.06.

## 2. Data drift

**What happens:** the incoming data changes. In the demo every district's income (`MedInc`) is shifted up by 2.0 ($20k),
like a new customer population the model never saw.
**What ML Sentry sees:** `max_psi` jumps to about 2.4-3.0 (limit 0.2), and since the model is now
guessing outside what it learned, RMSE rises to about 0.88 and R² falls to about 0.4.
**Alert:** critical (PSI more than 10x its limit).
**After retraining** on data that includes the new population: everything returns to normal, ML Sentry sends **RECOVERED**.

## 3. Performance decay (concept drift)

**What happens:** the inputs look the same, but the right answers changed. In the demo all prices rise 30%,
so the old model keeps predicting old prices.
**What ML Sentry sees:** PSI stays normal (the inputs did not change), but RMSE rises to about 0.9 and R² falls to about 0.6.
**Alert:** moderate at first; it becomes **critical once the problem lasts 3 runs in a row**.
**After retraining:** R² recovers to 0.84, but RMSE stays around 0.57-0.65, because errors on 30% higher prices
are 30% bigger in dollars. Some batches are still above the 0.60 limit, so ML Sentry **alerts again**.
This is the "re-check after retraining" working: the team must decide whether the limit should change with the new price level.

## 4. Bias (one group treated worse)

**What happens:** prices rise 40% only for coastal districts. The model becomes much worse for coastal houses than inland ones.
**What ML Sentry sees:** `rmse_gap` jumps from under 0.1 to about 0.6-0.8 (limit 0.2); overall RMSE also rises to about 0.75.
**Alert:** critical (gap about 3x its limit).
**After retraining:** the gap falls to about 0.10-0.20, mostly fixed.

## 5. A metric is missing

If a run does not log one of the `METRICS`, ML Sentry checks the others and writes a warning
(and lists it in the alert if the run also breaks a limit). Usually a typo in `METRICS` or in the model's logging.

## 6. MLflow or the LLM is down

- MLflow unreachable: ML Sentry logs an error and tries again next poll. Nothing is lost, because it picks up all runs it has not seen yet.
- Groq fails: Ollama is asked. Ollama fails: built-in rules decide. An alert is always sent.
- Slack/Telegram fails: the alert is still printed to the console and saved in SQLite; the error is logged.

## 7. First start with old results

When ML Sentry starts with an empty history, runs already in MLflow are saved as **history only**, with no alerts,
so you are not flooded with alerts about the past. Use `python -m mlsentry --alert-existing` if you do want them checked.
