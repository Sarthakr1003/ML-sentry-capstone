# The setup questions (setup wizard)

You do not have to edit `.env` by hand. The first time you start ML Sentry it asks a few questions in the terminal,
tries every answer out, and writes `.env` for you. Later, the same command opens a settings menu.

| You use | Run, in the repo folder |
|---|---|
| Docker (normal use) | Windows: `.\mlsentry.cmd` · macOS/Linux: `./mlsentry.sh` |
| Python, no Docker | `mlsentry` (with the venv active) |

With Docker, the launcher asks the questions inside the container and then starts ML Sentry in the background.
Without Docker, the last question is "Start watching your model now?".

Keys, webhooks and tokens are typed hidden and are only saved in `.env` on your own computer (never on GitHub).
Press **Ctrl+C** at any time to stop: nothing is saved until you confirm at the end.

---

## The questions

| # | Question | Needed? | What to answer | What ML Sentry checks |
|---|---|---|---|---|
| 1 | Groq API key | Optional | Your key from console.groq.com/keys, or Enter to skip | One tiny free call to Groq |
| 2 | What type of model is ML Sentry watching? | **Yes** | Regression, Classification or RAG | – (cannot be changed later) |
| 3 | Which port is your MLflow running on? | **Yes** | e.g. `5000` (the `--port` you started MLflow with) | That MLflow answers |
| 4 | Which MLflow experiment is your model logging to? | **Yes** | Pick it from the list, or type the name | – (cannot be changed later) |
| 5 | Which metrics should ML Sentry watch? | **Yes**, at least one | Tick with Space, confirm with Enter | Only names your model really logs can be ticked |
| 5b | For "x", which is better? | Only for metrics ML Sentry does not know | Lower or higher is better | – |
| 6 | Alert if x goes ABOVE / BELOW | **Yes**, one per metric | A number. The suggestion in the box is a little worse than your model's latest value | Must be a number in a sensible range |
| 7 | Where should alerts go? | Optional | Slack and/or Telegram (none = only the terminal / Docker logs) | – |
| 7a | Slack webhook URL | For Slack | From your Slack app: Incoming Webhooks | Sends a test message |
| 7b | Telegram bot token | For Telegram | From @BotFather. Then send your bot any message and press Enter | Finds your chat ID by itself, sends a test message |
| 8 | Change advanced settings? | Optional | How often to check MLflow (default every 10 s), the local Ollama model | – |
| 9 | Save these settings? | **Yes** | Check the summary, then save (or go back and change) | – |

### How the metric list is built (question 5)
ML Sentry first reads which metrics your model really logs in MLflow (from its latest runs) and shows three groups:

| Group | Can you tick it? | Notes |
|---|---|---|
| **Recommended**: logged by your model and known for this model type | Yes, ticked by default | ML Sentry knows if lower or higher is better; `acc`, `val_accuracy` etc. are recognised as accuracy |
| **Other metrics your model logs** | Yes | You are asked whether lower or higher is better |
| **Standard metrics NOT logged by your model** | No (greyed out) | Shown so you know what is missing: log it from your model first |

If your model has not logged any run yet, the standard names for the model type are shown instead. Make sure your
model then logs exactly those names.

Standard metrics: **regression** rmse, mae, mse, r2, rmse_gap, max_psi · **classification** accuracy, precision,
recall, f1, roc_auc, log_loss, max_psi · **RAG (RAGAS)** faithfulness, answer_relevancy, context_precision,
context_recall, answer_correctness.

---

## Changing settings later (the menu)

Run the same command again (`.\mlsentry.cmd`, `./mlsentry.sh` or `mlsentry`). You can change: the Groq key,
metrics and thresholds, the MLflow port, the alert channels and the advanced settings. You can also send a test alert
or show the current settings (secrets masked). Choose **Save and exit** to keep your changes; with Docker,
ML Sentry restarts by itself with the new settings.

**Locked:** the model type and the experiment. One ML Sentry watches one model. To watch a second model,
clone ML Sentry again into another folder and run the setup there.

---

## Messages and what they mean

| Message | Meaning | What to do |
|---|---|---|
| Groq rejected this key (error 401) | Wrong or expired key | Copy it again, or create a new one at console.groq.com/keys |
| Groq says this key is not allowed (error 403) | The account cannot use the API | Check your Groq account |
| Groq is limiting requests (error 429) | Too many requests right now | Wait a minute and try again |
| Groq did not answer in time / Cannot reach Groq | No internet, or Groq is down | Check the connection, or continue without Groq (Ollama and the rules still work) |
| The key works, but Groq has no model called '…' | Groq retired that model | Pick a current model in the advanced settings / `GROQ_MODEL` |
| Cannot reach MLflow at … Is MLflow running, and on this port? | Nothing answers on that port | Start MLflow, or check the `--port` you used |
| … ML Sentry runs in Docker, so MLflow must accept outside connections | MLflow only accepts connections from your own computer | Start MLflow with `--host 0.0.0.0` instead of `--host 127.0.0.1` |
| No finished runs found in MLflow yet | Your model has not logged anything to this experiment | Fine for now; the standard names are shown |
| Please type a number / The limit cannot be above 1 | The threshold is not a number, or impossible (e.g. accuracy 1.5) | Type a plain number like `0.6` |
| Pick at least one metric | Nothing was ticked | Tick at least one with Space |
| A Slack webhook starts with https://hooks.slack.com/ | That is not a Slack webhook | Copy it from Incoming Webhooks |
| Slack rejected the webhook (HTTP 403/404) | The webhook is wrong or was deleted | Create a new webhook |
| Telegram rejected this bot token | Wrong token | Copy it again from @BotFather |
| No message found yet | You have not sent the bot a message | Open the bot, press Start, send any message, then try again |
| This bot is connected to a webhook (error 409) | The bot is used by another program | Create a new bot with @BotFather |
| The ML Sentry setup needs a terminal | It was started without a keyboard (e.g. `docker compose up`) | Use the launcher or run `mlsentry` in a terminal |
| Stopped. Nothing was saved. | You pressed Ctrl+C | Run the command again when ready |
