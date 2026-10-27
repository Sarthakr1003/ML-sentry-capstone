# ML Sentry: instructions for Claude Code

Read this file first, then read `context/PROJECT_CONTEXT.md` (full project context, decisions, repo state, plan). The raw audio transcripts and written feedback that shaped the design are in `context/AUDIO_TRANSCRIPTS.md`; read them when you need the original wording. The latest audio (Audio 3) wins where they disagree.

## One-paragraph summary
ML Sentry is an open-source, model-agnostic governance layer that runs in Docker next to a live model. It reads the model's evaluation metrics from the user's own MLflow, stores history in SQLite, compares them with thresholds from `.env`, uses a LangGraph agent (Groq primary, Ollama fallback) to judge severity (minor/moderate/critical), alerts on Slack (Telegram optional), and re-checks after retraining. The model is never uploaded or moved. Phase 1 (alert system) is due 31 Oct 2026 (30 Oct is the safe target). Phase 2 (reasoning and auto-fix, probably Model Cards) is next semester.

Owner: Sarthak Ramachandra (B.Tech AI&ML, DSU). Teammates Rishabh, Saumyaa, Sonika. Supervisor Dr. Abdul Haq Nalband.

## Current state (26 Oct 2026)
- **Phase 1 code is built and pushed** (on 26 Oct Sarthak asked Claude to write all the code; he learns it afterwards, so the next job is walking him through each module and quizzing him).
- ML Sentry core in `src/mlsentry/`: `config.py`, `mlflow_reader.py` (MLflow REST), `storage.py` (SQLite), `checker.py` (MAX/MIN limits + rule severity), `agent.py` (LangGraph gather -> judge -> write; Groq -> Ollama -> rules), `alerts.py` (console/Slack/Telegram), `__main__.py` (poll loop, `python -m mlsentry`). Installed with `pip install -e .` (pyproject.toml).
- Demo in `examples/california_housing/`: `data_prep.py`, `train_baseline.py` (RMSE 0.476, R2 0.826), `injection.py` (drift/decay/bias), `simulate.py` (batches -> MLflow, PSI via Evidently), `retrain.py`. German Credit files in `examples/german_credit/` are legacy.
- Docker: built and tested end to end on 27 Oct (container read MLflow on host port 5001, raised drift alerts via host Ollama). Sarthak's other project's containers (`ragops_mlflow` on 5000, `ragops_ollama` on 11434) clash with the demo ports: stop them or use port 5001. Host Ollama sometimes crashes with a CUDA error on first load; the rules fallback covers it.
- Docs in `docs/`: installation, configuration, running, scenarios, troubleshooting. Tests in `tests/` (`python -m pytest -q`).
- Sarthak's own `.env` still has the old German Credit settings (`METRICS=accuracy,f1`, `*_THRESHOLD`): it must be updated from `.env.example`. Testing used `state/test.env` via `MLSENTRY_ENV_FILE`.
- Open: test Docker, real Slack webhook and Groq key, update the KSCST doc for regression, questions for Rishabh/supervisor (PROJECT_CONTEXT.md section 10).

## How to work with Sarthak (IMPORTANT, overrides your default of doing everything)
- He is learning and must be able to defend every line at the viva. Explain in plain language, from scratch, before acting. Be concise.
- For core logic (threshold comparison, drift/bias/decay checks, the LangGraph agent) do NOT write it for him. Explain the steps, let him type it, then review his code. You may do the mechanical parts yourself: git, installs, path/env fixes, running tests, writing docs.
- Use plan mode for new steps. Say what you will change before changing it. Never edit many files at once without telling him.
- After each finished step ask him 3 short questions about it.
- When he asks for "the code", give complete files, not small edits.
- Commit and push to GitHub after every finished step: small commits, clear messages.
- Secrets only in `.env` (gitignored). Keep `.env.example` current with fake values. One `.env` for everything a user edits.
- Nothing hardcoded: column names, paths, metric names and thresholds come from `.env`.
- After each finished task write a short plain-language doc in `docs/` (separate `.md` files: installation, configuration, running, troubleshooting/errors and what they mean, scenarios).
- Do not copy code from Rishabh's repo (github.com/rishsoraganvi/ML_Sentry). It is a different layer (approvals, retrain, release, rollback, audit; no LLM). See PROJECT_CONTEXT.md section 9.

## Environment
Windows, PowerShell. Repo `C:\Users\sarth\ML-sentry-capstone`, GitHub `Sarthakr1003/ML-sentry-capstone`. venv at `.\venv` (activate with `venv\Scripts\activate`). Python 3.14. Use `python -m pip`. The pip warning "Ignoring invalid distribution ~lotly" is harmless.
