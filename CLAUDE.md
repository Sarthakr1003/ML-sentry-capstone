# ML Sentry: instructions for Claude Code

Read this file first, then read `context/PROJECT_CONTEXT.md` (full project context, decisions, repo state, plan). The raw audio transcripts and written feedback that shaped the design are in `context/AUDIO_TRANSCRIPTS.md`; read them when you need the original wording. The latest audio (Audio 3) wins where they disagree.

## One-paragraph summary
ML Sentry is an open-source, model-agnostic governance layer that runs in Docker next to a live model. It reads the model's evaluation metrics from the user's own MLflow, stores history in SQLite, compares them with thresholds from `.env`, uses a LangGraph agent (Groq primary, Ollama fallback) to judge severity (minor/moderate/critical), alerts on Slack (Telegram optional), and re-checks after retraining. The model is never uploaded or moved. Phase 1 (alert system) is due 31 Oct 2026 (30 Oct is the safe target). Phase 2 (reasoning and auto-fix, probably Model Cards) is next semester.

Owner: Sarthak Ramachandra (B.Tech AI&ML, DSU). Teammates Rishabh, Saumyaa, Sonika. Supervisor Dr. Abdul Haq Nalband.

## Current state (25 Oct 2026)
- Done and pushed: repo restructure, `.gitignore`, `.env.example`, `src/mlsentry/config.py` (`load_config()`).
- **Demo dataset just changed to California Housing (regression)**, because teammate Rishabh's ML Sentry repo is regression-only. German Credit files in `examples/german_credit/` and the column names in `src/mlsentry/injection.py` are now outdated.
- `context/data_prep.py` and `context/train_baseline.py` are the new California Housing scripts. Sarthak may not have saved them in the repo yet: check `examples/california_housing/`; if missing, copy them there (do not rewrite them differently).
- Next: run them and check the metrics (expect RMSE about 0.45-0.5, R2 about 0.83-0.85), make sure `.gitignore` has `data/processed/` and `models/`, commit and push, then rewrite `injection.py`, then MLflow + `simulate.py` + poller.

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
