# ML Sentry: progress so far

Status as of **28 Oct 2026**. Phase 1 deadline: **31 Oct 2026** (safe target 30 Oct).
For what ML Sentry is and how to use it, see [README.md](README.md) and [docs/](docs/).

## In one line

Phase 1 (the alert system) is **built, tested and on GitHub**. What remains is setting it up with real accounts
(Groq, Slack), one full demo dry run, learning the code for the viva, and updating the slides.

## What ML Sentry does

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

1. Every few seconds it asks MLflow for new evaluation results (one MLflow run = one result).
2. It saves them in SQLite and compares each metric with its limit from `.env` (`<NAME>_MAX` or `<NAME>_MIN`).
3. Out of bounds: the agent decides **minor / moderate / critical**, explains why and suggests what to do, and an alert is sent.
4. Back within bounds after a problem (e.g. after retraining): a **RECOVERED** message is sent. Still bad after retraining: it alerts again.

The model is never uploaded, moved or loaded; ML Sentry only reads numbers, so it works with any model.

## Key decisions

| Decision | Why |
|---|---|
| Open-source layer that runs in Docker next to a live model (option D) | Model stays where it runs; no hosting; works for any model |
| Read metrics from the user's own MLflow over its REST API | No dependency on the model or the mlflow package |
| Demo dataset switched from German Credit to **California Housing** (regression) | Teammate Rishabh's ML Sentry is regression-only |
| Limits as `_MAX` / `_MIN` instead of `_THRESHOLD` | Errors (RMSE) must stay below a limit, scores (R²) above |
| Drift measured with **PSI** (via Evidently) | Evidently's default test flagged drift even on healthy data; PSI is clean (healthy ≤ 0.05, drifted ≈ 2.4) |
| Bias for regression = **error gap between groups** (coastal vs inland) | No approve/reject in regression, so Fairlearn's demographic parity does not apply |
| Fault injection moved into the demo, not ML Sentry | ML Sentry itself never changes anyone's data |
| Agent order Groq → Ollama → rules | An alert is always sent, even with no internet or no LLM |
| A problem lasting 3 runs in a row is raised one severity level | Persistent problems matter more than one bad batch |
| On first start, existing MLflow runs are saved as history without alerts | Avoids flooding the user with alerts about the past |

## Done (26–27 Oct)

| Area | Files | Result |
|---|---|---|
| Demo data and baseline | `examples/california_housing/data_prep.py`, `train_baseline.py` | 60/20/20 split; RMSE 0.476, R² 0.826 |
| Fault injection | `examples/california_housing/injection.py` | drift (MedInc +2.0), decay (prices ×1.3), bias (coastal prices ×1.4) |
| Simulator | `examples/california_housing/simulate.py` | replays live data in batches of 500, logs rmse, r2, rmse_gap, PSI to MLflow |
| Retraining | `examples/california_housing/retrain.py` | new model + new reference data for the "new normal" |
| Settings | `src/mlsentry/config.py`, `.env.example` | everything from `.env`, nothing hardcoded |
| Reading MLflow | `src/mlsentry/mlflow_reader.py` | REST API, clear errors when MLflow is down or the experiment is missing |
| History | `src/mlsentry/storage.py` | SQLite: runs, metrics, incidents |
| Checks | `src/mlsentry/checker.py` | MAX/MIN limits, rule-based severity |
| Agent | `src/mlsentry/agent.py` | LangGraph gather → judge → write |
| Alerts | `src/mlsentry/alerts.py` | console, Slack, Telegram; a failed channel never stops ML Sentry |
| Main loop | `src/mlsentry/__main__.py` | `python -m mlsentry` (or `--once`) |
| Docker | `Dockerfile`, `docker-compose.yml` | image builds (294 MB); container tested end to end |
| Tests | `tests/` | 9 tests pass (`python -m pytest -q`) |
| Docs | `docs/installation.md`, `configuration.md`, `running.md`, `scenarios.md`, `troubleshooting.md` | written after each step |

## Test results

Healthy batches: RMSE 0.44-0.50, R² 0.84-0.85, gap ≤ 0.09, PSI ≤ 0.05. Limits used: RMSE 0.60, R² 0.70, gap 0.20, PSI 0.20.

| Scenario | What ML Sentry saw | Alert | After retraining |
|---|---|---|---|
| Healthy | all inside limits | none | n/a |
| Drift | PSI ≈ 2.4-3.0, RMSE ≈ 0.88 | critical | fully recovered → RECOVERED message |
| Decay | RMSE ≈ 0.9, R² ≈ 0.6, PSI normal | moderate, then critical after 3 runs | R² recovers, RMSE ≈ 0.57-0.65 → some batches alert again |
| Bias | coastal-inland gap ≈ 0.6-0.8 | critical | gap ≈ 0.10-0.20, mostly fixed |

Run locally (26 Oct) and inside Docker (27 Oct): both raised the expected alerts within seconds; local Ollama (`llama3.1:8b`) judged most of them.

## Problems found and fixed

| Problem | Fix |
|---|---|
| Windows console could not print MLflow's emoji (`UnicodeEncodeError`) | scripts switch their output to UTF-8 |
| Docker Desktop crashed on start (`dockerInference` file locked) | updated Docker Desktop; fix written in troubleshooting.md |
| Another project's containers (`ragops_mlflow` on port 5000, `ragops_ollama` on 11434) start with Docker and clash with the demo ports | stop them while using ML Sentry, or run MLflow on port 5001 |
| Local Ollama sometimes crashes on the GPU (CUDA error) on first load | not an ML Sentry bug; the rules fallback covers that run |
| The 8b Ollama model once called PSI a "fairness issue" | Groq's larger model is the primary; noted for the viva |

## Left to do

| # | Task | Time |
|---|---|---|
| 1 | ~~Update your own `.env` from `.env.example`~~ done 28 Oct | |
| 2 | ~~Add a Groq key~~ done 28 Oct: model `openai/gpt-oss-120b` (Groq retired `llama-3.3-70b-versatile`) | |
| 3 | ~~Slack incoming webhook~~ done 28 Oct: live drift run, 5/5 critical alerts judged by Groq reached #all-ml-sentry, RECOVERED after retraining | |
| 4 | Stop the ragops containers and do a full Docker demo dry run on the normal ports | 30 min |
| 5 | Learn the code module by module, with quizzes, for the viva | 1.5-2 days |
| 6 | Update the presentation to the final design (Docker layer, MLflow, regression demo) | half a day |

Optional: test Telegram; update the KSCST proposal doc for regression; ask Rishabh and the supervisor
whether it is one repo or two and who presents which part.

Suggested plan: **28 Oct** tasks 1-4 and start learning; **29 Oct** finish learning and update the slides;
**30 Oct** full practice run (demo plus viva questions).

## Phase 2 (next semester)

The agent applies safe fixes itself and reports what it did; Model Card / compliance reporting (EU AI Act, NIST AI RMF);
optional Streamlit dashboard and `pip` package.
