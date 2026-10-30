# ML Sentry: progress so far

Status as of **30 Oct 2026**. Phase 1 deadline: **31 Oct 2026**.
For what ML Sentry is and how to use it, see [README.md](README.md) and [docs/](docs/).

## In one line

Phase 1 (the alert system) is **built, tested, set up with Groq and Slack, and on GitHub**. The paper has been
**submitted** to IITCEE 2027 (Paper ID 1389) and the **Review-2 slides are ready**. What remains is learning the
code for the viva and practising the demo.

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
| Drift measured with **PSI** (via Evidently) | Evidently's default test flagged drift even on healthy data; PSI is clean |
| Bias for regression = **error gap between groups** (coastal vs inland) | No approve/reject in regression, so Fairlearn's demographic parity does not apply |
| Fault injection moved into the demo, not ML Sentry | ML Sentry itself never changes anyone's data |
| Agent order Groq → Ollama → rules | An alert is always sent, even with no internet or no LLM |
| Groq model `openai/gpt-oss-120b` | Groq retired `llama-3.3-70b-versatile` (404 model_not_found) |
| `127.0.0.1` instead of `localhost` for MLflow | On Windows `localhost` tries IPv6 first and reached another project's container |
| A problem lasting 3 runs in a row is raised one severity level | Persistent problems matter more than one bad batch |
| On first start, existing MLflow runs are saved as history without alerts | Avoids flooding the user with alerts about the past |

## Done

| Area | Files | Result |
|---|---|---|
| Demo data and baseline | `examples/california_housing/data_prep.py`, `train_baseline.py` | 60/20/20 split (seed 42); HistGradientBoostingRegressor, RMSE 0.476, R² 0.826; trained on Sarthak's laptop 26 Oct |
| Fault injection | `examples/california_housing/injection.py` | drift (MedInc +2.0), decay (prices ×1.3), bias (coastal prices ×1.4) |
| Simulator | `examples/california_housing/simulate.py` | 8 batches of 500, fault from batch 4, logs 16 metrics per batch to MLflow |
| Retraining | `examples/california_housing/retrain.py` | new model + new reference data for the "new normal" |
| ML Sentry core | `src/mlsentry/` (config, mlflow_reader, storage, checker, agent, alerts, `__main__`) | everything from `.env`, nothing hardcoded |
| Docker | `Dockerfile`, `docker-compose.yml` | image builds (294 MB); container tested end to end |
| Tests | `tests/` | 9 tests pass (`python -m pytest -q`) |
| Docs | `docs/installation.md`, `configuration.md`, `running.md`, `scenarios.md`, `troubleshooting.md` | written after each step |
| Groq + Slack (28 Oct) | your `.env` | live drift run: 5/5 CRITICAL alerts judged by Groq reached #all-ml-sentry, then RECOVERED |
| Paper sections III-VI (28-29 Oct) | `docs/paper/ML_Sentry_Paper_Sections_III-VI.pdf`, `docs/paper/figures/` | IEEE two-column, audited twice against the real run data |
| Team paper audit (29 Oct) | `Capstone_Project_Paper.pdf` (Downloads) | all metric numbers correct; wording fixes listed below |
| Paper submitted (30 Oct) | Microsoft CMT | **IITCEE 2027**, Paper ID **1389**; primary area Applications of AI & ML, secondary Software Engineering. This is a submission, not an acceptance yet |
| Review-2 slides (30 Oct) | `docs/paper/ML_Sentry_Review2.pptx` | 23 slides on the department template, incl. results, Gantt, 12 IEEE references, submission and KSCST screenshots |

## Verified results (from MLflow and SQLite)

Limits: RMSE 0.60, R² 0.70, gap 0.20, PSI 0.20.

| Scenario | Watched metric: healthy → with fault | Detected | Severity (batches 4-8) | After retraining |
|---|---|---|---|---|
| Healthy (8 batches) | RMSE 0.422-0.503, R² 0.822-0.867, gap 0.007-0.142, PSI 0.032-0.062 | 0 false alarms | – | – |
| Drift | max PSI → 2.34-3.00; RMSE 0.875-0.920; R² 0.33-0.46 | 5/5, delay 0 | C C C C C | 8/8 within limits; RECOVERED sent live |
| Decay | RMSE → 0.88-0.99; R² → 0.59-0.63 | 5/5, delay 0 | M M C C C | 3/8 within; RMSE 0.549-0.654 (4 minor, 1 moderate) |
| Bias | gap → 0.58-0.77; R² 0.68-0.72 | 5/5, delay 0 | C C C C C | 7/8 within; 1 minor (gap 0.201) |

Agent: 20 judgements. Groq 5 (7.0-14.7 s, median 12.9), Ollama 13 (20.9-44.3 s, measured on 9 of 13), rules 2
(one Ollama CUDA crash, one cause not recorded). All 18 LLM severities matched the rule severities.
Slack received 7 messages (5 CRITICAL + 2 RECOVERED). MLflow holds 96 finished runs.
Hardware: i5-12450HX, 16 GB RAM, RTX 4050 laptop.

## Known wording errors in the submitted paper (fix in the camera-ready version)

- Abstract: "dynamically set threshold limits" → the limits are fixed in `.env`.
- Abstract/Intro: "alerts within 15 seconds of any defect" → only the Groq alerts (7.0-14.7 s); Ollama took 20.9-44.3 s.
- VI-B: RMSE increased "from" 0.88-0.99 → "to".
- After Fig. 4: R² 0.33-0.72 is the range over all three faults; decay alone is 0.59-0.63.
- Table VII Healthy delay "0" → "–"; Table IX R² 0.82 → 0.822; ε denominator → |L|; "average 1.8 s" was one measurement.
- Also: decay is only partly recovered, both rule fallbacks were not CUDA crashes, section/table/figure numbering,
  missing Conclusion, references.

## Viva quick answers (from this week's questions)

- **Algorithm?** The demo model is a HistGradientBoostingRegressor. ML Sentry detects with PSI (drift), RMSE/R² (decay)
  and the coastal-inland RMSE gap (bias); a LangGraph LLM agent judges severity, with rules as the fallback.
- **What does the model predict?** Median house value of a California district, in $100k (4.5 = $450,000).
- **Did you train it yourself?** Yes, on the laptop (CPU, a few seconds), `train_baseline.py`, seed 42, saved to `models/baseline.joblib`.
- **Subject area:** Applications of Artificial Intelligence and Machine Learning (backup: Software Engineering).

## Problems found and fixed

| Problem | Fix |
|---|---|
| Windows console could not print MLflow's emoji (`UnicodeEncodeError`) | scripts switch their output to UTF-8 |
| Docker Desktop crashed on start (`dockerInference` file locked) | updated Docker Desktop; fix in troubleshooting.md |
| `ragops_mlflow` (port 5000) and `ragops_ollama` (11434) clash with the demo | `ragops_mlflow` stopped (stays stopped); or use port 5001 |
| "MLflow has no experiment named ..." although it exists | `localhost` resolved to IPv6 → another container; use `127.0.0.1` |
| Groq 404 model_not_found | switched to `openai/gpt-oss-120b` |
| Local Ollama sometimes crashes on the GPU (CUDA error) on first load | not an ML Sentry bug; the rules fallback covers that run |
| `pytest` "No module named pytest" | run from the repo folder with the venv active |

## Left to do

| # | Task | Time |
|---|---|---|
| 1 | Learn the code module by module, with quizzes (config → mlflow_reader → checker → storage → agent → alerts → `__main__` → demo scripts) | 1-2 days |
| 2 | Full demo dry run on the normal ports (MLflow, ML Sentry in Docker, one scenario, Slack) | 30 min |
| 3 | Practise the Review-2 presentation; confirm the base paper (SageMaker Model Monitor) and Gantt weeks with the team | 1 hour |
| 4 | When the paper is accepted: replace the submission screenshot with the acceptance letter; fix the wording errors above | later |

Optional: test Telegram; update the KSCST proposal doc for regression; ask Rishabh and the supervisor who presents which part.

## Phase 2 (next semester)

The agent applies safe fixes itself and reports what it did; Model Card / compliance reporting (EU AI Act, NIST AI RMF);
optional Streamlit dashboard and `pip` package.
