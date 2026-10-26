"""
Simulate a live model for the demo.

Replays data/processed/live.csv through the trained model in batches, as if
new labelled traffic were arriving. For each batch it computes evaluation
metrics and logs them to MLflow as one run. ML Sentry only ever sees these
MLflow numbers; it never touches this model or this data.

After --healthy-batches normal batches, the chosen fault is injected into the
remaining batches so the metrics go bad.

Usage (MLflow server must be running, see docs/running.md):
  python examples/california_housing/simulate.py --scenario drift
  python examples/california_housing/simulate.py --scenario healthy --model models/retrained.joblib

Metrics logged per batch:
  rmse, mae, r2         - overall error of the predictions
  rmse_<group>          - error inside each group (coastal / inland)
  rmse_gap              - largest minus smallest group error (our bias signal for regression)
  psi_<feature>         - drift of each feature vs the reference data (Population Stability Index)
  max_psi, drift_share  - the worst feature's PSI, and the share of features that drifted
"""

import argparse
import os
import sys
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from evidently import Report
from evidently.presets import DataDriftPreset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from injection import inject_bias, inject_decay, inject_drift

# MLflow prints emoji; the default Windows console encoding cannot show them
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(os.getenv("MLSENTRY_ENV_FILE", ROOT / ".env"))

DATA_DIR = ROOT / "data" / "processed"
TARGET = os.getenv("DEMO_TARGET_COL", "MedHouseVal")
GROUP = os.getenv("DEMO_GROUP_COL", "region")
BATCH_SIZE = int(os.getenv("DEMO_BATCH_SIZE", "500"))


def inject(batch: pd.DataFrame, scenario: str, args) -> pd.DataFrame:
    """Apply the chosen fault to one batch."""
    if scenario == "drift":
        return inject_drift(batch, args.drift_col, args.drift_shift)
    if scenario == "decay":
        return inject_decay(batch, TARGET, args.decay_factor)
    if scenario == "bias":
        return inject_bias(batch, TARGET, GROUP, args.bias_group, args.bias_factor)
    return batch


def add_fault_args(parser: argparse.ArgumentParser) -> None:
    """How strong each fault is. Shared with retrain.py so both use the same fault."""
    parser.add_argument("--drift-col", default="MedInc")
    parser.add_argument("--drift-shift", type=float, default=2.0)
    parser.add_argument("--decay-factor", type=float, default=1.3)
    parser.add_argument("--bias-group", default="coastal")
    parser.add_argument("--bias-factor", type=float, default=1.4)


def evaluate(model, features: list, batch: pd.DataFrame, reference: pd.DataFrame) -> dict:
    """Score one batch: error metrics, per-group error gap and feature drift."""
    pred = model.predict(batch[features])
    y = batch[TARGET].to_numpy()

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }

    group_rmse = {}
    for name in batch[GROUP].unique():
        rows = (batch[GROUP] == name).to_numpy()
        group_rmse[name] = float(np.sqrt(mean_squared_error(y[rows], pred[rows])))
        metrics[f"rmse_{name}"] = group_rmse[name]
    metrics["rmse_gap"] = max(group_rmse.values()) - min(group_rmse.values())

    # Drift: compare this batch's feature values with the reference data using PSI
    report = Report([DataDriftPreset(method="psi")])
    result = report.run(current_data=batch[features], reference_data=reference[features]).dict()
    psi = {}
    for item in result["metrics"]:
        name = item.get("metric_name", "")
        if name.startswith("ValueDrift(column="):
            column = name.split("column=")[1].split(",")[0]
            psi[column] = float(item["value"])
        elif name.startswith("DriftedColumnsCount"):
            metrics["drift_share"] = float(item["value"]["share"])
    for column, value in psi.items():
        metrics[f"psi_{column}"] = value
    metrics["max_psi"] = max(psi.values())
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay live data through the demo model and log metrics to MLflow")
    parser.add_argument("--scenario", choices=["healthy", "drift", "decay", "bias"], default="healthy")
    parser.add_argument("--healthy-batches", type=int, default=3, help="normal batches before the fault starts")
    parser.add_argument("--delay", type=float, default=3.0, help="seconds to wait between batches")
    parser.add_argument("--model", default="models/baseline.joblib", help="model file, relative to the repo root")
    parser.add_argument("--label", default="", help="extra tag for these runs, e.g. after-retrain")
    add_fault_args(parser)
    args = parser.parse_args()

    saved = joblib.load(ROOT / args.model)
    model, features = saved["model"], saved["features"]
    # A retrained model carries its own reference data (the "new normal"); the baseline uses reference.csv
    reference = pd.read_csv(ROOT / saved.get("reference", "data/processed/reference.csv"))
    live = pd.read_csv(DATA_DIR / "live.csv")

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT"])

    n_batches = len(live) // BATCH_SIZE
    print(f"Scenario '{args.scenario}': {n_batches} batches of {BATCH_SIZE} rows, "
          f"fault starts at batch {args.healthy_batches + 1}")

    for i in range(n_batches):
        batch = live.iloc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE].reset_index(drop=True)
        faulty = args.scenario != "healthy" and i >= args.healthy_batches
        if faulty:
            batch = inject(batch, args.scenario, args)

        metrics = evaluate(model, features, batch, reference)

        run_name = f"{args.label + '-' if args.label else ''}{args.scenario}-batch-{i + 1}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({"scenario": args.scenario, "batch": i + 1, "faulty": faulty,
                             "model": Path(args.model).name, "label": args.label})
            mlflow.log_metrics(metrics)

        print(f"batch {i + 1}/{n_batches} {'FAULT' if faulty else 'ok   '} "
              f"rmse={metrics['rmse']:.3f} r2={metrics['r2']:.3f} "
              f"rmse_gap={metrics['rmse_gap']:.3f} max_psi={metrics['max_psi']:.3f}")
        if i < n_batches - 1:
            time.sleep(args.delay)


if __name__ == "__main__":
    main()
