"""Train the baseline regression model and record its healthy-state metrics.

Outputs (both in models/, which must be gitignored):
  baseline.joblib         - the trained model
  baseline_metrics.json   - RMSE / MAE / R2 on the reference set, overall and per region
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
TARGET = "MedHouseVal"
GROUP = "region"
SEED = 42


def score(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    reference = pd.read_csv(DATA_DIR / "reference.csv")

    features = [c for c in train.columns if c not in (TARGET, GROUP)]

    model = HistGradientBoostingRegressor(random_state=SEED)
    model.fit(train[features], train[TARGET])

    pred = model.predict(reference[features])
    metrics = {"overall": score(reference[TARGET], pred), "by_group": {}}
    for name, part in reference.groupby(GROUP):
        metrics["by_group"][name] = score(part[TARGET], pred[part.index])

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": features}, MODEL_DIR / "baseline.joblib")
    (MODEL_DIR / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
