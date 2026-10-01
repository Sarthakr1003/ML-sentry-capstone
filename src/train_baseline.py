"""
ML Sentry — Phase 1: Baseline Model Training
-----------------------------------------------
Trains two baseline classifiers (Logistic Regression and Random Forest)
on the preprocessed German Credit Data, evaluates them on the held-out
test set, and serializes both models + the exact feature column order
to disk. These become the fixed "deployed models" that every later
phase (drift injection, monitoring, agentic reasoning) treats as the
stable reference point.

Run directly (after src/data_prep.py has been run once):
    python src/train_baseline.py
"""

import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_split():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    return train_df, test_df


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "auc_roc": roc_auc_score(y_test, probs),
    }
    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")
    return metrics


def main():
    train_df, test_df = load_split()

    # The columns `age_group` (string) is a convenience column for fairness
    # analysis in later phases — not a model input. Drop it (and `age` stays
    # in as a numeric feature; `age_group` is just a derived bucket of it).
    drop_cols = ["target", "age_group"]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_test, y_test = test_df[feature_cols], test_df["target"]

    print(f"Training on {len(X_train)} rows, {len(feature_cols)} features.")

    # --- Logistic Regression: simple, interpretable baseline ---
    log_reg = LogisticRegression(max_iter=2000, random_state=42)
    log_reg.fit(X_train, y_train)
    lr_metrics = evaluate("Logistic Regression", log_reg, X_test, y_test)

    # --- Random Forest: higher-capacity comparison model ---
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)
    rf_metrics = evaluate("Random Forest", rf, X_test, y_test)

    # Save both models, the exact feature column order (critical — Phase 2's
    # perturbed batches must be built with these same columns), and the
    # evaluation metrics as the recorded baseline for later comparison.
    joblib.dump(log_reg, MODEL_DIR / "logistic_regression.pkl")
    joblib.dump(rf, MODEL_DIR / "random_forest.pkl")

    with open(MODEL_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    baseline_metrics = {"logistic_regression": lr_metrics, "random_forest": rf_metrics}
    with open(MODEL_DIR / "baseline_metrics.json", "w") as f:
        json.dump(baseline_metrics, f, indent=2)

    print(f"\nSaved models to {MODEL_DIR}/")
    print("  - logistic_regression.pkl")
    print("  - random_forest.pkl")
    print("  - feature_columns.json  (exact column order, needed by Phase 2+)")
    print("  - baseline_metrics.json (recorded baseline performance)")


if __name__ == "__main__":
    main()
