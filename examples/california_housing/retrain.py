"""
Retrain the demo model after a fault, so ML Sentry can re-check it.

In real life, after an alert the team collects fresh labelled data that shows
the "new reality" and retrains on it. Here we fake that: we apply the same
fault to the training data and train a new model on it. The faulty-looking
data is now the new normal, so it also becomes the model's new reference
for drift checks.

Outputs:
  models/retrained.joblib                        - the new model (+ which reference data it uses)
  data/processed/reference_<scenario>.csv        - the new reference data

Then replay the faulty traffic through the new model:
  python examples/california_housing/simulate.py --scenario decay --healthy-batches 0 --model models/retrained.joblib --label after-retrain
"""

import argparse
import json

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from simulate import DATA_DIR, GROUP, ROOT, TARGET, add_fault_args, evaluate, inject

SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain the demo model on data that contains the fault")
    parser.add_argument("--scenario", choices=["drift", "decay", "bias"], required=True)
    add_fault_args(parser)
    args = parser.parse_args()

    train = inject(pd.read_csv(DATA_DIR / "train.csv"), args.scenario, args)
    reference = inject(pd.read_csv(DATA_DIR / "reference.csv"), args.scenario, args)
    features = [c for c in train.columns if c not in (TARGET, GROUP)]

    model = HistGradientBoostingRegressor(random_state=SEED)
    model.fit(train[features], train[TARGET])

    reference_name = f"data/processed/reference_{args.scenario}.csv"
    reference.to_csv(ROOT / reference_name, index=False)
    joblib.dump({"model": model, "features": features, "reference": reference_name},
                ROOT / "models" / "retrained.joblib")

    # Score the new model on the new reference data (it should look healthy again)
    metrics = evaluate(model, features, reference, reference)
    shown = {k: round(v, 4) for k, v in metrics.items() if not k.startswith("psi_")}
    print(f"Retrained for '{args.scenario}'. Metrics on the new reference data:")
    print(json.dumps(shown, indent=2))
    print("Saved models/retrained.joblib")


if __name__ == "__main__":
    main()
