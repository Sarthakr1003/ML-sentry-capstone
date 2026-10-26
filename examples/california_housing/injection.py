"""
Fault injection for the California Housing demo.

These functions deliberately break a copy of the data so we can show
ML Sentry catching each kind of problem. They are part of the demo only;
ML Sentry itself never changes anyone's data.

  inject_drift  - the incoming feature values move away from what the model learned
  inject_decay  - the real-world relationship changes (prices rise everywhere),
                  so the model's predictions get worse across the board
  inject_bias   - prices change for one group only (e.g. coastal),
                  so the model becomes much worse for that group than for the other
"""

import pandas as pd


def inject_drift(df, feature_col: str, shift_amount: float):
    """
    Returns a copy of df with feature_col shifted by shift_amount.
    Simulates data drift: the incoming population's values for this
    feature no longer match what the model was trained on.
    Example: MedInc + 2.0 means every district looks $20k richer.
    """
    df_drifted = df.copy()
    df_drifted[feature_col] = df_drifted[feature_col] + shift_amount
    return df_drifted


def inject_decay(df, target_col: str, factor: float):
    """
    Returns a copy of df where every true target is multiplied by factor.
    Simulates performance decay (concept drift): the inputs look the same,
    but the right answer has changed, e.g. a 1.3 factor means house prices
    rose 30% everywhere. The old model keeps predicting the old prices.
    """
    df_decay = df.copy()
    df_decay[target_col] = df_decay[target_col] * factor
    return df_decay


def inject_bias(df, target_col: str, group_col: str, group_value: str, factor: float):
    """
    Returns a copy of df where the target is multiplied by factor only for rows
    where group_col == group_value. Rows of the other group are untouched.
    The model's error grows for that one group, so the gap between groups grows.
    """
    df_bias = df.copy()
    mask = df_bias[group_col] == group_value
    df_bias.loc[mask, target_col] = df_bias.loc[mask, target_col] * factor
    print(f"Group '{group_value}': changed {int(mask.sum())} of {len(df_bias)} rows (factor={factor})")
    return df_bias


if __name__ == "__main__":
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent.parent
    df = pd.read_csv(ROOT / "data" / "processed" / "reference.csv")

    # --- Test inject_drift ---
    print("Drift: MedInc mean before", round(df["MedInc"].mean(), 3))
    drifted = inject_drift(df, "MedInc", shift_amount=2.0)
    print("Drift: MedInc mean after ", round(drifted["MedInc"].mean(), 3))
    print("Original untouched:      ", round(df["MedInc"].mean(), 3))
    print()

    # --- Test inject_decay ---
    print("Decay: price mean before", round(df["MedHouseVal"].mean(), 3))
    decayed = inject_decay(df, "MedHouseVal", factor=1.3)
    print("Decay: price mean after ", round(decayed["MedHouseVal"].mean(), 3))
    print()

    # --- Test inject_bias ---
    print("Bias: price mean by region before")
    print(df.groupby("region")["MedHouseVal"].mean().round(3).to_string())
    biased = inject_bias(df, "MedHouseVal", "region", "coastal", factor=1.4)
    print("Bias: price mean by region after")
    print(biased.groupby("region")["MedHouseVal"].mean().round(3).to_string())
