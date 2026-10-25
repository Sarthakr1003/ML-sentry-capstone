import pandas as pd


def inject_drift(df, feature_col: str, shift_amount: float):
    """
    Returns a copy of df with feature_col shifted by shift_amount.
    Simulates data drift: the incoming population's values for this
    feature no longer match what the model was trained on.
    """
    df_drifted = df.copy()
    df_drifted[feature_col] = df_drifted[feature_col] + shift_amount
    return df_drifted


def inject_bias(df, protected_col: str, group_value: str, flip_fraction: float):
    """
    Returns a copy of df where flip_fraction of rows matching
    protected_col == group_value (and currently target == 1, good risk)
    have their target flipped to 0 (bad risk).
    Rows NOT matching group_value are left completely untouched.
    """
    df_bias = df.copy()

    mask = (df_bias["target"] == 1) & (df_bias[protected_col] == group_value)
    eligible_indices = df_bias[mask].index

    n_to_flip = int(len(eligible_indices) * flip_fraction)
    flip_indices = pd.Series(eligible_indices).sample(n=n_to_flip, random_state=42)

    df_bias.loc[flip_indices, "target"] = 0

    print(f"Group '{group_value}': {len(eligible_indices)} eligible rows, flipped {n_to_flip}")
    return df_bias


def inject_decay(df, noise_fraction: float):
    """
    Returns a copy of df where noise_fraction of rows have their
    target label randomly flipped (regardless of group), simulating
    general performance decay - the model just getting worse at
    its job, independent of any specific feature or group.
    """
    df_decay = df.copy()

    n_to_flip = int(len(df_decay) * noise_fraction)
    flip_indices = df_decay.sample(n=n_to_flip, random_state=42).index

    df_decay.loc[flip_indices, "target"] = 1 - df_decay.loc[flip_indices, "target"]

    print(f"Flipped {n_to_flip} of {len(df_decay)} rows (noise_fraction={noise_fraction})")
    return df_decay


if __name__ == "__main__":
    df = pd.read_csv("data/processed/full_clean.csv")

    # --- Test inject_drift ---
    print("Before:", df["credit_amount"].mean())
    drifted = inject_drift(df, "credit_amount", shift_amount=2000)
    print("After:", drifted["credit_amount"].mean())
    print("Original untouched:", df["credit_amount"].mean())

    print()

    # --- Test inject_bias ---
    print("Before bias injection:")
    print(df.groupby("sex")["target"].mean())

    biased = inject_bias(df, protected_col="sex", group_value="female", flip_fraction=0.3)

    print("\nAfter bias injection:")
    print(biased.groupby("sex")["target"].mean())

    print("\nOriginal untouched:")
    print(df.groupby("sex")["target"].mean())

    print()

    # --- Test inject_decay ---
    print("Before decay injection:")
    print("Target proportion:", df["target"].mean())

    decayed = inject_decay(df, noise_fraction=0.15)

    print("\nAfter decay injection:")
    print("Target proportion:", decayed["target"].mean())
    print("Rows changed:", (df["target"] != decayed["target"]).sum())

    print("\nOriginal untouched:")
    print("Target proportion:", df["target"].mean())