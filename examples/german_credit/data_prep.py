"""
ML Sentry — Phase 1: Data & Baseline Preparation
--------------------------------------------------
Loads the raw German Credit Data (UCI Statlog format), decodes the
categorical attribute codes (A11, A34, A93, ...) into readable values,
derives the protected attributes (age, sex) needed for later fairness
analysis, and produces a clean, encoded, train/test-split dataset ready
for baseline model training.

Run directly:
    python src/data_prep.py

Outputs:
    data/processed/train.csv
    data/processed/test.csv
    data/processed/full_clean.csv   (clean, human-readable version before encoding)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "german.data"
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Column names in the order they appear in german.data (UCI Statlog documentation)
COLUMNS = [
    "checking_account_status", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings_account", "employment_since", "installment_rate_pct",
    "personal_status_sex", "other_debtors", "present_residence_since", "property",
    "age", "other_installment_plans", "housing", "num_existing_credits",
    "job", "num_dependents", "telephone", "foreign_worker", "target",
]

# Decode maps straight from the UCI Statlog German Credit documentation
CHECKING_MAP = {"A11": "< 0 DM", "A12": "0 <= ... < 200 DM", "A13": ">= 200 DM", "A14": "no checking account"}
CREDIT_HISTORY_MAP = {
    "A30": "no credits taken", "A31": "all credits at this bank paid duly",
    "A32": "existing credits paid duly till now", "A33": "delay in paying off in the past",
    "A34": "critical account / other credits elsewhere",
}
PURPOSE_MAP = {
    "A40": "car (new)", "A41": "car (used)", "A42": "furniture/equipment",
    "A43": "radio/television", "A44": "domestic appliances", "A45": "repairs",
    "A46": "education", "A47": "vacation", "A48": "retraining",
    "A49": "business", "A410": "others",
}
SAVINGS_MAP = {
    "A61": "< 100 DM", "A62": "100 <= ... < 500 DM", "A63": "500 <= ... < 1000 DM",
    "A64": ">= 1000 DM", "A65": "unknown / no savings account",
}
EMPLOYMENT_MAP = {
    "A71": "unemployed", "A72": "< 1 year", "A73": "1 <= ... < 4 years",
    "A74": "4 <= ... < 7 years", "A75": ">= 7 years",
}
# Personal status/sex code also encodes marital status; we split out `sex` separately below.
PERSONAL_STATUS_SEX_MAP = {
    "A91": "male : divorced/separated", "A92": "female : divorced/separated/married",
    "A93": "male : single", "A94": "male : married/widowed", "A95": "female : single",
}
OTHER_DEBTORS_MAP = {"A101": "none", "A102": "co-applicant", "A103": "guarantor"}
PROPERTY_MAP = {
    "A121": "real estate", "A122": "building society savings / life insurance",
    "A123": "car or other", "A124": "unknown / no property",
}
OTHER_INSTALLMENT_MAP = {"A141": "bank", "A142": "stores", "A143": "none"}
HOUSING_MAP = {"A151": "rent", "A152": "own", "A153": "for free"}
JOB_MAP = {
    "A171": "unemployed/unskilled non-resident", "A172": "unskilled resident",
    "A173": "skilled employee/official", "A174": "management/self-employed/highly qualified",
}
TELEPHONE_MAP = {"A191": "none", "A192": "yes"}
FOREIGN_WORKER_MAP = {"A201": "yes", "A202": "no"}

DECODE_MAPS = {
    "checking_account_status": CHECKING_MAP,
    "credit_history": CREDIT_HISTORY_MAP,
    "purpose": PURPOSE_MAP,
    "savings_account": SAVINGS_MAP,
    "employment_since": EMPLOYMENT_MAP,
    "personal_status_sex": PERSONAL_STATUS_SEX_MAP,
    "other_debtors": OTHER_DEBTORS_MAP,
    "property": PROPERTY_MAP,
    "other_installment_plans": OTHER_INSTALLMENT_MAP,
    "housing": HOUSING_MAP,
    "job": JOB_MAP,
    "telephone": TELEPHONE_MAP,
    "foreign_worker": FOREIGN_WORKER_MAP,
}

CATEGORICAL_COLS = list(DECODE_MAPS.keys())
NUMERIC_COLS = [
    "duration_months", "credit_amount", "installment_rate_pct",
    "present_residence_since", "age", "num_existing_credits", "num_dependents",
]


def load_raw() -> pd.DataFrame:
    """Load the whitespace-delimited raw file and assign column names."""
    df = pd.read_csv(RAW_PATH, sep=r"\s+", header=None, names=COLUMNS)
    return df


def decode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Replace A## style codes with human-readable category labels."""
    df = df.copy()
    for col, mapping in DECODE_MAPS.items():
        df[col] = df[col].map(mapping)
    return df


def derive_protected_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pull out `sex` as its own protected attribute from personal_status_sex,
    and bucket `age` into two bands for fairness-group comparisons. These
    are the protected attributes Fairlearn will use in Phase 3.
    """
    df = df.copy()
    df["sex"] = df["personal_status_sex"].apply(
        lambda v: "male" if v.startswith("male") else "female"
    )
    df["age_group"] = df["age"].apply(lambda a: "under_40" if a < 40 else "40_and_over")
    return df


def recode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    UCI encodes target as 1 = good credit risk, 2 = bad credit risk.
    Recode to a standard binary classification target: 1 = good, 0 = bad.
    """
    df = df.copy()
    df["target"] = df["target"].map({1: 1, 2: 0})
    return df


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical features for the baseline classifiers.
    Protected attributes (sex, age_group) and raw `age` are kept aside
    separately so they remain easy to reference in later phases, but
    `sex` is also one-hot encoded here since it is a legitimate model
    input in this baseline (consistent with how the literature treats
    this dataset, e.g. Hardt et al. 2021 use the same German Credit
    features including sex).
    """
    encode_cols = [c for c in CATEGORICAL_COLS if c != "personal_status_sex"] + ["sex"]
    df_encoded = pd.get_dummies(df, columns=encode_cols, drop_first=True)
    df_encoded = df_encoded.drop(columns=["personal_status_sex"])
    return df_encoded


def main():
    print(f"Loading raw data from {RAW_PATH} ...")
    df = load_raw()
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

    df = decode_categoricals(df)
    df = derive_protected_attributes(df)
    df = recode_target(df)

    # Save a clean, human-readable version before encoding (useful for
    # inspection and for Phase 2's drift/bias injection script).
    clean_path = OUT_DIR / "full_clean.csv"
    df.to_csv(clean_path, index=False)
    print(f"Saved human-readable cleaned dataset -> {clean_path}")

    df_encoded = one_hot_encode(df)

    # Stratified split on the target, preserving class balance.
    train_df, test_df = train_test_split(
        df_encoded, test_size=0.2, random_state=42, stratify=df_encoded["target"]
    )

    # Standardize numeric features (fit on train only, applied to both) to
    # mitigate scale discrepancies — e.g. credit_amount runs into the
    # thousands while one-hot flags are 0/1. This is what lets Logistic
    # Regression converge cleanly and keeps both baselines comparable.
    scaler = StandardScaler()
    train_df[NUMERIC_COLS] = scaler.fit_transform(train_df[NUMERIC_COLS])
    test_df[NUMERIC_COLS] = scaler.transform(test_df[NUMERIC_COLS])
    models_dir = OUT_DIR.parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, models_dir / "scaler.pkl")

    train_path = OUT_DIR / "train.csv"
    test_path = OUT_DIR / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train set: {len(train_df)} rows -> {train_path}")
    print(f"Test set:  {len(test_df)} rows -> {test_path}")
    print(f"Target balance (full): \n{df['target'].value_counts(normalize=True)}")

    # Sanity check: confirm protected attribute distribution is preserved
    # across the split, since fairness analysis in Phase 3 depends on this.
    print("\nSex distribution — train vs test (by one-hot sex_male column):")
    if "sex_male" in train_df.columns:
        print("  train:", train_df["sex_male"].value_counts(normalize=True).to_dict())
        print("  test: ", test_df["sex_male"].value_counts(normalize=True).to_dict())


if __name__ == "__main__":
    main()
