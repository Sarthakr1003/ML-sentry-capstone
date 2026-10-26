"""Prepare the California Housing demo data.

Splits the data into three parts:
  train.csv      - used to train the baseline model
  reference.csv  - held-out data with true labels; the "healthy" reference for drift and metrics
  live.csv       - the stream the simulator replays to mimic production traffic

Also adds a `region` column (coastal / inland) used for the group error-gap (bias) check.
"""
from pathlib import Path

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
TARGET = "MedHouseVal"
SEED = 42


def main() -> None:
    df = fetch_california_housing(as_frame=True).frame

    # Simple demo proxy: districts west of this longitude are treated as coastal.
    df["region"] = (df["Longitude"] < -121.0).map({True: "coastal", False: "inland"})

    train, rest = train_test_split(df, test_size=0.4, random_state=SEED)
    reference, live = train_test_split(rest, test_size=0.5, random_state=SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT_DIR / "train.csv", index=False)
    reference.to_csv(OUT_DIR / "reference.csv", index=False)
    live.to_csv(OUT_DIR / "live.csv", index=False)

    print(f"train {len(train)}, reference {len(reference)}, live {len(live)}")
    print(df["region"].value_counts().to_string())


if __name__ == "__main__":
    main()
