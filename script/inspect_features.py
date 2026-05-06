"""
Loads the processed dataset joblib file and prints column names,
shapes, and a summary of all features used for model training and prediction.

Usage:
    python inspect_processed_columns.py
    python inspect_processed_columns.py --path data/processed_data/data_4_7_8_9_10_11_12_14_15_16_20_21_24_25.joblib
"""

import argparse
import joblib
import pandas as pd


def inspect_processed_data(path="data/processed_data/data_4_7_8_9_10_11_12_14_15_16_20_21_24_25.joblib"):
    """Load processed joblib file and print column info for X_train and X_test."""

    print(f"Loading: {path}\n")
    try:
        data = joblib.load(path)
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    for split in ["X_train", "X_test"]:
        if split not in data:
            print(f"{split} not found in joblib file.")
            continue

        df = data[split]
        print(f"--- {split} ---")
        print(f"Shape: {df.shape}")
        print(f"Columns ({len(df.columns)}):")
        for col in sorted(df.columns):
            print(f"  - {col}")
        print()

    # Summary of features common to both splits
    if "X_train" in data and "X_test" in data:
        train_cols = set(data["X_train"].columns)
        test_cols = set(data["X_test"].columns)

        only_in_train = train_cols - test_cols
        only_in_test = test_cols - train_cols
        common = train_cols & test_cols

        print(f"{'='*50}")
        print(f"Common features (train & test): {len(common)}")
        print(f"Only in X_train: {only_in_train if only_in_train else 'none'}")
        print(f"Only in X_test:  {only_in_test if only_in_test else 'none'}")

        print(f"\nAll unique features ({len(train_cols | test_cols)} total):")
        for col in sorted(train_cols | test_cols):
            print(f"  - {col}")

    # Print y distribution if available
    for split in ["y_train", "y_test"]:
        if split in data:
            s = pd.Series(data[split])
            print(f"\n--- {split} distribution ---")
            print(s.value_counts().sort_index().to_string())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect columns of processed joblib dataset."
    )
    parser.add_argument(
        "--path",
        default="data/processed_data/data_4_7_8_9_10_11_12_14_15_16_20_21_24_25.joblib",
        help="Path to the processed joblib file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inspect_processed_data(path=args.path)