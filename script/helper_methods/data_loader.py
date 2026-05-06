"""
Helper methods to load raw data, standardize column names, parse time columns, and save datasets ready for labeling.
"""

import os
import glob
import joblib
import pandas as pd

def read_unify_data(path="../data/raw_data/data1/*.csv"):
    """Read a CSV file with unified separator and decimal"""

    with open(path, 'r') as file:
        heaeder_line = file.readline()

    sep = ';' if ';' in heaeder_line else ','

    decimal = '.'

    with open(path, 'r') as file:
        for _ in range(5):
            line = file.readline()
            if ',' in line and sep == ';':
                decimal = ','
                break
            if '.' in line and sep == ',':
                decimal = '.'
                break
    
    return pd.read_csv(path, sep=sep, decimal=decimal)

# Only useful for aligning mean columns when there are multiple files with different column names, but not needed for the current dataset.
COLUMN_RENAME_MAP = {
    "Time": "Time",
    "Flow rate (Arith. Mean)": "Flow rate (Mean)",
    "Pressure before pump (Arith. Mean)": "TS inlet pressure (Mean)",
    "Pressure before pump (Arith. Mean)": "TS outlet pressure (Mean)",
    "Pressure after pump (Arith. Mean)": "Pump outlet pressure (Mean)",
    "Temperature TS inlet (Arith. Mean)": "Temperature TS inlet (Mean)",
    "Temperature TS outlet (Arith. Mean)": "Temperature TS outlet (Mean)",
    "Tank temperature (Arith. Mean)": "Tank temperature (Mean)",
    "Bypass temperature (Arith. Mean)": "Bypass temperature (Mean)",
    "Differential pressure (Arith. Mean)": "Differential pressure (Mean)",
}#TODO

def standardize_column_names(df):
    """Standardize column names using a predefined mapping to ensure consistency across different files."""
    df = df.rename(columns=COLUMN_RENAME_MAP)
    return df


def parse_time_column(time_column):
    """Parse time column with various formats and return a datetime Series."""
    s = time_column.astype(str).str.strip()

    valid = s[s.notna() & (s != '')]
    if valid.empty:
        return pd.to_datetime(s, errors="coerce")

    first_valid = valid.iloc[0]
    s = s.str.replace(",", ".", regex=False)

    # Full datetime
    if " " in first_valid:
        if "/" in first_valid:
            return pd.to_datetime(s, format="%m/%d/%Y %H:%M:%S.%f", errors="coerce")
        else:
            return pd.to_datetime(s, format="%d.%m.%Y %H:%M:%S.%f", errors="coerce")

    # Date only
    if ":" not in first_valid:
        if "/" in first_valid:
            return pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")
        else:
            return pd.to_datetime(s, format="%d.%m.%Y", errors="coerce")

    # Time only
    return pd.to_datetime(s, format="%H:%M:%S.%f", errors="coerce")


def load_raw_data(path="../data/raw_data/data1/*.csv", reconstruct_time=False, start_time=None, freq_Hz=2):
    """Load and concatenate CSV files from a given path"""

    files = sorted(glob.glob(path))
    df_list = [read_unify_data(f) for f in files]
    df = pd.concat(df_list)

    if reconstruct_time:
        if start_time is None:
            start_time = "1900-01-01 00:00:00"
        else:
            start_time = pd.Timestamp(start_time)

        df["Time"] = pd.date_range(start=start_time, periods=len(df), freq = f"{int(10000 / freq_Hz)}ms")
    else:
        df["Time"] = parse_time_column(df["Time"])

    df.set_index('Time', inplace=True)
    df.sort_index(inplace=True)

    df = standardize_column_names(df)

    # Drop any columns that are unnamed
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    return df


def load_data(path="../data/raw_data/data1/*.csv"):
    """Load and concatenate CSV files from a given path"""

    files = sorted(glob.glob(path))
    
    df_list = [read_unify_data(f) for f in files]

    df = pd.concat(df_list)

    df["Time"] = parse_time_column(df["Time"])

    # Sort by Time 
    df.set_index('Time', inplace=True)
    df.sort_index(inplace=True)

    df.reset_index(drop=True, inplace=True)

    # Use column mapping to standardize column names across different files, if needed
    df = standardize_column_names(df) #Not useful for current datasets with 1Hz

    # Drop any columns that are unnamed
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    return df


def save_data(data: dict, dataset_name: str, base_path="../data/processed_data/"):
    """Save datasets to CSV files"""

    os.makedirs(base_path, exist_ok=True)
    for key, df in data.items():
        df.to_csv(f"{base_path}{dataset_name}_{key}.csv", index=False)
    print(f"💾 Saved data with base name: {dataset_name} to location {base_path}")


def save_dataset_artifact(data: dict, dataset_name: str, base_path: str):
    """Save dataset artifact to a joblib file, including training and test sets, feature names, and dataset name"""

    os.makedirs(base_path, exist_ok=True)

    artifact = {
        "X_train": data["X_train"],
        "X_test": data["X_test"],
        "y_train": data["y_train"],
        "y_test": data["y_test"],
        "feature_names": list(data["X_train"].columns),
        "dataset_name": dataset_name,
    }

    path = os.path.join(base_path, f"{dataset_name}.joblib")
    joblib.dump(artifact, path, compress=3)
    print(f"💾 Saved dataset artifact: {path}")
    return path


def load_dataset_artifact(dataset_name: str, base_path: str) -> dict:
    """Load dataset artifact from a joblib file and return the contained data."""

    path = os.path.join(base_path, f"{dataset_name}.joblib")
    artifact = joblib.load(path)
    return artifact