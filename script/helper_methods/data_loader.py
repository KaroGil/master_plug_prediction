"""
Helper methods to load raw data, parse time columns, and save datasets.
"""

import os
import glob
import joblib
import pandas as pd
from script.helper_methods.config import get_config

cfg = get_config()
dataset_nr = cfg['data']['datasets']
LABLED_PATH = cfg['data']['LABELED_PATH']


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


def validate_dataset(df, dataset_name, required_cols, numeric_cols, expected_hz, time_col="Time", tolerance=0.1, labels=True):
    """
    Validate dataset for required columns, numeric types, and frequency consistency.
    Raises ValueError if required columns are missing or have wrong types.
    Prints warnings for extra columns and frequency mismatches.
    Returns a list of extra columns that are not required.
    """
    
    # Check required columns exist
    if not labels:
        required_cols = [col for col in required_cols if col not in ["Plug_future", "Plug"]]  # Remove label column from required columns if not present

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset {dataset_name} is missing required columns: {missing_cols}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Check numeric columns are actually numeric
    wrong_types = [col for col in numeric_cols if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])]
    if wrong_types:
        raise ValueError(
            f"Dataset {dataset_name} has non-numeric values in columns: {wrong_types}"
        )

    # Warn about extra columns
    extra_cols = [col for col in df.columns if col not in required_cols]
    if extra_cols:
        print(f"⚠️  Dataset {dataset_name}: unexpected columns will be ignored: {extra_cols}")

    # Frequency check
    if time_col in df.columns:
        times = pd.to_datetime(df[time_col], errors="coerce")
        diffs = times.diff().dropna().dt.total_seconds()
        actual_hz = 1 / diffs.median()
        if abs(actual_hz - expected_hz) > tolerance:
            print(f"⚠️  Dataset {dataset_name}: expected {expected_hz}Hz but detected ~{actual_hz:.2f}Hz")
            print("This dataset may require resampling to the expected frequency for optimal model performance. But will attempt to process it as is.")
    else:
        print(f"⚠️  Dataset {dataset_name}: no time column '{time_col}' found, skipping frequency check")

    print(f"✅ Dataset {dataset_name} validated ({len(df)} rows, {len(df.columns)} columns)")

    return extra_cols


def delete_preprocessed_predict_files(preprocessed_predict_path="./data/processed_data/predict/"):
    """ Delete preprocessed predict datasets to avoid confusion """
    files = glob.glob(os.path.join(preprocessed_predict_path, "data_*.csv"))
    if files:
        for f in files:
            os.remove(f)
        print(f"🗑️ Deleted {len(files)} preprocessed predict file(s).")
    else:
        print("No preprocessed predict files found to delete.")


def load_labeled_dataset(dataset_name, LABLED_PATH=LABLED_PATH):
    """Load a single labeled dataset and validate it"""

    df = pd.read_csv(LABLED_PATH + f"data{dataset_name}.csv")

    # Validate dataset and get any extra columns that are not required
    extra_cols = validate_dataset(
        df=df,
        dataset_name=f"data{dataset_name}",
        required_cols=cfg["data"]["required_columns"],
        numeric_cols=cfg["data"]["numeric_columns"],
        expected_hz=cfg["data"]["frequency"]
    )

    # Drop any extra columns that deviate from the expected schema
    if extra_cols:
        df.drop(columns=extra_cols, inplace=True)

    return df

def load_labeled_datasets(dataset_nr=dataset_nr, LABLED_PATH=LABLED_PATH):
    datasets = []

    # Add data to include in training based on dataset numbers in config
    for i in dataset_nr:
        df = load_labeled_dataset(dataset_name=i, LABLED_PATH=LABLED_PATH)

        datasets.append(df)
    
    return datasets

def keep_relevant_columns(df, dataset_name=None):
    """
    Keep only pressure-related columns and essential metadata columns.
    All other columns (e.g. temperature, flow rate) are dropped.
    
    - `df`: input dataframe
    - `dataset_name`: optional name for logging
    """
    keep_mask = (
        df.columns.str.contains("press", case=False, na=False)
        | df.columns.isin(["Plug", "LogId", "Plug_future", "Time"])
    )
    
    dropped = df.columns[~keep_mask].tolist()
    if dropped and dataset_name:
        print(f"⚠️  {dataset_name}: dropping non-pressure columns: {dropped}")
    
    return df.loc[:, keep_mask]


def add_logId_column(df, log_id):
    '''Add a LogId column to the dataframe to specify which log the data comes from'''
    df = df.copy()
    df['LogId'] = int(log_id)
    return df


def load_raw_data(path="../data/raw_data/data1/*.csv"):
    """Load and concatenate raw data CSV files from a given path"""

    files = sorted(glob.glob(path))
    df_list = [read_unify_data(f) for f in files]
    print(f"Loaded {len(df_list)} files from {path} with shapes {[df.shape for df in df_list]}")
    df = pd.concat(df_list)

    df["Time"] = parse_time_column(df["Time"])

    # Sort by Time 
    df.set_index('Time', inplace=True)
    df.sort_index(inplace=True)

    df["Time"] = df.index  # Move Time back to a column for consistency with expected format

    df.reset_index(drop=True, inplace=True)

    # Drop any columns that are unnamed
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Standardize column names by replacing "Arith. Mean" with "Mean"
    df.columns = df.columns.str.replace("Arith. Mean", "Mean", regex=False)

    # Keep only relevant columns
    df = keep_relevant_columns(df, dataset_name="Loaded Data")

    # Add LogId column based on filename
    df = add_logId_column(df, log_id=path.split("/")[-2][4:])  # Extract dataset number from filename
    print(f"Added LogId column with value: {df['LogId'].iloc[0]}")

    validate_dataset(
        df=df,
        dataset_name="Loaded Data",
        required_cols=cfg["data"]["required_columns"],
        numeric_cols=cfg["data"]["numeric_columns"],
        expected_hz=cfg["data"]["frequency"],
        labels=False
    )

    return df


def load_dataset_artifact(dataset_name: str, base_path: str) -> dict:
    """Load dataset artifact from a joblib file and return the contained data."""

    path = os.path.join(base_path, f"{dataset_name}.joblib")
    artifact = joblib.load(path)
    return artifact


# ---- Saving methods ----


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