"""
Loads all 25 raw datasets and prints column names for each,
followed by a summary of all unique columns across all datasets.

Usage:
    python inspect_columns.py
    python inspect_columns.py --base-path /custom/path/to/raw_data
    python inspect_columns.py --num-datasets 10
"""

import glob
import argparse
import pandas as pd

def read_unify_data(f):
    return pd.read_csv(f)

def parse_time_column(col):
    return pd.to_datetime(col, errors="coerce")

def standardize_column_names(df):
    df.columns = df.columns.str.strip()
    return df


def load_data(path="data/raw_data/data1/*.csv", reconstruct_time=False, start_time=None, freq_Hz=2):
    '''Load and concatenate CSV files from a given path'''

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

    return df


def load_all_datasets_and_print_columns(base_path="data/raw_data", num_datasets=25):
    """Load all datasets and print column names for each."""
    
    all_columns = {}
    
    for i in range(1, num_datasets + 1):
        path = f"{base_path}/data{i}/*.csv"
        print(f"\n--- Dataset {i} ---")
        try:
            df = load_data(path=path)
            cols = df.columns.tolist()
            all_columns[f"data{i}"] = cols
            print(f"Columns ({len(cols)}): {cols}")
        except Exception as e:
            print(f"Failed to load: {e}")
            all_columns[f"data{i}"] = None
    
    # Print summary of unique columns across all datasets
    all_unique = set(col for cols in all_columns.values() if cols for col in cols)
    print(f"\n{'='*50}")
    print(f"All unique column names across all datasets ({len(all_unique)} total):")
    for col in sorted(all_unique):
        print(f"  - {col}")
    
    return all_columns

def parse_args():
    parser = argparse.ArgumentParser(
        description="Print column names for all raw datasets."
    )
    parser.add_argument(
        "--base-path",
        default="data/raw_data",
        help="Root folder containing data1/…data25/ subdirectories "
             "(default: ../data/raw_data)",
    )
    parser.add_argument(
        "--num-datasets",
        type=int,
        default=25,
        help="How many datasets to load (default: 25)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Base path : {args.base_path}")
    print(f"Datasets  : 1 – {args.num_datasets}")
    load_all_datasets_and_print_columns(
        base_path=args.base_path,
        num_datasets=args.num_datasets,
    )