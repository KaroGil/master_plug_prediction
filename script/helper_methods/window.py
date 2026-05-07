"""
This module contains helper methods for creating windowed features from the raw data.
The main function is prep_window, which takes a dataframe and creates windowed features using 
statistical summaries (mean, std, min, max, slope) for each signal over a specified window size.
The window size is determined by the frequency of the data and a specified window duration in seconds.
The resulting windowed features are returned as a new dataframe, along with the corresponding target values.
"""

import pandas as pd
import numpy as np

from .config import get_config

# Load config
cfg = get_config()
target_col = cfg["data"]["target"]
non_feature_columns = cfg["data"]["non_feature_columns"]
frequency = cfg["data"]["frequency"]

def make_windowed_Xy_stats(df, feature_cols, label_col, window, labels=True):
    """
    Create windowed features using statistical summaries.
    - df: The input dataframe containing the raw data.
    - feature_cols: List of columns to be used as features.
    - label_col: The column to be used as the target variable.
    - window: The size of the window in number of samples (determined by frequency and window duration).
    Returns:
    - X: A dataframe containing the windowed features.
    - y: A series containing the corresponding target values.
    """

    X_rows = []
    y = []
    log_ids = []

    # Loop through the dataframe creating windows
    for i in range(window, len(df)):
        w = df.iloc[i-window:i] # Get the window of data for the current index
        row = []

        for col in feature_cols:
            x = w[col].values
            # Add statistical summaries
            row.extend([
                x.mean(),
                x.std(),
                x.min(),
                x.max(),
                np.polyfit(np.arange(len(x)), x, 1)[0]  # slope
            ])

        # Append the row of features, the corresponding label, and the LogId for the current index
        X_rows.append(row)
        if labels:
            y.append(df.iloc[i][label_col])
        log_ids.append(df.iloc[i]['LogId'])

    # Put everything together
    X = np.asarray(X_rows)
    y = np.asarray(y)

    print(f"Created X with shape {X.shape}")
    print(f"Created y with shape {y.shape}")
    print(f"Window size: {window}")
    print(f"Total signals: {len(feature_cols)}")
    X = pd.DataFrame(X, columns=[f"{col}_{stat}" for col in feature_cols for stat in ["mean", "std", "min", "max", "slope"]])
    X['LogId'] = pd.Series(log_ids)
    
    return X, pd.Series(y)


def prep_window(df, features, window_size=2, labels=True):
    print(f"Preparing windowed features with frequency {frequency} Hz...")
    # Values for windowing
    FS = frequency # Hz
    WINDOW_S = window_size
    W = FS * WINDOW_S

    # Filter out non-feature columns
    features = [col for col in features if col not in non_feature_columns]

    # Make windows
    X, y = make_windowed_Xy_stats(
    df=df,
    feature_cols=features,
    label_col=target_col,
    window=W,
    labels=labels
    )

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    return X, y