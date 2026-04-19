import pandas as pd
import numpy as np

from .config import get_config

# Load config
cfg = get_config()
target_col = cfg["data"]["target"]
non_feature_columns = cfg["data"]["non_feature_columns"]

def make_windowed_Xy_stats(df, feature_cols, label_col, window):
    """
    Create windowed features using statistical summaries.
    """

    X_rows = []
    y = []
    log_ids = []

    for i in range(window, len(df)):
        w = df.iloc[i-window:i]
        row = []

        for col in feature_cols:
            x = w[col].values

            row.extend([
                x.mean(),
                x.std(),
                x.min(),
                x.max(),
                np.polyfit(np.arange(len(x)), x, 1)[0]  # slope
            ])

        X_rows.append(row)
        y.append(df.iloc[i][label_col])
        log_ids.append(df.iloc[i]['LogId'])

    X = np.asarray(X_rows)
    y = np.asarray(y)

    print(f"Created X with shape {X.shape}")
    print(f"Created y with shape {y.shape}")
    print(f"Window size: {window}")
    print(f"Total signals: {len(feature_cols)}")
    X = pd.DataFrame(X, columns=[f"{col}_{stat}" for col in feature_cols for stat in ["mean", "std", "min", "max", "slope"]])
    X['LogId'] = pd.Series(log_ids)

    return X, pd.Series(y)


def prep_window(df, features):
    FS = 2 # Hz
    WINDOW_S = 2
    W = FS * WINDOW_S

    features = [col for col in features if col not in non_feature_columns]

    X, y = make_windowed_Xy_stats(
    df=df,
    feature_cols=features,
    label_col=target_col,
    window=W
    )

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    return X, y