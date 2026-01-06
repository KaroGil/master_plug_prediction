import pandas as pd
import numpy as np


def make_windowed_Xy_stats(df, feature_cols, label_col, window):
    X_rows = []
    y = []

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

    X = np.asarray(X_rows)
    y = np.asarray(y)

    print(f"Created X with shape {X.shape}")
    print(f"Created y with shape {y.shape}")
    print(f"Window size: {window}")
    print(f"Features per signal: 5")
    print(f"Total signals: {len(feature_cols)}")

    return pd.DataFrame(X, columns=[f"{col}_{stat}" for col in feature_cols for stat in ["mean", "std", "min", "max", "slope"]]), pd.Series(y)





def prep_window(df, features):
    FS = 20
    WINDOW_S = 2
    W = FS * WINDOW_S  # 60

    assert "Plug_future" in df.columns, df.columns
    assert "Flow rate (Mean)" in df.columns, df.columns


    X, y = make_windowed_Xy_stats(
    df=df,
    feature_cols=features,
    label_col="Plug_future",
    window=W
    )


    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")


    return X, y