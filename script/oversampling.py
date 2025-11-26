import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler

def window_data(df, window_seconds = 10, sampling_rate = 0.05):
    window_size = int(window_seconds / sampling_rate)

    feature_cols = [
        col for col in df.columns if col not in ['Anomaly', 'Plug', 'Plug_future', "Elapsed_seconds"]
    ]

    X_windows = []
    y_windows = []

    for i in range(len(df) - window_size):
        window = df.iloc[i:i + window_size][feature_cols].values
        label = df.iloc[i + window_size]['Plug_future']

        X_windows.append(window)
        y_windows.append(label)
    
    X_windows = np.array(X_windows)
    y_windows = np.array(y_windows)

    return X_windows, y_windows

def oversample_within_windows(X_windows, y_windows):
    n_samples, win_len, n_features = X_windows.shape
    X_flat = X_windows.reshape(n_samples, win_len * n_features)

    ros = RandomOverSampler(sampling_strategy='auto')
    X_balanced, y_balanced = ros.fit_resample(X_flat, y_windows)

    return X_balanced, y_balanced

def oversample_minority(X: pd.DataFrame,
                                y: pd.Series,
                                target_ratio: float = 1.0,
                                random_state: int = 42):
    """
    Upsample minority class (label == 1) directly on time-step level, no windowing.

    - X, y: train split only (pandas DataFrame + Series)
    - target_ratio: desired minority/majority ratio.
        - 1.0  -> make class 1 as frequent as class 0
        - 0.5  -> class 1 has 50% of class 0, etc.
    - Keeps original time order of *real* samples.
    - Duplicated rows are added but sorted back by index.

    Returns:
        X_up, y_up  (both pandas, same types as input)
    """
    print("🪜 Oversampling minority class to balance dataset...")
    # make sure y is 1D Series and index aligned with X
    y = y.squeeze()
    y = pd.Series(y, index=X.index)

    if y.nunique() != 2:
        print("y must be binary for oversampling.")
        return X, y

    # masks
    minor_mask = (y == 1)
    majority_mask = ~minor_mask

    X_minor = X[minor_mask]
    X_major = X[majority_mask]
    y_minor = y[minor_mask]

    n_major = len(X_major)
    n_minor = len(X_minor)

    desired_n_minor = int(target_ratio * n_major)

    if desired_n_minor <= n_minor:
        return X.copy(), y.copy()

    n_to_add = desired_n_minor - n_minor

    # sample extra minority rows WITH replacement TODO: check if this is ok, bc of time series
    extra_X_minor = X_minor.sample(n=n_to_add,
                                   replace=True,
                                   random_state=random_state)

    extra_y_minor = y_minor.loc[extra_X_minor.index]

    X_up = pd.concat([X, extra_X_minor], axis=0).sort_index()
    y_up = pd.concat([y, extra_y_minor], axis=0).sort_index()

    print(f" - Original minority samples: {n_minor}, majority samples: {n_major}")
    print(f" - After oversampling: minority samples: {len(X_up[y_up==1])}, majority samples: {len(X_up[y_up==0])}")

    print("⚖️ Oversampling completed.")

    return X_up, y_up
