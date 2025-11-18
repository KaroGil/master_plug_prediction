import numpy as np 
import pandas as pd

# def sliding_widndow(data, window_size=10, step_size=1):
#     '''Generate sliding windows from time series data.'''

#     windows = []

#     for start in range(0, len(data) - window_size + 1, step_size):
#         end = start + window_size
#         windows.append(data.iloc[start:end].reset_index(drop=False))

#     return windows


# def upsample_minority(X, y, window_size=100, step_size_minority=2, step_size_majority=10):
#     '''Upsample minority class to balance dataset.'''

#     y = y.squeeze()
#     y = pd.Series(y, index=X.index)

#     X_minority = X[y == 1]
#     X_major = X[y == 0]

#     X_minority = sliding_widndow(X_minority, window_size=window_size, step_size=step_size_minority)
#     X_major = sliding_widndow(X_major, window_size=window_size, step_size=step_size_majority)

#     y_minority = [1] * len(X_minority)
#     y_majority = [0] * len(X_major)

#     X_final = X_major + X_minority
#     y_final = y_majority + y_minority

#     return X_final, y_final

# def sliding_window_blocks(df, window, step):
#     data = df.values  # (N, features)
#     n_samples = data.shape[0]
#     windows = []

#     for start in range(0, n_samples - window + 1, step):
#         end = start + window
#         windows.append(data[start:end, :])  # shape (window, features)

#     return np.array(windows)  # shape (num_windows, window, features)


# def upsample_minority(X, y, window=100, step_min=2, step_maj=10):
#     y = y.squeeze()

#     # minority and majority subsets
#     minor = X[y == 1]
#     major = X[y == 0]

#     # sliding windows
#     win_minor = sliding_window_blocks(minor, window, step_min)
#     win_major = sliding_window_blocks(major, window, step_maj)

#     # labels
#     y_minor = np.ones(len(win_minor))
#     y_major = np.zeros(len(win_major))

#     # combine (without shuffle)
#     X_all = np.concatenate([win_major, win_minor], axis=0)
#     y_all = np.concatenate([y_major, y_minor], axis=0)

#     # ---- FLATTEN to make 1 DataFrame ----
#     n_windows, w_size, n_features = X_all.shape

#     X_flat = X_all.reshape(n_windows, w_size * n_features)

#     # meaningful flat column names
#     col_names = [
#         f"{col}_t{t}"
#         for t in range(w_size)
#         for col in X.columns
#     ]

#     X_df = pd.DataFrame(X_flat, columns=col_names)
#     y_df = pd.Series(y_all)

#     return X_df, y_df

import pandas as pd
import numpy as np

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
