import pandas as pd

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
