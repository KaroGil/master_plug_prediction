import pandas as pd
import numpy as np

def rolling_features(df, window=10, functions=['mean', 'std', 'min', 'max']):
    '''
    Generate rolling features for the given DataFrame.
    '''
    # Create a copy of the DataFrame to avoid modifying the original data
    df_rolling = df.copy()

    # Exclude non-numeric columns
    df_rolling = df_rolling.select_dtypes(include=['number'])

    # Generate rolling features
    for col in df_rolling.columns:
        rolling = df_rolling[col].rolling(window=window, min_periods=1)

        for func in functions:
            col_name = f"{col}_rolling_{func}_{window}"
            df_rolling[col_name] = getattr(rolling, func)().bfill()

    return df_rolling


def augment_minority_continuous_timeseries(X_train, y_train, n_augmentations=3):
    """
    Augment minority class in time-series continuous data by adding Gaussian noise.
    Maintains temporal order without shuffling.
    """
    cols = X_train.columns

    X_minority = X_train[y_train == 1]
    X_majority = X_train[y_train == 0]
    y_minority = y_train[y_train == 1]
    y_majority = y_train[y_train == 0]
    
    # Augment minority
    X_augmented = [X_minority]
    y_augmented = [y_minority]
    
    for _ in range(n_augmentations):
        noise = np.random.normal(0, 0.01 * np.std(X_minority, axis=0), X_minority.shape)
        X_noisy = pd.DataFrame(X_minority.values + noise, columns=cols)
        X_augmented.append(X_noisy)
        y_augmented.append(y_minority)
    
    # Combine majority and augmented minority
    X_balanced = pd.concat([X_majority] + X_augmented, ignore_index=True)
    y_balanced = pd.concat([y_majority] + y_augmented, ignore_index=True)

    return X_balanced, y_balanced