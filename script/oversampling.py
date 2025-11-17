import numpy as np

def create_windows(X, y, window_size=200, step_size=50):
    X_windows = []
    y_windows = []

    for start in range(0, len(X) - window_size + 1, step_size):
        end = start + window_size
        window_X = X[start:end]
        window_y = y[start:end]

        label = 1 if np.any(window_y == 1) else 0

        X_windows.append(window_X)
        y_windows.append(label)

    return np.array(X_windows), np.array(y_windows)


def extract_features(X_windows):
    feature_list = []

    for w in X_windows:
        f = []
 
        f.extend(np.mean(w, axis=0))
        f.extend(np.std(w, axis=0))
        f.extend(np.min(w, axis=0))
        f.extend(np.max(w, axis=0))

        f.extend(np.mean(np.diff(w, axis=0), axis=0))
        f.extend(np.std(np.diff(w, axis=0), axis=0))

        feature_list.append(f)

    return np.array(feature_list)


def duplicate_minority_class(X, y, multiplier = 3):
    '''
    Oversample minority class for imbalanced time-series data. 
    Duplicates each minority class instance 'multiplier' times.
    '''

    X_out = []
    y_out = []

    for w, label in zip(X, y):
        X_out.append(w)
        y_out.append(label)

        # Oversample minority class (Plug = 1)
        if label == 1: 
            for _ in range(multiplier - 1):
                X_out.append(w)
                y_out.append(label)


    return np.array(X_out), np.array(y_out)


def oversampling_pipeline(X, y, window_size=200, step_size=50, multiplier=3):
    '''
    Complete oversampling pipeline: create windows, extract features, oversample minority class.
    '''

    X_windows, y_windows = create_windows(X, y, window_size, step_size)
    X_features = extract_features(X_windows)
    X_resampled, y_resampled = duplicate_minority_class(X_features, y_windows, multiplier)

    return X_resampled, y_resampled