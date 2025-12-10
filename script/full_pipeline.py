import pandas as pd
import numpy as np
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression


# ======================================================
# 1. CONFIGURATION
# ======================================================

BASE_FOLDER = "data/raw_data"

plug_experiments = {
    "data5/21-08.xlsx - sheet1.csv",
    "data6/23-09.xlsx - sheet1.csv",
}

WINDOW_SIZE = 100
STRIDE = 20

SENSOR_COLS = [
    "Pressure before pump (Arith. Mean)",
    "Pressure after pump (Arith. Mean)",
    "Differential pressure (Arith. Mean)",
    "Flow rate (Arith. Mean)",
]


# ======================================================
# 2. TIMESTAMP DETECTION
# ======================================================
def detect_time_column(df):
    possible_names = ["Time", "time", "Timestamp", "timestamp", "DateTime", "datetime"]

    for col in df.columns:
        if col in possible_names:
            return col
        if col.lower() in possible_names:
            return col

    return None  # no usable column found


# ======================================================
# 3. LOAD EXPERIMENT
# ======================================================
def load_experiment(path):
    df = pd.read_csv(path)

    # detect timestamp column dynamically
    time_col = detect_time_column(df)

    if time_col is None:
        print(f"⚠ Skipping file (no timestamp column found): {path}")
        return None  # skip file

    # parse time safely
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values(time_col).reset_index(drop=True)

    return df


# ======================================================
# 4. MAKE WINDOWS
# ======================================================
def make_windows(df):
    windows = []
    for start in range(0, len(df) - WINDOW_SIZE, STRIDE):
        windows.append(df.iloc[start:start + WINDOW_SIZE])
    return windows


# ======================================================
# 5. FEATURE EXTRACTION (MISSING COL SAFE)
# ======================================================
def extract_window_features(w):
    feats = {}

    for col in SENSOR_COLS:
        if col not in w.columns:
            feats[f"{col}_mean"] = np.nan
            feats[f"{col}_std"] = np.nan
            feats[f"{col}_min"] = np.nan
            feats[f"{col}_max"] = np.nan
            feats[f"{col}_range"] = np.nan
            feats[f"{col}_slope"] = np.nan
            continue

        s = w[col].values

        feats[f"{col}_mean"] = s.mean()
        feats[f"{col}_std"] = s.std()
        feats[f"{col}_min"] = s.min()
        feats[f"{col}_max"] = s.max()
        feats[f"{col}_range"] = s.max() - s.min()

        X = np.arange(len(s)).reshape(-1, 1)
        lr = LinearRegression().fit(X, s)
        feats[f"{col}_slope"] = lr.coef_[0]

    return feats


# ======================================================
# 6. RECURSIVELY LOAD ALL EXPERIMENTS
# ======================================================
files = glob.glob(BASE_FOLDER + "/**/*.csv", recursive=True)
files = [f.replace("\\", "/") for f in files]

print("\nFound files:")
for f in files:
    print(" -", f)

X_list, y_list, exp_id_list = [], [], []

for path in files:
    df = load_experiment(path)

    if df is None:  # skip files without timestamp column
        continue

    file_id = "/".join(path.split("/")[-2:])

    windows = make_windows(df)
    print(f"\nProcessing {file_id} → windows: {len(windows)}")

    label = 1 if file_id in plug_experiments else 0

    for w in windows:
        feats = extract_window_features(w)
        X_list.append(feats)
        y_list.append(label)
        exp_id_list.append(file_id)


# ======================================================
# 7. FINAL DATASET
# ======================================================
X = pd.DataFrame(X_list).fillna(0)
y = np.array(y_list)

print("\nDataset:")
print(" - Windows:", len(X))
print(" - Plug windows:", y.sum())
print(" - No-plug windows:", len(y) - y.sum())


# ======================================================
# 8. TIME-SERIES SAFE TRAIN/TEST SPLIT
# ======================================================
unique_exps = list(dict.fromkeys(exp_id_list))
num_exps = len(unique_exps)

print("\nExperiments detected:", num_exps)
print(unique_exps)

if num_exps == 1:
    train_exps = set(unique_exps)
    test_exps = set()

elif num_exps == 2:
    train_exps = {unique_exps[0]}
    test_exps = {unique_exps[1]}

else:
    n_train = int(0.7 * num_exps)
    train_exps = set(unique_exps[:n_train])
    test_exps  = set(unique_exps[n_train:])

train_mask = np.array([eid in train_exps for eid in exp_id_list])
test_mask  = np.array([eid in test_exps for eid in exp_id_list])

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print("\nTrain windows:", len(X_train))
print("Test windows:", len(X_test))


# ======================================================
# 9. TRAIN MODEL
# ======================================================
if len(X_train) == 0:
    raise RuntimeError("No training samples available! Check window size or file formats.")

model = RandomForestClassifier(n_estimators=400, random_state=42)
model.fit(X_train, y_train)

print("\nModel trained successfully.")


# ======================================================
# 10. EVALUATE
# ======================================================
if len(X_test) > 0:
    pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, pred))
else:
    print("\nNo test set available.")


# ======================================================
# 11. PREDICT NEW EXPERIMENT
# ======================================================
def predict_experiment(path):
    df = load_experiment(path)
    if df is None:
        raise ValueError("File missing timestamp column.")

    windows = make_windows(df)
    feats = [extract_window_features(w) for w in windows]
    X_new = pd.DataFrame(feats).fillna(0)
    return model.predict_proba(X_new)[:, 1]



import matplotlib.pyplot as plt

def plot_plug_probability(path):
    probs = predict_experiment(path)
    plt.figure(figsize=(12,4))
    plt.plot(probs)
    plt.title(f"Plug probability over time for: {path}")
    plt.xlabel("Window index")
    plt.ylabel("Probability of plug")
    plt.grid(True)
    plt.show()



import matplotlib.pyplot as plt
import numpy as np

def visualize_windows(file_path, window_size=WINDOW_SIZE, stride=STRIDE, columns=None):
    """
    Visualize raw time series with sliding windows marked.
    
    Args:
        file_path: path to experiment CSV
        window_size: number of rows per window (same as model)
        stride: step size between windows (same as model)
        columns: which sensor columns to plot
    """
    
    # Load experiment
    df = load_experiment(file_path)
    if df is None:
        raise ValueError("File missing timestamp column or unreadable.")

    # Use default columns if none specified
    if columns is None:
        columns = SENSOR_COLS

    # We plot only columns that exist
    available_cols = [c for c in columns if c in df.columns]

    # Create windows
    windows = make_windows(df)

    time_col = detect_time_column(df)

    # Plot raw signals
    plt.figure(figsize=(16, 6))
    
    for col in available_cols:
        plt.plot(df[time_col], df[col], label=col)

    # Overlay window blocks
    for start in range(0, len(df) - window_size, stride):
        t_start = df[time_col].iloc[start]
        t_end = df[time_col].iloc[start + window_size - 1]

        # transparent rectangle
        plt.axvspan(t_start, t_end, alpha=0.1, color="red")

    plt.title(f"Sliding windows on raw data: {file_path}")
    plt.xlabel("Time")
    plt.ylabel("Sensor values")
    plt.legend()
    plt.grid(True)
    plt.show()


import matplotlib.pyplot as plt
import numpy as np

def visualize_clean(file_path):
    df = load_experiment(file_path)
    if df is None:
        raise ValueError("Missing timestamp column")

    time_col = detect_time_column(df)

    # Predictions
    probs = predict_experiment(file_path)

    # Window timestamps
    window_times = [
        df[time_col].iloc[start + WINDOW_SIZE // 2]
        for start in range(0, len(df) - WINDOW_SIZE, STRIDE)
    ]

    # ---- Create plot ----
    fig, ax1 = plt.subplots(figsize=(16, 6))

    # Left axis → raw signals
    for col in SENSOR_COLS:
        if col in df.columns:
            ax1.plot(df[time_col], df[col], label=col)

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Sensor values")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # Right axis → plug probability
    ax2 = ax1.twinx()
    ax2.plot(window_times, probs, color="red", linewidth=2, label="Plug probability")
    ax2.set_ylabel("Plug probability (0–1)", color="red")
    ax2.set_ylim([-0.05, 1.05])
    ax2.tick_params(axis='y', labelcolor="red")

    plt.title(f"Raw signals + Plug probability: {file_path}")
    plt.show()


visualize_clean("data/raw_data/data5/21-08.xlsx - sheet1.csv")
visualize_clean("data/raw_data/data6/23-09.xlsx - sheet1.csv")


new_pred = predict_experiment("data/data8/24-01-23-LF.xlsx - sheet1.csv")
print("New experiment predictions (first 10):", new_pred[:10])