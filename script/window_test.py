import numpy as np
import pandas as pd
from script.helper_methods.config import get_config
from script.helper_methods.data_modeling import model_data
from script.helper_methods.data_preprocessing import preprocess_data
from script.helper_methods.data_visualization import plot_test_f1_vs_horizon, plot_test_f1_vs_horizon_bar

cfg = get_config()
dataset_nr = cfg['data']['datasets']
LABLED_PATH = cfg['data']['LABELED_PATH']
frequency = cfg["data"]["frequency"]
horizon = cfg["experiment"]["horizon"]

datasets = []
for i in dataset_nr:
    datasets.append(pd.read_csv(LABLED_PATH + f"data{i}.csv"))

window_sizes = [2, 10, 30]
scores = {}

for window_size in window_sizes:
    print(f"\n\n=== WINDOW SIZE: {window_size} samples ({window_size}s at 1Hz) ===")
    
    X_train, X_test, y_train, y_test = preprocess_data(
        datasets, 
        [f"data{i}" for i in dataset_nr], 
        horizon=horizon,
        window_size=window_size
    )

    _, test_f1_score = model_data(X_train, y_train, X_test, y_test, horizon=horizon)

    summary = pd.read_csv("models/model_comparison_summary.csv")
    scores[window_size] = (
        summary["Best Validation F1 Score"].iloc[1],  # RF
        summary["Best Validation F1 Score"].iloc[2],  # XGBoost
        test_f1_score
    )

    print(f"RF val F1:      {scores[window_size][0]:.4f}")
    print(f"XGBoost val F1: {scores[window_size][1]:.4f}")
    print(f"Test F1:        {scores[window_size][2]:.4f}")

# Find best window size based on validation F1
print("\n\n=== SUMMARY ===")
best_val_scores = [max(x[0], x[1]) for x in scores.values()]
best_window_idx = np.argmax(best_val_scores)
best_window = window_sizes[best_window_idx]
best_score = scores[best_window]
best_model_name = "RF" if best_score[0] > best_score[1] else "XGBoost"

for window_size, score in scores.items():
    print(f"Window {window_size}s: RF={score[0]:.4f}, XGB={score[1]:.4f}, Test={score[2]:.4f}")

print(f"\nBest window size: {best_window}s")
print(f"Best model: {best_model_name}")
print(f"Validation F1: {best_val_scores[best_window_idx]:.4f}")
print(f"Test F1: {best_score[2]:.4f}")

# Plot
val_scores = [max(x[0], x[1]) for x in scores.values()]
test_scores = [x[2] for x in scores.values()]

plot_test_f1_vs_horizon(window_sizes, test_scores, test_or_val="Test_window")
plot_test_f1_vs_horizon_bar(window_sizes, test_scores, test_or_val="Test_window")
plot_test_f1_vs_horizon(window_sizes, val_scores, test_or_val="Validation_window")
plot_test_f1_vs_horizon_bar(window_sizes, val_scores, test_or_val="Validation_window")