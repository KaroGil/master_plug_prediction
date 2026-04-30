import pandas as pd
from script.predict_all import predict_all
from script.helper_methods.config import get_config
from script.helper_methods.data_modeling import model_data
from script.helper_methods.data_preprocessing import preprocess_data
from script.helper_methods.data_visualization import plot_test_f1_vs_horizon, plot_test_f1_vs_horizon_bar


# Load config
cfg = get_config()
dataset_nr = cfg['data']['datasets']
LABLED_PATH = cfg['data']['LABELED_PATH']

datasets = []

# Additional data to include in training
for i in dataset_nr:
    datasets.append(pd.read_csv(LABLED_PATH + f"data{i}.csv"))

freq = 1 # Hz
#horizons = [5, 10, 15, 25, 50]  # seconds
horizons = [1, 10, 50, 100, 500]
horizons = [h * freq for h in horizons]  # convert to number of samples
print(f"Testing with horizons (in samples): {horizons}")

scores = {}

for horizon in horizons:
    print(f"\n\n=== HORIZON: {horizon} samples ===")
    X_train, X_test, y_train, y_test = preprocess_data(datasets, [f"data{i}" for i in dataset_nr], horizon=horizon)

    _, test_f1_score = model_data(X_train, y_train, X_test, y_test, horizon=horizon) 

    summary = pd.read_csv("models/model_comparison_summary.csv")
    scores[horizon] = (summary["Best Validation F1 Score"].iloc[1],summary["Best Validation F1 Score"].iloc[2], test_f1_score)

    predict_all(runId=f"{horizon}_samples", samples=horizon)
    print(f"\n\n=== SUMMARY OF SCORES FOR HORIZON {horizon} SAMPLES ===")
    if scores[horizon][0] > scores[horizon][1]:
        print("Model chosen RF")
    else: 
        print("Model chosen XGB")
    print(f"RF: {scores[horizon][0]}")
    print(f"XGB: {scores[horizon][1]}")
    print(f"Test: {scores[horizon][2]}")

print("\n\n=== SUMMARY OF SCORES ===")
for horizon, score in scores.items():
    print(f"Horizon: {horizon} samples")
    if score[0] > score[1]:
        print("Model chosen RF")
    else: 
        print("Model chosen XGB")
    print(f"RF: {score[0]}")
    print(f"XGB: {score[1]}")
    print(f"Test: {score[2]}")

test_scores = [x[2] for x in scores.values()]

plot_test_f1_vs_horizon(horizons, test_scores)
plot_test_f1_vs_horizon_bar(horizons, test_scores)