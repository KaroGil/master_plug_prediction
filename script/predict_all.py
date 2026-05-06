"""
This script loads the best model and makes predictions on all datasets, then visualizes the results. 
It is meant to be run after train.py has been run at least once to ensure that the preprocessed datasets and the best model are saved.
The script performs the following steps:
1. Loads the preprocessed datasets for prediction. If they do not exist, it runs the preprocessing pipeline to create them.
2. Loads the best model saved from the training phase.
3. Makes predictions on all datasets and calculates the F1-scores.
4. Visualizes the F1-scores as bar plots.
5. Visualizes the false alarm rates for no-plug datasets.
6. Visualizes the predicted vs true values as line plots for both plug and no-plug datasets.
7. Visualizes a summary of the test datasets.
"""

import joblib
import pandas as pd
from sklearn.metrics import f1_score
from script.helper_methods.config import get_config
from script.preprocess_predict import preped_for_prediction_exists, preprocess_and_save
from script.helper_methods.data_visualization import f1_score_bar_plot, plot_all_predictions, plot_test_data_summary, plot_false_alarm_rates

# Load config
cfg = get_config()
target_col = cfg["data"]["target"]
datasets = cfg["data"]["datasets"]
HORIZON = cfg["experiment"]["horizon"]
test_sets = cfg["data"]["test_sets"]

BASE_PATH = "data/labeled/labeled_"
BASE_PATH_PREPROCESSED_PREDICT = "data/processed_data/predict/"

def predict_all(runId, horizon=HORIZON):
    if preped_for_prediction_exists() and horizon == HORIZON:
        # If preprocessed data for prediction already exists and we are using the default horizon, skip preprocessing and load the data
        print("Preprocessed data already exists. Skipping preprocessing.")
        dataset_ids = []
        X_y_list = []
        for i in datasets:
            dataset_ids.append(i)
            X = pd.read_csv(f"{BASE_PATH_PREPROCESSED_PREDICT}data_{i}_X.csv")
            y = pd.read_csv(f"{BASE_PATH_PREPROCESSED_PREDICT}data_{i}_y.csv").squeeze()
            X_y_list.append((X, y))
    else:
        print("Preprocessed data does not exist. Starting preprocessing.")
        preprocess_and_save(horizon=horizon) #First preprocess and save the data, then load it
        dataset_ids = []
        X_y_list = []
        for i in datasets:
            dataset_ids.append(i)
            X = pd.read_csv(f"{BASE_PATH_PREPROCESSED_PREDICT}data_{i}_X.csv")
            y = pd.read_csv(f"{BASE_PATH_PREPROCESSED_PREDICT}data_{i}_y.csv").squeeze()
            X_y_list.append((X, y))
    
    # Load model
    print("🔮 Loading model and making predictions...")
    model = joblib.load("models/best_model.joblib")
    print("Model used: " + type(model).__name__ + " with parameters: " + str(model.get_params()))

    # Predict 
    print("🖨️ Making predictions on all datasets...")
    y_preds = []
    f1scores = []
    for dataset_id, (X, y) in zip(dataset_ids, X_y_list):
        prediction = model.predict(X)
        y_preds.append(prediction)
        f1 = f1_score(y, prediction, zero_division=0, average='weighted')
        f1scores.append(f1)
        print(f"F1-score for dataset {dataset_id} run was {f1}")

    # Visualize predictions as bar plot
    print("📊 Visualizing f1-scores as bar plots...")
    f1_score_bar_plot(dataset_ids, f1scores, runId)

    # Visualize false alarm rates for no-plug datasets
    print("📊 Visualizing false alarm rates...")
    plot_false_alarm_rates(dataset_ids, X_y_list, y_preds, runId)

    # Visualize as true vs predicted events line plots
    print("📊 Visualizing predicted vs true values...")

    # Separate plug and no-plug datasets for visualization
    all_data = list(zip(dataset_ids, X_y_list, y_preds))
    plug_data = [(id, (X, y), pred) for id, (X, y), pred in all_data if y.unique().size > 1]
    no_plug_data = [(id, (X, y), pred) for id, (X, y), pred in all_data if y.unique().size < 2]
    
    def plot(data, suffix):
        # Unpack data for plotting
        dataset_ids, Xys, preds = zip(*data)
        plot_all_predictions(Xys, preds, dataset_ids, f"{runId}_{suffix}")

    # Plot plug and no-plug datasets separately
    plot([(id, Xy, p) for id, Xy, p in plug_data], suffix="plug")
    plot(no_plug_data, suffix="no_plug")

    # Visualize summary of test datasets
    test_data = [(id, Xy, p) for id, Xy, p in all_data if id in test_sets]
    X_y_list = [Xy for _, Xy, _ in test_data]
    y_preds = [p for _, _, p in test_data]
    dataset_ids = [id for id, _, _ in test_data]

    plot_test_data_summary(X_y_list, y_preds, dataset_ids, runId="test_sets")

if __name__ == "__main__":
    predict_all(runId="default_run")