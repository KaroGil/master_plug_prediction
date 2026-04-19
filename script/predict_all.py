import joblib
import pandas as pd
from sklearn.metrics import f1_score
from script.helper_methods.config import get_config
from script.helper_methods.data_preprocessing import preprocess_data_predict
from script.helper_methods.data_visualization import f1_score_bar_plot, plot_all_predictions, plot_one

# Load config
cfg = get_config()
target_col = cfg["data"]["target"]
datasets = cfg["data"]["datasets"]
horizon = cfg["experiment"]["horizon"]
flow_rate_missing_sets = cfg["data"]["flow_rate_missing"]

BASE_PATH = "data/labeled/labeled_"

def predict_all(runId, samples=horizon):
    # Load data
    print("💾 Loading multiple datasets for prediction...")
    data_list = []
    dataset_ids = []
    for i in datasets:
        if i in [2]:
            continue  # Skip data2 because of its size
        data_list.append(pd.read_csv(BASE_PATH + f"data{i}.csv"))
        dataset_ids.append(i)

    # Preprocess 
    print("🛠️ Preprocessing datasets for prediction...")
    X_y_list = []
    for dataset_id, d in zip(dataset_ids, data_list):
        print(f"🔢 Preprocessing dataset {dataset_id}...")
        preped = preprocess_data_predict(d, dataset_name=f"data{dataset_id}")
        X_y_list.append((preped[0], preped[1], d["Flow rate (Mean)"]))

    # Load model
    print("🔮 Loading model and making predictions...")
    model = joblib.load("models/best_model.joblib")
    print("Model used: " + type(model).__name__ + " with parameters: " + str(model.get_params()))

    # Predict 
    print("🖨️ Making predictions on all datasets...")
    y_preds = []
    f1scores = []
    for dataset_id, (X, y, _) in zip(dataset_ids, X_y_list):
        prediction = model.predict(X)
        y_preds.append(prediction)
        f1 = f1_score(y, prediction, zero_division=0)
        f1scores.append(f1)
        print(f"F1-score for dataset {dataset_id} run was {f1}")

    # Visualize predictions as bar plot
    print("📊 Visualizing f1-scores as bar plots...")
    f1_score_bar_plot(dataset_ids, f1scores, runId)

    # Visualize as true vs predicted events line plots
    print("📊 Visualizing predicted vs true values...")
    print(f"Datasets with flow rate missing: {flow_rate_missing_sets}")
    plot_all_predictions(X_y_list, y_preds, dataset_ids, flow_rate_missing_sets, samples, model, runId) 

if __name__ == "__main__":
    predict_all(runId="default_run")