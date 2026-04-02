import math

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from script.helper_methods.data_preprocessing import preprocess_data_predict
from script.helper_methods.data_visualization import plot_one
from script.helper_methods.config import get_config

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
            continue  # Skip data2
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
    for dataset_id, (X, y, _) in zip(dataset_ids, X_y_list):
        prediction = model.predict(X)
        y_preds.append(prediction)
        print(f"F1-score for dataset {dataset_id} run was {f1_score(y, prediction, zero_division=0)}")


    # Visualize 
    print("📊 Visualizing predicted vs true values...")

    n_plots = len(X_y_list)  
    ncols = 5                
    nrows = math.ceil(n_plots / ncols)

    plt.figure(figsize=(4*ncols, 3.2*nrows))  

    for subplot_idx, ((X, y, flow), y_pred, dataset_id) in enumerate(zip(X_y_list, y_preds, dataset_ids), start=1):
        X["flow_rate"] = flow
        print(f"Dataset {dataset_id} has flow rate missing: {dataset_id in flow_rate_missing_sets}")
        plot_one(X, y_pred, subplot_idx, y, nrows, ncols, dataset_id=dataset_id, show_flow=(dataset_id not in flow_rate_missing_sets))
   
    plt.subplots_adjust(hspace=0.5)

    # Save the plot
    plt.suptitle(f"Predicted vs True {target_col}=1 Events for {samples} samples using {str(type(model).__name__)}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"plots/{runId}.png", dpi=300)
    print(f"Plot saved as plots/{runId}.png")
    plt.close()

if __name__ == "__main__":
    predict_all(runId="default_run")