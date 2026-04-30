import os

import pandas as pd
from script.helper_methods.config import get_config
from script.helper_methods.data_preprocessing import preprocess_data_predict
from script.helper_methods.data_loader import save_data

# Load config
cfg = get_config()
datasets = cfg["data"]["datasets"]
horizon = cfg["experiment"]["horizon"]
test_sets = cfg["data"]["test_sets"]

BASE_PATH = "data/labeled/labeled_"
BASE_PATH_PREPROCESSED_PREDICT = "data/processed_data/predict/"

def preped_for_prediction_exists():
    for i in datasets:
        if i in [2]:
            continue
        if not all(os.path.exists(f"{BASE_PATH_PREPROCESSED_PREDICT}data_{i}_{key}.csv") 
                   for key in ["X", "y", "flow_rate"]):
            return False
    return True

def preprocess_and_save(runId="default_run"):
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


    for dataset_id, (X, y, flow_rate) in zip(dataset_ids, X_y_list):
        df = X.copy()
        df["target"] = y
        df["flow_rate"] = flow_rate
        save_data({"X": X, "y": y, "flow_rate": flow_rate}, dataset_name=f"data_{dataset_id}", base_path="./data/processed_data/predict/")

if __name__ == "__main__":
    if preped_for_prediction_exists():
        print("Preprocessed data already exists. Skipping preprocessing.")
    else:
        print("Preprocessed data does not exist. Starting preprocessing.")
        preprocess_and_save()


