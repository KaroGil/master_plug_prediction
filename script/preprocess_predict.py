"""
Script and helper functions for preprocessing the data for prediction. 
Can be run alone or is run as part of predict_all.py.
This is run before predict_all.py to ensure the data is preprocessed and ready for prediction. 
The preprocessed data is saved in data/processed_data/predict/ and can be loaded by predict_all.py for faster testing.
"""
import os

import pandas as pd
from script.helper_methods.config import get_config
from script.helper_methods.data_preprocessing import preprocess_data_predict
from script.helper_methods.data_loader import save_data

# Load config
cfg = get_config()
datasets = cfg["data"]["datasets"]
HORIZON = cfg["experiment"]["horizon"]

BASE_PATH = "data/labeled/labeled_"
BASE_PATH_PREPROCESSED_PREDICT = "data/processed_data/predict/"

def preped_for_prediction_exists(datasets=datasets, base_path=BASE_PATH_PREPROCESSED_PREDICT):
    """
    Checks if the preprocessed data for prediction already exists in the specified path.
    Returns:        
    bool: True if the preprocessed data exists, False otherwise.
    """
    for i in datasets:
        if not all(os.path.exists(f"{base_path}data_{i}_{key}.csv") 
                   for key in ["X", "y"]):
            return False
    return True

def preprocess_and_save(datasets=datasets, horizon=HORIZON):
    """
    Preprocesses the data for prediction and saves the preprocessed data to the specified path.
    - horizon (int): The horizon to use for preprocessing the data. Default is HORIZON from config.
    """
    # Load data
    print("💾 Loading multiple datasets for prediction...")
    data_list = []
    dataset_ids = []
    for i in datasets:
        data_list.append(pd.read_csv(BASE_PATH + f"data{i}.csv"))
        dataset_ids.append(i)

    # Preprocess 
    print("🛠️ Preprocessing datasets for prediction...")
    X_y_list = []
    for dataset_id, d in zip(dataset_ids, data_list):
        print(f"🔢 Preprocessing dataset {dataset_id}...")
        preped = preprocess_data_predict(d, dataset_name=f"data{dataset_id}", horizon=horizon)
        X_y_list.append((preped[0], preped[1]))

    # Save preprocessed data
    for dataset_id, (X, y) in zip(dataset_ids, X_y_list):
        df = X.copy()
        df["target"] = y
        save_data({"X": X, "y": y}, dataset_name=f"data_{dataset_id}", base_path="./data/processed_data/predict/")

    return X_y_list, dataset_ids

def load_preprocessed_data(datasets=datasets, base_path=BASE_PATH_PREPROCESSED_PREDICT):
    """
    Loads the preprocessed data for prediction from the specified path.
    Returns:
    list: A list of tuples containing the preprocessed features (X) and target (y) for each dataset.
    """
    dataset_ids = []
    X_y_list = []
    for i in datasets:
        dataset_ids.append(i)
        X = pd.read_csv(f"{base_path}data_{i}_X.csv")
        y = pd.read_csv(f"{base_path}data_{i}_y.csv").squeeze()
        X_y_list.append((X, y))
    return X_y_list, dataset_ids


if __name__ == "__main__":
    if preped_for_prediction_exists():
        print("Preprocessed data already exists. Skipping preprocessing.")
    else:
        print("Preprocessed data does not exist. Starting preprocessing.")
        preprocess_and_save()
