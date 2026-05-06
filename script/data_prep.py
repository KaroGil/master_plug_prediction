"""
Script to prepare the data for training and testing the models. 
It loads the datasets, preprocesses them, and splits them into training and testing sets. 
The preprocessed data is then saved and ready to be used for model training and evaluation.
Saved preprocessed data can be found in "data/preprocessed/".
"""
import os
import glob
import pandas as pd
from script.helper_methods.config import get_config
from script.helper_methods.data_preprocessing import preprocess_data

# Load config
cfg = get_config()
dataset_nr = cfg['data']['datasets']
LABLED_PATH = cfg['data']['LABELED_PATH']

def load_and_preprocess_data():
    # Delete preprocessed predict datasets to avoid confusion 
    preprocessed_predict_path = "./data/processed_data/predict/"
    files = glob.glob(os.path.join(preprocessed_predict_path, "data_*.csv"))
    if files:
        for f in files:
            os.remove(f)
        print(f"🗑️ Deleted {len(files)} preprocessed predict file(s).")
    else:
        print("No preprocessed predict files found to delete.")

    datasets = []

    # Add data to include in training based on dataset numbers in config
    for i in dataset_nr:
        datasets.append(pd.read_csv(LABLED_PATH + f"data{i}.csv"))

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(datasets, [f"data{i}" for i in dataset_nr])
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()