"""
This script is responsible for training the models on the labeled datasets. 
It loads the datasets, preprocesses the data, and trains the models. 
The script also includes a step to delete any preprocessed datasets in the predict folder to avoid confusion with new datasets. 
The preprocessing includes creating future targets, feature engineering, feature reduction, and splitting the data into training and test sets.
The models are evaluated using the test set, and the results are printed out.
The trained models are then saved for later use in prediction and evaluation.
"""
import os
import glob
import pandas as pd
from script.helper_methods.config import get_config
from script.helper_methods.data_modeling import model_data
from script.helper_methods.data_preprocessing import preprocess_data

# Load config
cfg = get_config()
dataset_nr = cfg['data']['datasets']
LABLED_PATH = cfg['data']['LABELED_PATH']

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

# Train and evaluate models
model_data(X_train, y_train, X_test, y_test)
