"""
Script to prepare the data for training and testing the models. 
It loads the datasets, preprocesses them, and splits them into training and testing sets. 
The preprocessed data is then saved and ready to be used for model training and evaluation.
Saved preprocessed data can be found in "data/preprocessed/".
"""
import pandas as pd
from script.helper_methods.config import get_config
from script.helper_methods.data_preprocessing import preprocess_data

# Load config
cfg = get_config()
dataset_nr = cfg['data']['datasets']
LABLED_PATH = cfg['data']['LABELED_PATH']

datasets = []

# Additional data to include in training
for i in dataset_nr:
    datasets.append(pd.read_csv(LABLED_PATH + f"data{i}.csv"))

X_train, X_test, y_train, y_test = preprocess_data(datasets, [f"data{i}" for i in dataset_nr])
