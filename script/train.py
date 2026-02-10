import yaml
import pandas as pd
from script.helper_methods.data_modeling import model_data
from script.helper_methods.data_preprocessing import preprocess_data

LABLED_PATH = "data/labeled/labeled_"

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

dataset_nr = cfg['data']['datasets']

datasets = []

# Additional data to include in training
for i in dataset_nr:
    datasets.append(pd.read_csv(LABLED_PATH + f"data{i}.csv"))

X_train, X_test, y_train, y_test = preprocess_data(datasets, [f"data{i}" for i in dataset_nr])

model_data(X_train, y_train, X_test, y_test)