import os
import joblib
from script.helper_methods.data_modeling import model_data

basePath = "./data/processed_data/"
latest = joblib.load(os.path.join(basePath, "LATEST.joblib"))
DATASET_PATH = latest["artifact_path"]

artifact = joblib.load(DATASET_PATH)
print("Using dataset:", DATASET_PATH)

X_train = artifact["X_train"]
y_train = artifact["y_train"]
X_test = artifact["X_test"]
y_test = artifact["y_test"]

model_data(X_train, y_train, X_test, y_test)