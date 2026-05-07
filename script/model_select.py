"""
Script to select the best model for the dataset given preprocessed data is present. 
The selected model is saved for future use.
"""

import os
import joblib
from script.helper_methods.data_modeling import model_data

# Find the latest dataset artifact
basePath = "./data/processed_data/"
latest = joblib.load(os.path.join(basePath, "LATEST.joblib"))
DATASET_PATH = latest["artifact_path"]

# Load the dataset artifact
artifact = joblib.load(DATASET_PATH)
print("Using dataset:", DATASET_PATH)

# Extract the training and testing data from the artifact
X_train = artifact["X_train"]
y_train = artifact["y_train"]
X_test = artifact["X_test"]
y_test = artifact["y_test"]

# Run the model selection process
model_data(X_train, y_train, X_test, y_test)