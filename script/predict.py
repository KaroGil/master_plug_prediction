"""
Script to load the best model and make predictions on a specified dataset. 
The dataset can be specified as a command line argument, otherwise it defaults to the first dataset in the config.
The script also visualizes the predicted vs true values and prints out the unique predictions and their counts.
"""
import sys
import numpy as np
import pandas as pd
from script.helper_methods.config import get_config
from script.helper_methods.model_io import load_model
from script.helper_methods.data_visualization import visualize_predicted_vs_true
from script.preprocess_predict import preped_for_prediction_exists, preprocess_and_save

#TODO check this script
# Load config
cfg = get_config()
target_col = cfg["data"]["target"]
datasets = cfg["data"]["datasets"]

BASE_PATH = "data/labeled/labeled_"
BASE_PATH_PREPROCESSED_PREDICT = "data/processed_data/predict/"

# Load data
csv_file = datasets[0] if len(sys.argv) < 2 else sys.argv[1]
model_name = "best_model" if len(sys.argv) < 3 else sys.argv[2]
print(f"Loading data from: {csv_file}")

if preped_for_prediction_exists([csv_file[4:]]):
        # If preprocessed data for prediction already exists and we are using the default horizon, skip preprocessing and load the data
        print("Preprocessed data already exists. Skipping preprocessing.")
        X = pd.read_csv(f"{BASE_PATH_PREPROCESSED_PREDICT}data_{csv_file[4:]}_X.csv")
        y = pd.read_csv(f"{BASE_PATH_PREPROCESSED_PREDICT}data_{csv_file[4:]}_y.csv").squeeze()
else:
    print("Preprocessed data does not exist. Starting preprocessing.")
    preprocess_and_save([csv_file[4:]]) #First preprocess and save the data, then load it
    X = pd.read_csv(f"{BASE_PATH_PREPROCESSED_PREDICT}data_{csv_file[4:]}_X.csv")
    y = pd.read_csv(f"{BASE_PATH_PREPROCESSED_PREDICT}data_{csv_file[4:]}_y.csv").squeeze()

# Load model and make predictions
model = load_model(f"models/{model_name}.joblib")
print("Model used: " + type(model).__name__ + " with parameters: " + str(model.get_params()))
predictions = model.predict(X)

# Visualize predictions as line plot comparing the predicted vs true values
print("Visualizing predicted vs true values...")
X[target_col] = y
visualize_predicted_vs_true(X, predictions, model_name=type(model).__name__)

# Print unique predictions and their counts
print("Unique predictions:")
unique_vals, counts = np.unique(predictions, return_counts=True)
for i in range(len(unique_vals)):
    print(f"{str(unique_vals[i])} : {counts[i]}")

print(f"Total predictions: {len(predictions)}")
print("Predictions:")
print(predictions)