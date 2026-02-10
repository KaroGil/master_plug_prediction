import yaml
from script.helper_methods.data_preprocessing import preprocess_data_predict
import joblib
import sys
import numpy as np
import pandas as pd
from script.helper_methods import data_visualization as dv

# Load config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

target_col = cfg["data"]["target"]

BASE_PATH = "data/labeled/labeled_"

csv_file = "data1" if len(sys.argv) < 2 else sys.argv[1]
print(f"Loading data from: {BASE_PATH + csv_file + '/*.csv'}")

data = pd.read_csv(BASE_PATH + csv_file + '.csv')

X, y = preprocess_data_predict(data, dataset_name=csv_file)

model = joblib.load("models/best_model.joblib")
print("Model used: " + type(model).__name__ + " with parameters: " + str(model.get_params()))

predictions = model.predict(X)

print("Visualizing predicted vs true values...")
X[target_col] = y
dv.visualize_predicted_vs_true(X, predictions, anomalies=True, model_name=type(model).__name__)

print("Unique predictions:")
unique_vals, counts = np.unique(predictions, return_counts=True)
for i in range(len(unique_vals)):
    print(f"{str(unique_vals[i])} : {counts[i]}")

print(f"Total predictions: {len(predictions)}")
print("Predictions:")
print(predictions)