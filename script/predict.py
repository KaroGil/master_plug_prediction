from script.data_preprocessing import load_data, preprocess_data_predict
import joblib
import sys
import numpy as np
import pandas as pd
from script import data_visualization as dv

BASE_PATH = "data/raw_data/"

csv_file = "data1" if len(sys.argv) < 2 else sys.argv[1]
print(f"Loading data from: {BASE_PATH + csv_file + '/*.csv'}")

data = load_data(BASE_PATH + csv_file + '/*.csv')

X, y, unscaled_X = preprocess_data_predict(data)

model = joblib.load("models/best_model.joblib")

predictions = model.predict(X)

print("Visualizing predicted vs true values...")

dv.visualize_predicted_vs_true(pd.concat([unscaled_X, y], axis=1), predictions, anomalies=True, model_name=type(model).__name__)
dv.visualize_predicted_vs_true(pd.concat([unscaled_X, y], axis=1), predictions, model_name=type(model).__name__)

print("Unique predictions:")
unique_vals, counts = np.unique(predictions, return_counts=True)
for i in range(len(unique_vals)):
    print(f"{str(unique_vals[i])} : {counts[i]}")
print(f"Total predictions: {len(predictions)}")
print("Predictions:")
print(predictions)