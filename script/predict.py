from script.data_preprocessing import load_data, preprocess_data_predict
import joblib
import sys
import numpy as np

BASE_PATH = "data/raw_data/"

csv_file = "data1" if len(sys.argv) < 2 else sys.argv[1]
print(f"Loading data from: {BASE_PATH + csv_file + '/*.csv'}")

data = load_data(BASE_PATH + csv_file + '/*.csv')

data = preprocess_data_predict(data)

model = joblib.load("models/best_model.joblib")

predictions = model.predict(data)

print("Unique predictions:")
unique_vals, counts = np.unique(predictions, return_counts=True)
for i in range(len(unique_vals)):
    print(f"{str(unique_vals[i])} : {counts[i]}")
print(f"Total predictions: {len(predictions)}")
print("Predictions:")
print(predictions)