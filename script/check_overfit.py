import os
import sys
import joblib
import pandas as pd
from sklearn.metrics import f1_score

BASE_PATH = "data/processed_data/"
csv_file = "data1" if len(sys.argv) < 2 else sys.argv[1]
print(f"Loading data from: {BASE_PATH + csv_file + '_X_train.csv'}")  

train_X = pd.read_csv(BASE_PATH + csv_file + '_X_train.csv')
val_X = pd.read_csv(BASE_PATH + csv_file + "_X_val.csv")

train_y = pd.read_csv(BASE_PATH + csv_file + '_y_train.csv').values.ravel()
val_y = pd.read_csv(BASE_PATH + csv_file + "_y_val.csv").values.ravel()

model = joblib.load("models/best_model.joblib")
print("Model used: " + type(model).__name__ + " with parameters: " + str(model.get_params()))

train_predictions = model.predict(train_X)
val_predictions = model.predict(val_X)

print(f"Performance on Training set: {f1_score(train_y, train_predictions)} F1-score")
print(f"Performance on Validation set: {f1_score(val_y, val_predictions)} F1-score")
