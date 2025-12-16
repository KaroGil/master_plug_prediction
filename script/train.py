import sys
from script.data_modeling import model_data
from script.data_preprocessing import load_data, preprocess_data
import pandas as pd

BASE_PATH = "data/raw_data/"

csv_file = "data1" if len(sys.argv) < 2 else sys.argv[1]
print(f"Loading data from: {BASE_PATH + csv_file + '/*.csv'}")

data = load_data(BASE_PATH + csv_file + '/*.csv')
data5 = load_data(BASE_PATH + "data5/*.csv")

X_train, X_test, y_train, y_test = preprocess_data(data, csv_file, additional_data=[data5], additional_data_name=["data5"])

print("Final training columns:", X_train.columns)
model_data(X_train, y_train, X_test, y_test)