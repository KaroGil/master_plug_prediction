import sys
import pandas as pd
from script.helper_methods.data_modeling import model_data
from script.helper_methods.data_preprocessing import preprocess_data

LABLED_PATH = "data/labeled/labeled_"

csv_file = "data1" if len(sys.argv) < 2 else sys.argv[1]
print(f"Loading data from: {LABLED_PATH + csv_file + '.csv'}")

# Data specified when running the script
data = pd.read_csv(LABLED_PATH + csv_file + '.csv')

datasets = []

# Additional data to include in training
for i in [3,5,8,11,12]:
    if f"data{i}" in csv_file:
        continue
    datasets.append(pd.read_csv(LABLED_PATH + f"data{i}.csv"))

X_train, X_test, y_train, y_test = preprocess_data(datasets, [f"data{i}" for i in [3,5,8,11,12] if f"data{i}" not in csv_file])

# For testing purposes only - load preprocessed data
#X_train, X_test, y_train, y_test = pd.read_csv("data/processed_data/data_1_3_4_5_6_7_8_9_10_11_12_X_train.csv"), pd.read_csv("data/processed_data/data_1_3_4_5_6_7_8_9_10_11_12_X_test.csv"), pd.read_csv("data/processed_data/data_1_3_4_5_6_7_8_9_10_11_12_y_train.csv"), pd.read_csv("data/processed_data/data_1_3_4_5_6_7_8_9_10_11_12_y_test.csv")

model_data(X_train, y_train, X_test, y_test)