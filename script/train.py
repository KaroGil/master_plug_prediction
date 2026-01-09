import sys
from script.helper_methods.data_modeling import model_data
from script.helper_methods.data_preprocessing import load_data, preprocess_data

BASE_PATH = "data/raw_data/"

csv_file = "data1" if len(sys.argv) < 2 else sys.argv[1]
print(f"Loading data from: {BASE_PATH + csv_file + '/*.csv'}")

# Data specified when running the script
data = load_data(BASE_PATH + csv_file + '/*.csv')

# Additional data to include in training
data5 = load_data(BASE_PATH + "data5/*.csv")
data4 = load_data(BASE_PATH + "data4/*.csv")
data7 = load_data(BASE_PATH + "data7/*.csv")
data8 = load_data(BASE_PATH + "data8/*.csv")
data9 = load_data(BASE_PATH + "data9/*.csv")
data10 = load_data(BASE_PATH + "data10/*.csv")
data11 = load_data(BASE_PATH + "data11/*.csv")

X_train, X_test, y_train, y_test = preprocess_data(data, csv_file, additional_data=[data5, data4, data7, data8, data9, data10, data11], additional_data_name=["data5", "data4", "data7", "data8", "data9", "data10", "data11"])

print("Final training columns:", X_train.columns)
model_data(X_train, y_train, X_test, y_test)