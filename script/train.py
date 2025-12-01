import sys
from script.data_modeling import model_data
from script.data_preprocessing import load_data, preprocess_data

BASE_PATH = "data/raw_data/"

csv_file = "data1" if len(sys.argv) < 2 else sys.argv[1]
print(f"Loading data from: {BASE_PATH + csv_file + '/*.csv'}")

data = load_data(BASE_PATH + csv_file + '/*.csv')

X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(data, csv_file)

model_data(X_train, y_train, X_val, y_val, X_test, y_test)

def train():
    BASE_PATH = "data/raw_data/"

    csv_file = "data1" if len(sys.argv) < 2 else sys.argv[1]
    print(f"Loading data from: {BASE_PATH + csv_file + '/*.csv'}")

    data = load_data(BASE_PATH + csv_file + '/*.csv')

    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(data, csv_file)

    model_data(X_train, y_train, X_val, y_val, X_test, y_test)

    print("Training completed. Model and predictions are ready.")