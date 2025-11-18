from script.data_modeling import model_data
from script.data_preprocessing import load_data, preprocess_data

data = load_data("data/raw_data/data1/*.csv")

X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(data, "dataset_1")

final_model, y_test_pred = model_data(X_train, y_train, X_val, y_val, X_test, y_test)