import sys
import pandas as pd
from script.helper_methods.data_modeling import model_data
#from script.helper_methods.data_preprocessing import preprocess_data

LABLED_PATH = "data/labeled/labeled_"

csv_file = "data1" if len(sys.argv) < 2 else sys.argv[1]
# print(f"Loading data from: {LABLED_PATH + csv_file + '.csv'}")

# Data specified when running the script
# data = pd.read_csv(LABLED_PATH + csv_file + '.csv')

# # Additional data to include in training
# data5 = pd.read_csv(LABLED_PATH + "data5.csv")
# data4 = pd.read_csv(LABLED_PATH + "data4.csv")
# data7 = pd.read_csv(LABLED_PATH + "data7.csv")
# data9 = pd.read_csv(LABLED_PATH + "data9.csv")
# data11 = pd.read_csv(LABLED_PATH + "data11.csv")

#X_train, X_test, y_train, y_test = preprocess_data(data, csv_file, additional_data=[data5, data4, data7, data9, data11], additional_data_name=["data5", "data4", "data7", "data9", "data11"])
X_train, X_test, y_train, y_test = pd.read_csv("data/processed_data/data1_X_train.csv"), pd.read_csv("data/processed_data/data1_X_test.csv"), pd.read_csv("data/processed_data/data1_y_train.csv"), pd.read_csv("data/processed_data/data1_y_test.csv")

model_data(X_train, y_train, X_test, y_test)