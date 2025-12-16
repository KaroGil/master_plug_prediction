import sys
import joblib
import matplotlib.pyplot as plt
from script.data_preprocessing import load_data, preprocess_data_predict
import pandas as pd

BASE_PATH = "data/raw_data/"

csv_file = "data1" if len(sys.argv) < 2 else sys.argv[1]
print(f"Loading data from: {BASE_PATH + csv_file + '/*.csv'}")

print("💾 Loading multiple datasets for prediction...")
data = load_data(BASE_PATH + csv_file + '/*.csv')
#data2 = load_data(BASE_PATH + "data2/*.csv")
data3 = load_data(BASE_PATH + "data3/*.csv")
data4 = load_data(BASE_PATH + "data4/*.csv")
data5 = load_data(BASE_PATH + "data5/*.csv")
data6 = load_data(BASE_PATH + "data6/*.csv")
data7 = load_data(BASE_PATH + "data7/*.csv")
data8 = load_data(BASE_PATH + "data8/*.csv")
data9 = load_data(BASE_PATH + "data9/*.csv")
data10 = load_data(BASE_PATH + "data10/*.csv")
data11 = load_data(BASE_PATH + "data11/*.csv")
data12 = load_data(BASE_PATH + "data12/*.csv")

print("🛠️ Preprocessing datasets for prediction...")
X, y, unscaled_X = preprocess_data_predict(data)
#X2, y2, unscaled_X2 = preprocess_data_predict(data2)
X3, y3, unscaled_X3 = preprocess_data_predict(data3)
X4, y4, unscaled_X4 = preprocess_data_predict(data4)
X5, y5, unscaled_X5 = preprocess_data_predict(data5)    
X6, y6, unscaled_X6 = preprocess_data_predict(data6)
X7, y7, unscaled_X7 = preprocess_data_predict(data7)
X8, y8, unscaled_X8 = preprocess_data_predict(data8)
X9, y9, unscaled_X9 = preprocess_data_predict(data9)
X10, y10, unscaled_X10 = preprocess_data_predict(data10)
X11, y11, unscaled_X11 = preprocess_data_predict(data11)
X12, y12, unscaled_X12 = preprocess_data_predict(data12)

print("🔮 Loading model and making predictions...")
model = joblib.load("models/best_model.joblib")
print("Model used: " + type(model).__name__ + " with parameters: " + str(model.get_params()))

print("🖨️ Making predictions on all datasets...")
predictions = model.predict(X)
print("Predictions for dataset 1 completed.")
# predictions2 = model.predict(X2)
# print("Predictions for dataset 2 completed.")
predictions3 = model.predict(X3)
print("Predictions for dataset 3 completed.")
predictions4 = model.predict(X4)
print("Predictions for dataset 4 completed.")
predictions5 = model.predict(X5)
print("Predictions for dataset 5 completed.")
predictions6 = model.predict(X6)    
print("Predictions for dataset 6 completed.")
predictions7 = model.predict(X7)
print("Predictions for dataset 7 completed.")
predictions8 = model.predict(X8)
print("Predictions for dataset 8 completed.")
predictions9 = model.predict(X9)
print("Predictions for dataset 9 completed.")
predictions10 = model.predict(X10)
print("Predictions for dataset 10 completed.")
predictions11 = model.predict(X11)
print("Predictions for dataset 11 completed.")
predictions12 = model.predict(X12)
print("Predictions for dataset 12 completed.")


print("📊 Visualizing predicted vs true values...")
y_preds = [predictions, predictions3, predictions4, predictions5, predictions6, predictions7, predictions8, predictions9, predictions10, predictions11, predictions12] 
dfs = [data, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12]
unscaled = [unscaled_X, unscaled_X3, unscaled_X4, unscaled_X5, unscaled_X6, unscaled_X7, unscaled_X8, unscaled_X9, unscaled_X10, unscaled_X11, unscaled_X12]


plt.figure(figsize=(12,6))

def plot_one(df, y_pred, figureNum, flow_col="Flow rate (Mean)", pressure_col="Pump outlet pressure (Mean)"):
    if flow_col not in df.columns:
        flow_col = "Flow rate (Mean)_mean"
    plt.subplot(4,3,figureNum)
    plt.plot(df.index, df[flow_col], label="Flow rate", alpha=0.5)

    # Highlight predicted Plug events
    plug_events = df[y_pred == 1]
    plt.scatter(plug_events.index, plug_events[flow_col], color="yellow", label="Predicted plug (Flow)", zorder=7, marker='.') 
    plt.scatter(plug_events.index, plug_events[pressure_col], color="green", label="Predicted plug (Pressure)", zorder=7, marker='.')  if pressure_col in df.columns else None
    plt.xlabel("Elapsed_seconds")
    plt.ylabel("Value")
    plt.title(f"Predicted vs True Plug=1 Events for data nr {figureNum}")
    plt.legend()

for unscaled_df, y_pred, i in zip(unscaled, y_preds, range(1,12)):
    plot_one(unscaled_df, y_pred, i)

plt.show()
