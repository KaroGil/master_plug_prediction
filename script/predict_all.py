import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from script.helper_methods.data_preprocessing import preprocess_data_predict

BASE_PATH = "data/labeled/labeled_"

print("💾 Loading multiple datasets for prediction...")
data_list = []
for i in range(1, 13):
    if i == 2:
        continue  # Skip data2
    data_list.append(pd.read_csv(BASE_PATH + f"data{i}.csv"))

print("🛠️ Preprocessing datasets for prediction...")
X_y_list = []
for i, d in enumerate(data_list, 1):
    X_y_list.append(preprocess_data_predict(d, dataset_name=f"data{i + 1 if i >= 2 else 1}"))


### LOAD MODEL AND PREDICT ###
print("🔮 Loading model and making predictions...")
model = joblib.load("models/best_model.joblib")
print("Model used: " + type(model).__name__ + " with parameters: " + str(model.get_params()))

print("🖨️ Making predictions on all datasets...")
y_preds = []
for d in X_y_list:
    prediction = model.predict(d[0])
    y_preds.append(prediction)
    print(f"F1-score for this run was {f1_score(prediction, d[1])}")

print("📊 Visualizing predicted vs true values...")

### VISUALIZING RESULTS ###
def plot_one(df, y_pred, figureNum, y, flow_col="Flow rate (Mean)", pressure_col="Pump outlet pressure (Mean)"):
    if flow_col not in df.columns:
        flow_col = "Flow rate (Mean)_mean"


    plt.subplot(4,3,figureNum)
    plt.plot(df.index, df[flow_col], label="Flow rate", alpha=0.5)

    # Highlight true Plug events
    true_plug_events = df[y == 1]
    plt.scatter(true_plug_events.index, true_plug_events[flow_col], color="red", label="True Plug=1 (Flow)", zorder=6, marker='x')
    plt.scatter(true_plug_events.index, true_plug_events[pressure_col], color="blue", label="True Plug=1 (Pressure)", zorder=6, marker='x')  if pressure_col in df.columns else None
    
    # Highlight predicted Plug events
    plug_events = df[y_pred == 1]
    plt.scatter(plug_events.index, plug_events[flow_col], color="yellow", label="Predicted plug (Flow)", zorder=7, marker='.') 
    plt.scatter(plug_events.index, plug_events[pressure_col], color="green", label="Predicted plug (Pressure)", zorder=7, marker='.')  if pressure_col in df.columns else None
     
    plt.xlabel("Elapsed_seconds")
    plt.ylabel("Value")
    if figureNum == 12:
        plt.title(f"Predicted vs True Plug=1 Events for data nr {figureNum} [used as test set]")
    else:
        plt.title(f"Predicted vs True Plug=1 Events for data nr {figureNum}" if figureNum not in [3,5,8,11] else f"Predicted vs True Plug=1 Events for data nr {figureNum} [used for training]")
    plt.legend()


plt.figure(figsize=(12,6))

for X_y_u, y_pred, i in zip(X_y_list, y_preds, range(1,len(X_y_list)+1)):
    plot_one(X_y_u[0], y_pred, i + 1 if i >= 2 else 1, X_y_u[1])

plt.subplots_adjust(hspace=0.5)
plt.show()
