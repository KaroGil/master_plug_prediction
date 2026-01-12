import joblib
import matplotlib.pyplot as plt
from script.helper_methods.data_preprocessing import load_data, preprocess_data_predict

BASE_PATH = "data/raw_data/"

print("💾 Loading multiple datasets for prediction...")
data_list = []
for i in range(1, 13):
    data_list.append(load_data(BASE_PATH + f"data{i}/*.csv"))


print("🛠️ Preprocessing datasets for prediction...")
X_y_list = []
for d in data_list:
    X_y_list.append(preprocess_data_predict(d, dataset_name=f"data{i}"))


### LOAD MODEL AND PREDICT ###
print("🔮 Loading model and making predictions...")
model = joblib.load("models/best_model.joblib")
print("Model used: " + type(model).__name__ + " with parameters: " + str(model.get_params()))

print("🖨️ Making predictions on all datasets...")
y_preds = []
for d in X_y_list:
    y_preds.append(model.predict(d[0]))


print("📊 Visualizing predicted vs true values...")
unscaled = [d[2] for d in X_y_list]

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
    plt.title(f"Predicted vs True Plug=1 Events for data nr {figureNum}")
    plt.legend()


plt.figure(figsize=(12,6))

for X_y_u, y_pred, i in zip(X_y_list, y_preds, range(1,len(X_y_list)+1)):
    plot_one(X_y_u[2], y_pred, i, X_y_u[1])

plt.subplots_adjust(hspace=0.5)
plt.show()
