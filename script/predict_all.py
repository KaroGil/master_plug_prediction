import joblib
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from script.helper_methods.data_preprocessing import preprocess_data_predict
from script.helper_methods.data_visualization import plot_one

BASE_PATH = "data/labeled/labeled_"

def predict_all(runId):
    # Load data
    print("💾 Loading multiple datasets for prediction...")
    data_list = []
    for i in range(1, 13):
        if i == 2:
            continue  # Skip data2
        data_list.append(pd.read_csv(BASE_PATH + f"data{i}.csv"))

    # Preprocess 
    print("🛠️ Preprocessing datasets for prediction...")
    X_y_list = []
    for i, d in enumerate(data_list, 1):
        X_y_list.append(preprocess_data_predict(d, dataset_name=f"data{i + 1 if i >= 2 else 1}"))

    # Load model
    print("🔮 Loading model and making predictions...")
    model = joblib.load("models/best_model.joblib")
    print("Model used: " + type(model).__name__ + " with parameters: " + str(model.get_params()))

    # Predict 
    print("🖨️ Making predictions on all datasets...")
    y_preds = []
    for d in X_y_list:
        prediction = model.predict(d[0])
        y_preds.append(prediction)
        print(f"F1-score for this run was {f1_score(prediction, d[1])}")


    # Visualize 
    print("📊 Visualizing predicted vs true values...")

    plt.figure(figsize=(12,6))

    for X_y_u, y_pred, i in zip(X_y_list, y_preds, range(1,len(X_y_list)+1)):
        plot_one(X_y_u[0], y_pred, i + 1 if i >= 2 else 1, X_y_u[1])

    plt.subplots_adjust(hspace=0.5)
    #show
    #plt.show()
    plt.savefig(f"plots/{runId}.png", dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    predict_all()