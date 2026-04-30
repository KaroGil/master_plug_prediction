import joblib
import pandas as pd
from sklearn.metrics import f1_score
from script.helper_methods.config import get_config
from script.preprocess_predict import preped_for_prediction_exists, preprocess_and_save
from script.helper_methods.data_visualization import f1_score_bar_plot, plot_all_predictions, plot_test_data_summary

# Load config
cfg = get_config()
target_col = cfg["data"]["target"]
datasets = cfg["data"]["datasets"]
horizon = cfg["experiment"]["horizon"]
flow_rate_missing_sets = cfg["data"]["flow_rate_missing"]
test_sets = cfg["data"]["test_sets"]

BASE_PATH = "data/labeled/labeled_"
BASE_PATH_PREPROCESSED_PREDICT = "data/processed_data/predict/"

def predict_all(runId, samples=horizon):
    #NB! If train.py is run again with a different set of datasets the saved datasets in data/processed_data/predict/ have to be 
    # deleted so new can be made and the prediction is run on the correct datasets.
    if preped_for_prediction_exists():
        print("Preprocessed data already exists. Skipping preprocessing.")
        dataset_ids = []
        X_y_list = []
        for i in datasets:
            if i in [2]:
                continue
            dataset_ids.append(i)
            X = pd.read_csv(f"{BASE_PATH_PREPROCESSED_PREDICT}data_{i}_X.csv")
            y = pd.read_csv(f"{BASE_PATH_PREPROCESSED_PREDICT}data_{i}_y.csv").squeeze()
            flow = pd.read_csv(f"{BASE_PATH_PREPROCESSED_PREDICT}data_{i}_flow_rate.csv").squeeze()
            X_y_list.append((X, y, flow))
    else:
        print("Preprocessed data does not exist. Starting preprocessing.")
        preprocess_and_save() #First preprocess and save the data, then load it
        dataset_ids = []
        X_y_list = []
        for i in datasets:
            if i in [2]:
                continue
            dataset_ids.append(i)
            X = pd.read_csv(f"{BASE_PATH_PREPROCESSED_PREDICT}data_{i}_X.csv")
            y = pd.read_csv(f"{BASE_PATH_PREPROCESSED_PREDICT}data_{i}_y.csv").squeeze()
            flow = pd.read_csv(f"{BASE_PATH_PREPROCESSED_PREDICT}data_{i}_flow_rate.csv").squeeze()
            X_y_list.append((X, y, flow))
    
    # Load model
    print("🔮 Loading model and making predictions...")
    model = joblib.load("models/best_model.joblib")
    print("Model used: " + type(model).__name__ + " with parameters: " + str(model.get_params()))

    # Predict 
    print("🖨️ Making predictions on all datasets...")
    y_preds = []
    f1scores = []
    for dataset_id, (X, y, _) in zip(dataset_ids, X_y_list):
        prediction = model.predict(X)
        y_preds.append(prediction)
        f1 = f1_score(y, prediction, zero_division=0, average='weighted')
        f1scores.append(f1)
        print(f"F1-score for dataset {dataset_id} run was {f1}")

    # Visualize predictions as bar plot
    print("📊 Visualizing f1-scores as bar plots...")
    f1_score_bar_plot(dataset_ids, f1scores, runId)

    # Visualize as true vs predicted events line plots
    print("📊 Visualizing predicted vs true values...")
    print(f"Datasets with flow rate missing: {flow_rate_missing_sets}")

    all_data = list(zip(dataset_ids, X_y_list, y_preds))

    plug_data = [(id, (X, y, flow_rate), pred) for id, (X, y, flow_rate), pred in all_data if y.unique().size > 1]
    no_plug_data = [(id, (X, y, flow_rate), pred) for id, (X, y, flow_rate), pred in all_data if y.unique().size < 2]
    
    def plot(data, suffix):
        dataset_ids, Xys, preds = zip(*data)
        plot_all_predictions(Xys, preds, dataset_ids, flow_rate_missing_sets, samples, model, f"{runId}_{suffix}")


    #plot([(id, Xy, p) for id, Xy, p in plug_data if id not in flow_rate_missing_sets], suffix="flow_based")
    plot([(id, Xy, p) for id, Xy, p in plug_data if id in flow_rate_missing_sets], suffix="pressure_based")
    plot(no_plug_data, suffix="no_plug")
    plot([(id, Xy, p) for id, Xy, p in all_data if id == 16], suffix="dataset_16")

    test_data = [(id, Xy, p) for id, Xy, p in all_data if id in test_sets]
    X_y_list = [Xy for _, Xy, _ in test_data]
    y_preds = [p for _, _, p in test_data]
    dataset_ids = [id for id, _, _ in test_data]

    plot_test_data_summary(X_y_list, y_preds, dataset_ids, flow_rate_missing_sets, runId="test_sets")
    plot_test_data_summary(X_y_list, y_preds, dataset_ids, test_sets, runId="test_sets_pressure_based")

if __name__ == "__main__":
    predict_all(runId="default_run")