import joblib
import pandas as pd
from sklearn.metrics import f1_score
from script.helper_methods.config import get_config
from script.helper_methods.model_evaluation import false_alarm_analysis, per_dataset_statistics
from script.helper_methods.data_visualization import f1_score_bar_plot_comparison, f1_score_line_plot_comparison
from script.helper_methods.check_low_performers import analyse_low_performing_datasets
from script.preprocess_predict import preped_for_prediction_exists, preprocess_and_save

# Load config
cfg = get_config()
target_col = cfg["data"]["target"]
datasets = cfg["data"]["datasets"]
horizon = cfg["experiment"]["horizon"]
flow_rate_missing_sets = cfg["data"]["flow_rate_missing"]
test_sets = cfg["data"]["test_sets"]

BASE_PATH = "data/labeled/labeled_"
BASE_PATH_PREPROCESSED_PREDICT = "data/processed_data/predict/"

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

# # Load data
print("💾 Loading multiple datasets for prediction...")
data_list = []
dataset_ids = []
print(datasets)
for i in datasets:
    if i in [2]:
        continue  # Skip data2 because of its size
    data_list.append(pd.read_csv(BASE_PATH + f"data{i}.csv"))
    dataset_ids.append(i)

# Load model
print("Loading in the three models and making predictions...")
model_baseline = joblib.load("models/dummy.joblib")
model_RF = joblib.load("models/rf.joblib")
model_XGBoost = joblib.load("models/xgboost.joblib")
print("1. Model used: " + type(model_baseline).__name__ + " with parameters: " + str(model_baseline.get_params()))
print("2. Model used: " + type(model_RF).__name__ + " with parameters: " + str(model_RF.get_params()))
print("3. Model used: " + type(model_XGBoost).__name__ + " with parameters: " + str(model_XGBoost.get_params()))

# Predict 
print("🖨️ Making predictions on all datasets...")
y_preds_baseline = []
y_preds_RF = []
y_preds_XGBoost = []

f1score_baseline = []
f1scores_RF = []
f1scores_XGBoost = []

for dataset_id, (X, y, _) in zip(dataset_ids, X_y_list):
    prediction_baseline = model_baseline.predict(X)
    prediction_RF = model_RF.predict(X)
    prediction_XGBoost = model_XGBoost.predict(X)

    y_preds_baseline.append(prediction_baseline)
    y_preds_RF.append(prediction_RF)
    y_preds_XGBoost.append(prediction_XGBoost)

    f1_baseline = f1_score(y, prediction_baseline, zero_division=0, average='weighted')
    f1_RF = f1_score(y, prediction_RF, zero_division=0, average='weighted')
    f1_XGBoost = f1_score(y, prediction_XGBoost, zero_division=0, average='weighted')

    f1score_baseline.append(f1_baseline)
    f1scores_RF.append(f1_RF)
    f1scores_XGBoost.append(f1_XGBoost)

    print(f"F1-score for dataset {dataset_id} run was {f1_baseline} for baseline, {f1_RF} for RF and {f1_XGBoost} for XGBoost")

# Visualize predictions as bar plot comparing the F1-scores of the different datasets for the three model
print("📊 Visualizing f1-scores as bar plots...")
f1_score_bar_plot_comparison(dataset_ids, {
    "Random Forest": f1scores_RF,
    "XGBoost": f1scores_XGBoost
})

test_indices = [dataset_ids.index(i) for i in test_sets]
f1_score_bar_plot_comparison(test_sets, {
    "Random Forest": [f1scores_RF[i] for i in test_indices],
    "XGBoost": [f1scores_XGBoost[i] for i in test_indices]
}, name="test_sets")

f1_score_line_plot_comparison(dataset_ids, {
    "Random Forest": f1scores_RF,
    "XGBoost": f1scores_XGBoost
})


# Per-dataset statistics
per_dataset_statistics(
    {
        "Random Forest": f1scores_RF,
        "XGBoost":       f1scores_XGBoost,
    }
)

# False alarm analysis on test datasets
test_indices = [dataset_ids.index(i) for i in test_sets]

for idx, ds_id in zip(test_indices, test_sets):
    false_alarm_analysis(
        y_true=X_y_list[idx][1],  
        y_pred=y_preds_XGBoost[idx],
        dataset_id=ds_id,
        sample_rate_hz=2,
        model_name="XGBoost"
    )

for idx, ds_id in zip(test_indices, test_sets):
    false_alarm_analysis(
        y_true=X_y_list[idx][1],  
        y_pred=y_preds_RF[idx],
        dataset_id=ds_id,
        sample_rate_hz=2,
        model_name="RF"
    )

# Low performers analysis
print("\n📊 Analyzing low-performing datasets...")
analyse_low_performing_datasets(
    dataset_ids,
    {
        "Random Forest": f1scores_RF,
        "XGBoost":       f1scores_XGBoost,
    },
    X_y_list,
    threshold=0.8
)

raw_data = {ds_id: data_list[i] for i, ds_id in enumerate(dataset_ids)}
low_ds_ids = [16] # Set manually!  
low_indices = [dataset_ids.index(i) for i in low_ds_ids]

for idx, ds_id in zip(low_indices, low_ds_ids):
    false_alarm_analysis(
        y_true=X_y_list[idx][1],  
        y_pred=y_preds_RF[idx],
        dataset_id=ds_id,
        sample_rate_hz=2,
        model_name="RF"
    )
