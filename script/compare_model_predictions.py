"""
This script compares the predictions of the three models (baseline, RF and XGBoost) on all datasets. 
It calculates the F1-scores for each dataset and visualizes them as bar plots. 
It also performs a false alarm analysis on the test datasets and identifies low-performing datasets for further analysis.
"""

import joblib
import pandas as pd
from sklearn.metrics import f1_score
from script.helper_methods.config import get_config
from script.helper_methods.check_low_performers import analyse_low_performing_datasets
from script.preprocess_predict import load_preprocessed_data, preped_for_prediction_exists, preprocess_and_save
from script.helper_methods.data_visualization.f1_score import f1_score_bar_plot_comparison, f1_score_line_plot_comparison
from script.helper_methods.model_evaluation import false_alarm_analysis, per_dataset_statistics, print_class_distribution_across_datasets

# Load config
cfg = get_config()
datasets = cfg["data"]["datasets"]
test_sets = cfg["data"]["test_sets"]
frequency_hz = cfg["data"]["frequency"]

BASE_PATH = "data/labeled/labeled_"
BASE_PATH_PREPROCESSED_PREDICT = "data/processed_data/predict/"

if preped_for_prediction_exists():
    # If preprocessed data already exists, load it instead of preprocessing again
    print("Preprocessed data already exists. Skipping preprocessing.")
    X_y_list, dataset_ids = load_preprocessed_data()
else:
    # If preprocessed data does not exist, preprocess and save it
    print("Preprocessed data does not exist. Starting preprocessing.")
    X_y_list, dataset_ids = preprocess_and_save() #Preprocess and save the data

# Load labeled unprocessed data
print("💾 Loading multiple datasets for prediction...")
data_list = []
dataset_ids = []
print(datasets)
for i in datasets:
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

# Iterate through all datasets, make predictions and calculate F1-scores for each model
for dataset_id, (X, y) in zip(dataset_ids, X_y_list):
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

# Print class distribution across datasets
print_class_distribution_across_datasets(dataset_ids, X_y_list)

# Visualize predictions as bar plot comparing the F1-scores of the different datasets for the three models
# All datasets
print("\n📊 Visualizing f1-scores as bar plots...")
f1_score_bar_plot_comparison(dataset_ids, {
    "Random Forest": f1scores_RF,
    "XGBoost": f1scores_XGBoost
})

# Test datasets only
test_indices = [dataset_ids.index(i) for i in test_sets]
f1_score_bar_plot_comparison(test_sets, {
    "Random Forest": [f1scores_RF[i] for i in test_indices],
    "XGBoost": [f1scores_XGBoost[i] for i in test_indices]
}, name="test_sets")

# Visualize predictions as line plot comparing the F1-scores of the different datasets for the three model
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
        sample_rate_hz=frequency_hz,
        model_name="XGBoost"
    )

for idx, ds_id in zip(test_indices, test_sets):
    false_alarm_analysis(
        y_true=X_y_list[idx][1],  
        y_pred=y_preds_RF[idx],
        dataset_id=ds_id,
        sample_rate_hz=frequency_hz,
        model_name="RF"
    )

# Low performers analysis
print("\n📊 Analyzing low-performing datasets...")
low_ds_ids = analyse_low_performing_datasets(
    dataset_ids,
    {
        "Random Forest": f1scores_RF,
        "XGBoost":       f1scores_XGBoost,
    },
    X_y_list,
    threshold=0.95
)

raw_data = {ds_id: data_list[i] for i, ds_id in enumerate(dataset_ids)}
low_indices = [dataset_ids.index(i) for i in low_ds_ids]

for idx, ds_id in zip(low_indices, low_ds_ids):
    false_alarm_analysis(
        y_true=X_y_list[idx][1],  
        y_pred=y_preds_RF[idx],
        dataset_id=ds_id,
        sample_rate_hz=frequency_hz,
        model_name="RF"
    )
    false_alarm_analysis(
        y_true=X_y_list[idx][1],  
        y_pred=y_preds_XGBoost[idx],
        dataset_id=ds_id,
        sample_rate_hz=frequency_hz,
        model_name="XGBoost"
    )
