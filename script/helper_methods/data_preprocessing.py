"""
This module contains all functions related to data preprocessing, including:
- Creating the target column based on defined thresholds
- Creating the future target column by shifting the current target
- Splitting the data into train and test sets without shuffling
- Printing the class distribution of the target variable
- Aligning features of the prediction dataset with the features used for training
- A full preprocessing pipeline for both model training and prediction
"""

import os
import joblib
import numpy as np
import pandas as pd

from . import window as w
from . import data_loader as dl
from . import feature_reduction as fr
from . import feature_engineering as fe
from .config import get_config

# Load config
cfg = get_config()

target_col = cfg["data"]["target"]
non_feature_columns = cfg["data"]["non_feature_columns"]
HORIZON = cfg["experiment"]["horizon"]
test_sets = cfg["data"]["test_sets"]

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', '..', 'models')
MODELS_DIR = os.path.abspath(MODELS_DIR)

scaler_path = os.path.join(MODELS_DIR, 'standard_scaler.pkl')
scaler_path = os.path.abspath(scaler_path)

FEATURES_PATH = os.path.join(MODELS_DIR, 'features_list.pkl')
FEATURES_PATH = os.path.abspath(FEATURES_PATH)


def create_future_target(df, horizon=HORIZON):
    '''Create target column 'Plug_future' by shifting 'Plug' column'''

    df[target_col] = (df['Plug'].shift(-1).rolling(window=horizon, min_periods=1).max())
    df.dropna(subset=[target_col], inplace=True)


def split_data(X,y):
    '''Split data into train, validation, and test sets without shuffling'''
    test_set_log_id = test_sets if test_sets else [X['LogId'].max()] # Use specified test sets or default to the last LogId
    # Extract test set 
    X_test = X.loc[X['LogId'].isin(test_set_log_id)]
    y_test = y.loc[y.index.isin(X_test.index)]
    # Extract training set
    X_train = X.loc[~X['LogId'].isin(test_set_log_id)]
    y_train = y.loc[y.index.isin(X_train.index)] 

    print(f"Train set LogIds: {X_train['LogId'].unique()}")
    print(f"Test set LogIds: {X_test['LogId'].unique()}")

    return X_train, X_test, y_train, y_test


def print_distribution(y, name):
    '''Print class distribution of the target variable'''

    unique, counts = np.unique(y, return_counts=True)
    distribution = dict(zip(unique, counts))
    print(f"{name} distribution:")
    for key, value in distribution.items():
        print(f"  Class {key}: {value} samples")


def align_features(df, FEATURES):
    '''Align features of the prediction dataset with the features used for training'''

    # Add missing features with default value 0
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0 

    return df[FEATURES]


def feature_engineering_windowing(df, dataset_name="data1", window_size=2, labels=True):
    '''Adding feature engineering steps and windowing to the data'''

    # Make sure LogId is present for windowing, if not create it based on dataset name
    if 'LogId' not in df.columns:
        df['LogId'] = dataset_name[4:]

    # Apply feature engineering pipeline
    df = fe.feature_engineering_pipeline(df) 

    # Add time derivative features
    print("Adding time derivative features...")
    df = fe.add_time_derivative_features(df)

    # Add plug index feature
    print("Adding plug index feature...")
    df = fe.plug_index(df)

    # Extract numeric features for windowing
    df_feat = df.select_dtypes(include=['number']).copy()

    # Apply windowing to create sequences of features and corresponding target values
    X, y = w.prep_window(df, [x for x in df_feat.columns.tolist() if x not in  non_feature_columns], window_size=window_size, labels=labels)

    return X, y


def preprocess_data(datasets, dataset_names, horizon=HORIZON, BASE_PATH = "", window_size=30):
    '''Full preprocessing pipeline for model selection and training'''

    # Process the first dataset to initialize X and y, then loop through the rest and concatenate
    print(f"Processing data: {dataset_names[0]}")
    create_future_target(datasets[0], horizon=horizon)
    X, y = feature_engineering_windowing(datasets[0], dataset_names[0], window_size=window_size)

    for df, name in zip(datasets[1:], dataset_names[1:]):
        print(f"Processing data: {name}") 
        create_future_target(df, horizon=horizon)
        X_add, y_add = feature_engineering_windowing(df, name, window_size=window_size) 

        common_cols_X = sorted(set.intersection(*(set(df.columns) for df in [X, X_add]))) # Keep only common columns in the same order

        # Combine datasets into one
        X = pd.concat([X[common_cols_X], X_add[common_cols_X]], ignore_index=True)
        y = pd.concat([y, y_add], ignore_index=True)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    print_distribution(y_train, "Training set") # Print distribution of training set
    print_distribution(y_test, "Test set") # Print distribution of test set

    # Feature reduction
    X_train, selected = fr.shap_feature_importance(X_train, y_train, shap_subset_size=50)
    X_test = fr.remove_shap_low_importance_features(X_test, selected)

    X_train, to_drop = fr.remove_correlated_features(X_train, threshold=0.9)
    X_test = X_test.drop(columns=to_drop)
    
    # Save data and artifact for later use
    data_to_save = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
    }
    basePath = BASE_PATH + "data/processed_data/"
    print("DatasetNames", dataset_names)
    dataset_name = "data_" + "_".join(name[4:] for name in dataset_names)
    artifact_path = dl.save_dataset_artifact(data_to_save, dataset_name, basePath)
    joblib.dump({"artifact_path": artifact_path}, os.path.join(basePath, "LATEST.joblib"))

    return X_train, X_test, y_train, y_test


def preprocess_data_predict(df, dataset_name, horizon=HORIZON, window_size=30, add_label=True):
    '''Full preprocessing pipeline for prediction'''

    # Load the latest preprocessed dataset artifact to get the feature names used for training
    latest = joblib.load(os.path.join("./data/processed_data/", "LATEST.joblib"))
    DATASET_PATH = latest["artifact_path"]

    artifact = joblib.load(DATASET_PATH)

    FEATURES = artifact["feature_names"]

    if add_label:
        create_future_target(df, horizon=horizon) # Create future target

    # Feature engineering and windowing
    X, y = feature_engineering_windowing(df, dataset_name, window_size=window_size, labels=add_label)

    # Align features of the prediction dataset with the features used for training
    X = align_features(X, FEATURES)

    X = X.drop(columns=['LogId']) # Drop LogId for prediction

    return X, y
