# File extension imports
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import window as w
from . import data_loader as dl
from . import feature_reduction as fr
from . import feature_engineering as fe
from .config import get_config

# Load config
cfg = get_config()

target_col = cfg["data"]["target"]
non_feature_columns = cfg["data"]["non_feature_columns"]
horizon = cfg["experiment"]["horizon"]
test_sets = cfg["data"]["test_sets"]

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', '..', 'models')
MODELS_DIR = os.path.abspath(MODELS_DIR)

scaler_path = os.path.join(MODELS_DIR, 'standard_scaler.pkl')
scaler_path = os.path.abspath(scaler_path)

FEATURES_PATH = os.path.join(MODELS_DIR, 'features_list.pkl')
FEATURES_PATH = os.path.abspath(FEATURES_PATH)


def create_target_column(df, flow_thresh=0.9, pressure_thresh=1.3, thresholds=['flow'], mask=300):
    ''' Make the target column based on tresholds '''

    if ("Flow rate (Mean)" not in df.columns and "flow" in thresholds) or ("Pump outlet pressure (Mean)" not in df.columns and "pressure" in thresholds):
        raise ValueError("❌ ERROR: Required columns for target creation are missing.")

    if "flow" in thresholds:
        flow = df["Flow rate (Mean)"]
        flow_thresh = flow.median() * flow_thresh

        print(f"⚙️ Flow threshold set to: {flow_thresh:.2f}")

        # Initial plug labeling
        df["Plug"] = np.where((flow < flow_thresh ), 1, 0)

        # Reset false positives
        for i in range(1, len(df) - mask):
            if df["Plug"].iloc[i] == 1 and (flow.iloc[i+mask] > flow_thresh or flow.iloc[i-mask] > flow_thresh):
                df.loc[df.index[i], "Plug"] = 0

    if "pressure" in thresholds:
        pressure = df["Pump outlet pressure (Mean)"]
        pressure_thresh = pressure.median() * 1.3

        print(f"⚙️ Pressure threshold set to: {pressure_thresh:.2f}")
        
        pressure_plug = pressure > pressure_thresh
        
        df["Plug"] = np.where((pressure_plug), 1, df["Plug"])


def create_future_target(df, horizon=horizon):
    '''Create target column 'Plug_future' by shifting 'Plug' column'''

    df[target_col] = (df['Plug'].shift(-1).rolling(window=horizon, min_periods=1).max())
    df.dropna(subset=[target_col], inplace=True)


def split_data(X,y):
    '''Split data into train, validation, and test sets without shuffling'''

    if test_sets:
            test_set_log_id = test_sets[-1] # Use the last test set specified in config
    else:
        test_set_log_id = X['LogId'].max()
    X_test = X.loc[X['LogId'] == test_set_log_id]
    y_test = y.loc[y.index.isin(X_test.index)]
    
    if test_sets: 
        X_train = X.loc[~X['LogId'].isin(test_sets)]
    else: 
        X_train = X.loc[X['LogId'] != test_set_log_id]
    
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


def scale_features(X_train, X_test):
    '''Standardize features using StandardScaler'''

    if X_train.shape[1] == 0 or X_test.shape[1] == 0:
        raise ValueError("❌ ERROR: No features left after SHAP/correlation. Check thresholds.")

    X_train, X_test = X_train.copy(), X_test.copy()

    numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in non_feature_columns]

    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    print("⚙️ Features scaled using StandardScaler")

    #export scalar to be used later
    joblib.dump(scaler, scaler_path)

    return X_train, X_test


def align_features(df, FEATURES):
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0 

    df = df[FEATURES]

    return df


def feature_engineering_windowing(df, dataset_name="data1"):
    '''Basic preprocessing pipeline without train/test split'''

    if 'LogId' not in df.columns:
        df['LogId'] = dataset_name[4:]

    df = fe.feature_engineering_pipeline(df) 

    print("Adding time derivative features...")
    df = fe.add_time_derivative_features(df)

    print("Adding plug index feature...")
    df = fe.plug_index(df)

    df_feat = df.select_dtypes(include=['number']).copy()

    X, y = w.prep_window(df, [x for x in df_feat.columns.tolist() if x not in  non_feature_columns])

    return X, y

def resample_data_1_Hz(dataset):
    """ Resamples data to 1Hz (1 sample per second) frequency to make sure each dataset has the same sample rate. """

    dataset.index = pd.to_datetime(dataset.index)

    label_cols = [c for c in ["Plug", "Plug_future"] if c in dataset.columns]
    numeric_cols = dataset.select_dtypes(include="number").columns.difference(label_cols)
    other_cols = dataset.select_dtypes(exclude="number").columns

    print(f"Label cols: {label_cols} \n Numeric cols: {numeric_cols} \n Other cols: {other_cols}")

    agg_dict = {col: "mean" if col in numeric_cols else "first" for col in dataset.columns}
    agg_dict.update({col: "max" for col in label_cols})

    resampled_dataset = dataset.resample("1s").agg(agg_dict).dropna(how="all")

    if "Plug_future" in dataset.columns:
        resampled_dataset = resampled_dataset.drop(columns = ["Plug_future"])

    return resampled_dataset

def preprocess_data(datasets, dataset_names, horizon=horizon, BASE_PATH = ""):
    '''Full preprocessing pipeline for model selection and training'''
    # Resample dataset so every dataset is sampled with 1.0s intervals. 
    print("Resampling datasets...")
    for dataset in datasets:
        dataset = resample_data_1_Hz(dataset)
    
    print(f"Processing data: {dataset_names[0]}")
    create_future_target(datasets[0], horizon=horizon)

    X, y = feature_engineering_windowing(datasets[0], dataset_names[0])

    for df, name in zip(datasets[1:], dataset_names[1:]):
        print(f"Processing data: {name}") 
        create_future_target(df, horizon=horizon)
        X_add, y_add = feature_engineering_windowing(df, name)
        common_cols_X = sorted(set.intersection(*(set(df.columns) for df in [X, X_add])))

        X = pd.concat([X[common_cols_X], X_add[common_cols_X]], ignore_index=True)
        y = pd.concat([y, y_add], ignore_index=True)

    X_train, X_test, y_train, y_test = split_data(X, y)
    print_distribution(y_train, "Training set")
    print_distribution(y_test, "Test set")

    # Feature reduction
    X_train, selected = fr.shap_feature_importance(X_train, y_train, shap_subset_size=50)
    X_test = fr.remove_shap_low_importance_features(X_test, selected)

    X_train, to_drop = fr.remove_correlated_features(X_train, threshold=0.9)
    X_test = X_test.drop(columns=to_drop)
    
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


def preprocess_data_predict(df, dataset_name, horizon=horizon):
    '''Full preprocessing pipeline for prediction'''

    # Resample dataset so every dataset is sampled with 1.0s intervals. 
    print("Resampling datasets...")
    print(f"Resampling dataset {dataset_name}")
    df = resample_data_1_Hz(df)

    latest = joblib.load(os.path.join("./data/processed_data/", "LATEST.joblib"))
    DATASET_PATH = latest["artifact_path"]

    artifact = joblib.load(DATASET_PATH)

    FEATURES = artifact["feature_names"]

    create_future_target(df, horizon=horizon)

    X, y = feature_engineering_windowing(df, dataset_name)

    X = align_features(X, FEATURES)

    X = X.drop(columns=['LogId'])

    return X, y
