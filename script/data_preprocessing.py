# File extension imports
import glob
import shap
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler

from . import feature_engineering as fe
from . import oversampling as ov


# Define scaler path
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
scaler_path = os.path.join(BASE_DIR, '..', 'models', 'standard_scaler.pkl')
scaler_path = os.path.abspath(scaler_path)

#### Functions for data preprocessing ###
def load_data(path="../data/raw_data/data1/*.csv"):
    '''Load and concatenate CSV files from a given path'''

    files = sorted(glob.glob(path))
    df_list = [pd.read_csv(f, sep=";", decimal=",") for f in files]

    df = pd.concat(df_list)
    df['Time'] = pd.to_datetime(df['Time'], format="%H:%M:%S,%f")
    df['Elapsed_seconds'] = (df['Time'] - df['Time'].iloc[0]).dt.total_seconds()

    df.drop(columns=['Time'], inplace=True)

    df = df.set_index('Elapsed_seconds')

    return df


def load_split_data(data: list, dataset_name: str, base_path="../data/processed_data/"):
    '''Load datasets to CSV files'''
    loaded_data = {}

    for key in data:
        loaded_data[key] = pd.read_csv(f"{base_path}{dataset_name}_{key}.csv", index_col=0)
    print(f"Load data with base name: {dataset_name}")

    return loaded_data
    

def sort_values_by_timestamp(df):
    ''' Sort DataFrame by its timestamp index '''
    return df.sort_index()


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
        df["Anomaly"] = 0

        # Reset false positives
        for i in range(1, len(df) - mask):
            if df["Plug"].iloc[i] == 1 and (flow.iloc[i+mask] > flow_thresh or flow.iloc[i-mask] > flow_thresh):
                df.loc[df.index[i], "Plug"] = 0

                df.loc[df.index[max(0, i-mask): min(len(df), i+mask)], "Anomaly"] = 1

    if "pressure" in thresholds:
        pressure = df["Pump outlet pressure (Mean)"]
        pressure_thresh = pressure.median() * 1.3

        print(f"⚙️ Pressure threshold set to: {pressure_thresh:.2f}")
        
        pressure_plug = pressure > pressure_thresh
        
        df["Plug"] = np.where((pressure_plug), 1, df["Plug"])


def create_future_target(df, shift=-10):
    '''Create target column 'Plug_future' by shifting 'Plug' column'''

    # Predict plug event 10 steps ahead
    df['Plug_future'] = df['Plug'].shift(shift)
    df.dropna(subset=['Plug_future'], inplace=True)


def train_val_test_split(X,y,test_size=0.0047):
    '''Split data into train, validation, and test sets without shuffling'''
    n = len(X)
    test_start_idx = int((1 - test_size) * n)
    val_start_idx = int((1 - 2 * test_size) * n)

    X_train = X[:val_start_idx]
    y_train = y[:val_start_idx]

    X_val = X[val_start_idx:test_start_idx]
    y_val = y[val_start_idx:test_start_idx] 
    X_test = X[test_start_idx:]
    y_test = y[test_start_idx:]

    return X_train, X_val, X_test, y_train, y_val, y_test


def split_data(df):
    '''Split data into features and target, then into train, val, test sets'''

    X = df.drop(columns=['Plug', 'Plug_future'])
    y = df['Plug_future'].squeeze()

    print("y shape :", y.shape)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    print("y shape after split:", y_train.shape, y_val.shape, y_test.shape)
    return X_train, X_val, X_test, y_train, y_val, y_test


def print_distribution(y, name):
    '''Print class distribution of the target variable'''

    unique, counts = np.unique(y, return_counts=True)
    distribution = dict(zip(unique, counts))
    print(f"{name} distribution:")
    for key, value in distribution.items():
        print(f"  Class {key}: {value} samples")


def scale_features(X_train, X_val, X_test, scaler_path="scaler.pkl"):
    """
    Scale features using StandardScaler.
    Works for NumPy arrays (not DataFrames).
    """

    # Sanity check
    if X_train.shape[1] == 0:
        raise ValueError("❌ ERROR: No features left to scale.")

    # Convert to float (SVM/RF/XGB require numeric)
    X_train = X_train.astype(float)
    X_val   = X_val.astype(float)
    X_test  = X_test.astype(float)

    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform val/test
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Save for inference
    joblib.dump(scaler, scaler_path)

    print("⚙️ Features scaled using StandardScaler")

    return X_train_scaled, X_val_scaled, X_test_scaled



def descale_features(df):
    '''Inverse transform standardized features'''
    
    scaler = joblib.load(scaler_path)

    df = df.copy()

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['Plug', 'Plug_future', 'Anomaly']]

    df[numeric_cols] = scaler.inverse_transform(df[numeric_cols])

    print("⚙️ Features descaled using StandardScaler inverse transform")

    return df

def reduce_features(X_train, X_val, X_test, features_to_remove):
    '''Remove specified features (columns) from datasets'''

    X_train_reduced = X_train.drop(columns=features_to_remove)
    X_val_reduced = X_val.drop(columns=features_to_remove)
    X_test_reduced = X_test.drop(columns=features_to_remove)

    return X_train_reduced, X_val_reduced, X_test_reduced


def shap_feature_importance(X_train, y_train, feature_names, shap_subset_size=1000):

    # 1. Train baseline RF model for SHAP importance
    baseline = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    baseline.fit(X_train, y_train)
    print("🛠️ Baseline SHAP model trained.")

    # 2. Subset for SHAP computation
    shap_idx = np.arange(max(0, len(X_train) - shap_subset_size), len(X_train))
    X_shap = X_train[shap_idx]

    # 3. Use TreeExplainer (fast + correct for RF/XGB)
    explainer = shap.TreeExplainer(baseline)
    shap_values = explainer.shap_values(X_shap)

    # 4. Handle binary classification format
    if isinstance(shap_values, list):
        shap_pos = shap_values[1]   # positive class
    else:
        shap_pos = shap_values

    # 5. Compute global feature importance
    importance = np.mean(np.abs(shap_pos), axis=0)

    # 6. Threshold-based selection
    threshold = np.percentile(importance, 30)
    selected_mask = importance > threshold

    # 7. Always-keep features (if needed)
    always_keep = []
    for feat in always_keep:
        if feat in feature_names:
            idx = feature_names.index(feat)
            selected_mask[idx] = True

    # 8. Reduce X_train
    X_train_reduced = X_train[:, selected_mask]

    # 9. Store selection
    joblib.dump(selected_mask, "shap_selected_mask.pkl")

    return X_train_reduced, selected_mask


def remove_shap_low_importance_features(X, selected):
    '''Remove low importance features based on SHAP selection mask'''

    return X[:, selected]


def save_data(data: dict, dataset_name: str, base_path="../data/processed_data/"):
    '''Save datasets to CSV files'''
    for key, df in data.items():
        df.to_csv(f"{base_path}{dataset_name}_{key}.csv", index=True)
    print(f"💾 Saved data with base name: {dataset_name}")
    

def preprocess_data(df, dataset_name):

    df = sort_values_by_timestamp(df)

    create_target_column(df) 
    create_future_target(df)

    X, y = ov.window_data(
        df,
        window_seconds=10,
        sampling_rate=0.05
    )

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
    print_distribution(y_train, "Training set") 
    print_distribution(y_val, "Validation set") 
    print_distribution(y_test, "Test set")

    X_train_bal, y_train_bal = ov.oversample_within_windows(X_train, y_train)

    print("After oversampling:")
    print_distribution(y_train_bal, "Training set") 

    X_train_scaled, X_val_scaled, X_test_scaled = scale_features(
        X_train_bal, X_val, X_test
    )

    data_to_save = {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train_bal,
        'y_val': y_val,
        'y_test': y_test
    }
    save_data(data_to_save, dataset_name, base_path="data/processed_data/")

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train_bal, y_val, y_test



def preprocess_data_predict(df):
    '''Full preprocessing pipeline for prediction'''

    df = sort_values_by_timestamp(df)

    create_target_column(df)
    create_future_target(df)

    X = df.drop(columns=['Plug', 'Plug_future'])
    y = df['Plug_future'].squeeze()

    X = fe.rolling_features(X)

    shap_path = os.path.join(BASE_DIR, '..', 'models', 'shap_selected_mask.pkl')
    shap_path = os.path.abspath(shap_path)
    
    shap_selected = joblib.load(shap_path)

    X = remove_shap_low_importance_features(X, shap_selected)

    scalar = joblib.load(scaler_path)
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['Plug', 'Plug_future', 'Anomaly']]
    unscaled_X = X.copy()
    X[numeric_cols] = scalar.transform(X[numeric_cols])

    return X, y, unscaled_X