# File extension imports
import glob
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import data_modeling as dm
import feature_engineering as fe


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

    df_sorted = df.sort_index()
    return df_sorted


def create_target_column(df, flow_thresh=0.9, pressure_thresh=1.3, thresholds=['flow'], mask=300):
    ''' Make the target column based on tresholds '''

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
                df["Plug"].iloc[i] = 0

                df["Anomaly"].iloc[i-mask:i+mask] = 1

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


def train_val_test_split(X,y):
    '''Split data into train, validation, and test sets without shuffling'''

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.0095, random_state=42, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False)

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


def scale_features(X_train, X_val, X_test):
    '''Standardize features using StandardScaler'''

    if X_train.shape[1] == 0 or X_val.shape[1] == 0 or X_test.shape[1] == 0:
        raise ValueError("❌ ERROR: No features left after SHAP/correlation. Check thresholds.")

    X_train, X_val, X_test = X_train.copy(), X_val.copy(), X_test.copy()

    numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['Plug', 'Plug_future', 'Anomaly']]

    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    print("⚙️ Features scaled using StandardScaler")

    #export scalar to be used later during inference
    scaler_path = '../models/standard_scaler.pkl'
    joblib.dump(scaler, scaler_path)

    return X_train, X_val, X_test


def descale_features(df):
    '''Inverse transform standardized features'''

    scaler = joblib.load('../models/standard_scaler.pkl')

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


def remove_correlated_features(df, threshold=0.9):
    '''Remove highly correlated features from DataFrame'''

    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    print(f"⚙️ Removing {len(to_drop)} correlated features with threshold > {threshold}: {to_drop}")

    selected = ~df.columns.isin(to_drop)

    if selected.sum() == 0:
        print("Warning: No features left after correlation removal. Adjust threshold.")            
        selected = np.ones(len(df.columns), dtype=bool)
    
    return selected


def save_data(data: dict, dataset_name: str, base_path="../data/processed_data/"):
    '''Save datasets to CSV files'''
    for key, df in data.items():
        df.to_csv(f"{base_path}{dataset_name}_{key}.csv", index=True)
    print(f"💾 Saved data with base name: {dataset_name}")
    

def preprocess_data(df, dataset_name):
    '''Full preprocessing pipeline'''

    df = sort_values_by_timestamp(df)

    create_target_column(df)
    create_future_target(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    X_train = fe.rolling_features(X_train)
    X_val = fe.rolling_features(X_val)
    X_test = fe.rolling_features(X_test)

    X_train, selected = dm.shap_feature_importance(X_train, y_train)
    X_val = dm.remove_shap_low_importance_features(X_val, selected)
    X_test = dm.remove_shap_low_importance_features(X_test, selected)

    corr_mask = remove_correlated_features(X_train, threshold=0.9)
    X_train = X_train.loc[:, corr_mask]
    X_val = X_val.loc[:, corr_mask]
    X_test = X_test.loc[:, corr_mask]

    X_train, X_val, X_test = scale_features(X_train, X_val, X_test)

    data_to_save = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }

    save_data(data_to_save, dataset_name)

    return X_train, X_val, X_test, y_train, y_val, y_test