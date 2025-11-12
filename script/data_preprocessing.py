# File extension imports
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import script.data_modeling as dm


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


def load_split_data(dataset_name):
    '''Load pre-saved processed data from CSV files'''
    X_train = pd.read_csv(f'../data/processed_data/{dataset_name}_X_train.csv')
    X_val = pd.read_csv(f'../data/processed_data/{dataset_name}_X_val.csv')
    X_test = pd.read_csv(f'../data/processed_data/{dataset_name}_X_test.csv')
    y_train = pd.read_csv(f'../data/processed_data/{dataset_name}_y_train.csv')
    y_val = pd.read_csv(f'../data/processed_data/{dataset_name}_y_val.csv')
    y_test = pd.read_csv(f'../data/processed_data/{dataset_name}_y_test.csv')

    return X_train, X_val, X_test, y_train.squeeze(), y_val.squeeze(), y_test.squeeze()


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

    X_train, X_val, X_test = X_train.copy(), X_val.copy(), X_test.copy()

    numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()

    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    print("⚙️ Features scaled using StandardScaler")

    return X_train, X_val, X_test


def get_feature_importance(model, X_train):
    '''Get feature importance from the trained model'''

    importances = model.feature_importances_
    feature_names = X_train.columns
    feat_imp = pd.Series(importances, index=feature_names)
    feat_imp = feat_imp / feat_imp.sum()

    return feat_imp

def remove_low_importance_features(X_train, X_val, X_test, feat_imp, threshold=0.04):
    ''' Remove low-importance features '''

    low_importance = feat_imp[feat_imp < threshold].index
    X_train_reduced, X_val_reduced, X_test_reduced = reduce_features(X_train, X_val, X_test, low_importance)

    print(f"⚙️ Removed {len(low_importance)} low-importance features (importance < {threshold})")
    print(f"    Remaining features: {X_train_reduced.shape[1]}")

    return X_train_reduced, X_val_reduced, X_test_reduced


def remove_highly_correlated_features(X_train_reduced, X_val_reduced, X_test_reduced, threshold=0.9):
    ''' Remove one of each pair of highly correlated features '''

    corr_matrix = X_train_reduced.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    X_train_reduced, X_val_reduced, X_test_reduced = reduce_features(X_train_reduced, X_val_reduced, X_test_reduced, to_drop)

    print(f"⚙️ Removed {len(to_drop)} highly correlated features (correlation > {threshold})")
    print(f"    Remaining features: {X_train_reduced.shape[1]}")

    return X_train_reduced, X_val_reduced, X_test_reduced


def reduce_features(X_train, X_val, X_test, features_to_remove):
    '''Remove specified features (columns) from datasets'''

    X_train_reduced = X_train.drop(columns=features_to_remove)
    X_val_reduced = X_val.drop(columns=features_to_remove)
    X_test_reduced = X_test.drop(columns=features_to_remove)

    return X_train_reduced, X_val_reduced, X_test_reduced


def save_data(X_train, X_val, X_test, y_train, y_val, y_test, dataset_name):
    '''Save datasets to CSV files'''
    X_train.to_csv(f'../data/processed_data/{dataset_name}_X_train.csv', index=False)
    X_val.to_csv(f'../data/processed_data/{dataset_name}_X_val.csv', index=False)
    X_test.to_csv(f'../data/processed_data/{dataset_name}_X_test.csv', index=False)
    y_train.to_csv(f'../data/processed_data/{dataset_name}_y_train.csv', index=False)
    y_val.to_csv(f'../data/processed_data/{dataset_name}_y_val.csv', index=False)
    y_test.to_csv(f'../data/processed_data/{dataset_name}_y_test.csv', index=False)


def preprocess_data(df, dataset_name, scale=True, remove_low_imp=True, remove_corr=True):
    '''Full preprocessing pipeline'''

    df = sort_values_by_timestamp(df)

    create_target_column(df)
    create_future_target(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    if scale:
        X_train, X_val, X_test = scale_features(X_train, X_val, X_test)

    if remove_low_imp:
        model = dm.train_and_evaluate_rf(X_train, X_val, y_train, y_val)
        feat_imp = get_feature_importance(model, X_train)
        X_train, X_val, X_test = remove_low_importance_features(X_train, X_val, X_test, feat_imp)

    if remove_corr:
        X_train, X_val, X_test = remove_highly_correlated_features(X_train, X_val, X_test)

    save_data(X_train, X_val, X_test, y_train, y_val, y_test, dataset_name)

    return X_train, X_val, X_test, y_train, y_val, y_test