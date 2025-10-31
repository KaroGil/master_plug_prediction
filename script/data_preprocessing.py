# File extension imports
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path="../data/raw_data/data1/*.csv"):
    '''Load and concatenate CSV files from a given path'''

    files = sorted(glob.glob(path))
    df_list = [pd.read_csv(f, sep=";", decimal=",") for f in files]

    df = pd.concat(df_list)
    df['Time'] = pd.to_datetime(df['Time'], format="%H:%M:%S,%f")  
    df = df.set_index('Time')
    
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


def create_target_column(df):
    ''' Make the target column based on tresholds '''

    flow = df["Flow rate (Mean)"]
    pressure = df["Pump outlet pressure (Mean)"]

    # Thresholds
    flow_thresh = flow.median() * 0.7
    pressure_thresh = pressure.median() * 1.3

    df["Plug"] = np.where((flow < flow_thresh) & (pressure > pressure_thresh), 1, 0)


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


def remove_low_importance_features(X_train, X_val, X_test, feat_imp, threshold=0.04):
    ''' Remove low-importance features '''

    low_importance = feat_imp[feat_imp < threshold].index
    X_train_reduced, X_val_reduced, X_test_reduced = reduce_features(X_train, X_val, X_test, low_importance)

    return X_train_reduced, X_val_reduced, X_test_reduced


def remove_highly_correlated_features(X_train_reduced, X_val_reduced, X_test_reduced, threshold=0.9):
    ''' Remove one of each pair of highly correlated features '''

    corr_matrix = X_train_reduced.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    X_train_reduced, X_val_reduced, X_test_reduced = reduce_features(X_train_reduced, X_val_reduced, X_test_reduced, to_drop)

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