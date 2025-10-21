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

def create_target_column(df):
    # Make the target column based on tresholds
    flow = df["Flow rate (Mean)"]
    pressure = df["Pump outlet pressure (Mean)"]

    # thresholds
    flow_thresh = flow.median() * 0.7
    pressure_thresh = pressure.median() * 1.3

    df["Plug"] = np.where((flow < flow_thresh) & (pressure > pressure_thresh), 1, 0)


def train_val_test_split(X,y):
    '''Split data into train, validation, and test sets without shuffling'''
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.0095, random_state=42, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False)

    return X_train, X_val, X_test, y_train, y_val, y_test

def split_data(df):
    '''Split data into features and target, then into train, val, test sets'''
    X = df.drop(columns=['Plug', 'Plug_future'])
    y = df['Plug_future']

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    '''Standardize features using StandardScaler'''

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled


def reduce_features(X_train, X_val, X_test, features_to_remove):
    '''Remove specified features (columns) from datasets'''

    X_train_reduced = X_train.drop(columns=features_to_remove)
    X_val_reduced = X_val.drop(columns=features_to_remove)
    X_test_reduced = X_test.drop(columns=features_to_remove)

    return X_train_reduced, X_val_reduced, X_test_reduced

