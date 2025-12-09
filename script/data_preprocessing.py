# File extension imports
import glob
import shap
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from . import feature_engineering as fe
from . import oversampling as ov


# Define scaler path
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
scaler_path = os.path.join(BASE_DIR, '..', 'models', 'standard_scaler.pkl')
scaler_path = os.path.abspath(scaler_path)

#### Functions for data preprocessing ###
def read_unify_data(path="../data/raw_data/data1/*.csv"):
    '''Read a CSV file with unified separator and decimal'''

    with open(path, 'r') as file:
        heaeder_line = file.readline()

    sep = ';' if ';' in heaeder_line else ','

    decimal = '.'

    with open(path, 'r') as file:
        for _ in range(5):
            line = file.readline()
            if ',' in line and sep == ';':
                decimal = ','
                break
            if '.' in line and sep == ',':
                decimal = '.'
                break
    
    return pd.read_csv(path, sep=sep, decimal=decimal)


COLUMN_RENAME_MAP = {
    "Time": "Time",
    "Flow rate (Arith. Mean)": "Flow rate (Mean)",
    "Pressure before pump (Arith. Mean)": "TS inlet pressure (Mean)",
    "Pressure before pump (Arith. Mean)": "TS outlet pressure (Mean)",
    "Pressure after pump (Arith. Mean)": "Pump outlet pressure (Mean)",
    "Temperature TS inlet (Arith. Mean)": "Temperature TS inlet (Mean)",
    "Temperature TS outlet (Arith. Mean)": "Temperature TS outlet (Mean)",
    "Tank temperature (Arith. Mean)": "Tank temperature (Mean)",
    "Bypass temperature (Arith. Mean)": "Bypass temperature (Mean)",
    "Differential pressure (Arith. Mean)": "Differential pressure (Mean)",
}

def standardize_column_names(df):
    df = df.rename(columns=COLUMN_RENAME_MAP)
    return df


def load_data(path="../data/raw_data/data1/*.csv"):
    '''Load and concatenate CSV files from a given path'''

    files = sorted(glob.glob(path))
    
    df_list = [read_unify_data(f) for f in files]

    df = pd.concat(df_list)

    df['Time'] = df['Time'].str.split(' ').str[1] if ' ' in df['Time'].iloc[0] else df['Time']

    df['Time'] = df['Time'].str.replace(',', '.', regex=False)

    df['Time'] = pd.to_datetime(df['Time'], format="%H:%M:%S.%f")
    
    # Sort by Time 
    df.set_index('Time', inplace=True)
    df.sort_index(inplace=True)

    # Create Elapsed_seconds column to track time progression
    #df['Elapsed_seconds'] = (df.index - df.index[0]).total_seconds()

    df.reset_index(drop=True, inplace=True)

    df = standardize_column_names(df)

    return df


def load_split_data(data: list, dataset_name: str, base_path="../data/processed_data/"):
    '''Load datasets to CSV files'''
    loaded_data = {}

    for key in data:
        loaded_data[key] = pd.read_csv(f"{base_path}{dataset_name}_{key}.csv", index_col=0)
    print(f"Load data with base name: {dataset_name}")

    return loaded_data
    

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


def train_val_test_split(X,y,test_size=0.01):
    '''Split data into train, validation, and test sets without shuffling'''
    n = len(X)
    test_start_idx = int((1 - test_size) * n)
    val_start_idx = int((1 - 2 * test_size) * n)

    X_train = X.iloc[:val_start_idx]
    y_train = y.iloc[:val_start_idx]

    # X_val = X.iloc[val_start_idx:test_start_idx]
    # y_val = y.iloc[val_start_idx:test_start_idx]

    X_test = X.iloc[test_start_idx:]
    y_test = y.iloc[test_start_idx:]

    return X_train, X_test, y_train, y_test


def split_data(df):
    '''Split data into features and target, then into train, val, test sets'''

    X = df.drop(columns=['Plug', 'Plug_future'])
    y = df['Plug_future'].squeeze()

    print("y shape :", y.shape)

    X_train, X_test, y_train, y_test = train_val_test_split(X, y)

    print("y shape after split:", y_train.shape, y_test.shape)
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
    numeric_cols = [col for col in numeric_cols if col not in ['Plug', 'Plug_future', 'Anomaly']]

    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    print("⚙️ Features scaled using StandardScaler")

    #export scalar to be used later during inference
    joblib.dump(scaler, scaler_path)

    return X_train, X_test


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


def shap_feature_importance(X_train, y_train, shap_subset_size=100):
    ''' 
    Calculate SHAP feature importance for the given model and training data, and
    remove features with low importance. 
    '''

    print("DEBUG: X_train shape entering SHAP:", X_train.shape)
    print("DEBUG: first 5 columns:", list(X_train.columns[:5]))
    

    baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    baseline.fit(X_train, y_train)
    print("🛠️ Baseline model trained.")

    shap_idx = np.arange(max(0, len(X_train) - shap_subset_size), len(X_train))
    print("SHAP calculation indices:", shap_idx)
    X_shap = X_train.iloc[shap_idx]
    print("Calculating SHAP values on subset of size:", X_shap.shape)

    explainer = shap.TreeExplainer(baseline)

    raw_shap = explainer.shap_values(X_shap, check_additivity=False)

    if isinstance(raw_shap, list):
        # Classic SHAP shape: list[class] → (n_samples, n_features)
        shap_values = raw_shap[1]
    else:
        # SHAP sometimes returns (n_samples, n_features, 2)
        if raw_shap.ndim == 3:
            shap_values = raw_shap[..., 1]   # take only class-1 contributions
        else:
            shap_values = raw_shap

    importance = np.mean(np.abs(shap_values), axis=0)  

    print("SHAP importance shape:", importance.shape)
    print("Top 10 important features:", list(X_train.columns[np.argsort(importance)[-10:]]))

    threshold = np.percentile(importance, 30)         
    selected = importance > threshold    

    if selected.sum() == 0:
        print("Warning: No features selected based on SHAP importance. Keeping top 10 features.")            
        idx = np.argsort(importance)[-10:]
        selected = np.zeros_like(importance, dtype=bool)
        selected[idx] = True

    always_keep = ['Flow rate (Mean)', 'Pump outlet pressure (Mean)', 'Anomaly']
    always_keep_idx = [X_train.columns.get_loc(col) for col in always_keep if col in X_train.columns]
    selected[always_keep_idx] = True

    X_train_reduced = X_train.loc[:, selected]

    print("Original features:", X_train.shape[1])
    print("Reduced features:", X_train_reduced.shape[1])

    # save selected mask for later use
    shap_path = os.path.join(BASE_DIR, '..', 'models', 'shap_selected_mask.pkl')
    shap_path = os.path.abspath(shap_path)
    joblib.dump(selected, shap_path)

    return X_train_reduced, selected


def remove_shap_low_importance_features(X, selected):
    '''Remove low importance features based on SHAP selection mask'''

    return X.loc[:, selected]


def remove_correlated_features(X, threshold=0.9):
    '''Remove highly correlated features based on a correlation threshold'''

    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    print("Correlation matrix:", corr_matrix)

    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]

    X_reduced = X.drop(columns=to_drop)

    print(f"⚙️ Removed {len(to_drop)} correlated features with threshold > {threshold}")
    print("Remaining features:", X_reduced.shape[1])

    return X_reduced, to_drop

def save_data(data: dict, dataset_name: str, base_path="../data/processed_data/"):
    '''Save datasets to CSV files'''
    for key, df in data.items():
        df.to_csv(f"{base_path}{dataset_name}_{key}.csv", index=False)
    print(f"💾 Saved data with base name: {dataset_name}")


def align_features(df, FEATURES):
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0 

    df = df[FEATURES]

    return df
    

def preprocess_data(df, dataset_name):
    '''Full preprocessing pipeline for model selection and training'''
    
    create_target_column(df) # Makes plug = 1/0 column, based on flow/pressure threshold
    create_future_target(df) # Creates Plug_future column by shifting Plug column by -10

    df_feat = df.select_dtypes(include=['number']).copy()
    df_feat = df_feat.drop(columns=['Plug', 'Plug_future', 'Anomaly'])
    df = fe.build_time_features(df, sensor_cols=df_feat.columns.tolist())

    print("DEBUG: df shape after build_time_features:", df.shape)
    print("DEBUG: first 10 columns after FE:", list(df.columns[:10]))


    X_train, X_test, y_train, y_test = split_data(df) # Splits data into X and y, then into train/test sets. Removes Plug and Plug_future from X
    print_distribution(y_train, "Training set")
    print_distribution(y_test, "Test set")

    # X_train = fe.rolling_features(X_train)
    # X_test = fe.rolling_features(X_test)

    X_train, selected = shap_feature_importance(X_train, y_train, shap_subset_size=50)
    X_test = remove_shap_low_importance_features(X_test, selected)

    X_train, to_drop = remove_correlated_features(X_train, threshold=0.9)
    X_test = X_test.drop(columns=to_drop)


    X_train, X_test = scale_features(X_train, X_test)

    X_train, y_train = fe.augment_minority_continuous_timeseries(X_train, y_train)

    data_to_save = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

    save_data(data_to_save, dataset_name, base_path="data/processed_data/")

    FEATURES = X_train.columns.tolist()
    FEATURES_PATH = os.path.join(BASE_DIR, '..', 'models', 'features_list.pkl')
    FEATURES_PATH = os.path.abspath(FEATURES_PATH)
    joblib.dump(FEATURES, FEATURES_PATH)

    return X_train, X_test, y_train, y_test


def preprocess_data_predict(df):
    '''Full preprocessing pipeline for prediction'''

    FEATURES_PATH = os.path.join(BASE_DIR, '..', 'models', 'features_list.pkl')
    FEATURES_PATH = os.path.abspath(FEATURES_PATH)
    FEATURES = joblib.load(open(FEATURES_PATH, "rb"))

    create_target_column(df)
    create_future_target(df)

    df_feat = df.select_dtypes(include=['number']).copy()
    df_feat = df_feat.drop(columns=['Plug', 'Plug_future', 'Anomaly'])
    df = fe.build_time_features(df, sensor_cols=df_feat.columns.tolist())

    X = df.drop(columns=['Plug', 'Plug_future'])
    y = df['Plug_future'].squeeze()

    #X = fe.rolling_features(X)
    # Don´t need to remove shap low importance features here, as we will align to training features
    X = align_features(X, FEATURES)

    scalar = joblib.load(scaler_path)
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['Plug', 'Plug_future', 'Anomaly']]
    unscaled_X = X.copy()
    X[numeric_cols] = scalar.transform(X[numeric_cols])

    return X, y, unscaled_X




''''
Flow rate (Mean),TS outlet pressure (Mean),Pump outlet pressure (Mean),Temperature TS outlet (Mean),Temperature TS inlet (Mean),Bypass temperature (Mean),Anomaly,Flow rate (Mean)_lag1,Flow rate (Mean)_lag5,Flow rate (Mean)_lag10,TS outlet pressure (Mean)_lag5,TS outlet pressure (Mean)_lag10,TS inlet pressure (Mean)_lag1,TS inlet pressure (Mean)_lag10,Pump outlet pressure (Mean)_lag5,Pump outlet pressure (Mean)_lag10,Temperature TS outlet (Mean)_lag1,Temperature TS outlet (Mean)_lag5,Temperature TS outlet (Mean)_lag10,Tank temperature (Mean)_lag1,Tank temperature (Mean)_lag10,Temperature TS inlet (Mean)_lag1,Temperature TS inlet (Mean)_lag5,Temperature TS inlet (Mean)_lag10,Bypass temperature (Mean)_lag1,Bypass temperature (Mean)_lag5,Bypass temperature (Mean)_lag10,Flow rate (Mean)_rollmean_10,Flow rate (Mean)_rollstd_10,Flow rate (Mean)_rollmin_10,Flow rate (Mean)_rollmax_10,Flow rate (Mean)_rollmean_30,Flow rate (Mean)_rollstd_30,Flow rate (Mean)_rollmin_30,Flow rate (Mean)_rollmax_30,Flow rate (Mean)_rollslope_30,Flow rate (Mean)_rollmean_60,Flow rate (Mean)_rollstd_60,Flow rate (Mean)_rollmin_60,Flow rate (Mean)_rollmax_60,TS outlet pressure (Mean)_rollmean_10,TS outlet pressure (Mean)_rollmax_10,TS outlet pressure (Mean)_rollmean_30,TS outlet pressure (Mean)_rollmin_30,TS outlet pressure (Mean)_rollmax_30,TS outlet pressure (Mean)_rollslope_30,TS outlet pressure (Mean)_rollmean_60,TS outlet pressure (Mean)_rollstd_60,TS outlet pressure (Mean)_rollmin_60,TS outlet pressure (Mean)_rollslope_60,TS inlet pressure (Mean)_rollmean_10,TS inlet pressure (Mean)_rollstd_10,TS inlet pressure (Mean)_rollmean_30,TS inlet pressure (Mean)_rollstd_30,TS inlet pressure (Mean)_rollmin_30,TS inlet pressure (Mean)_rollmean_60,TS inlet pressure (Mean)_rollmin_60,TS inlet pressure (Mean)_rollmax_60,TS inlet pressure (Mean)_rollslope_60,Pump outlet pressure (Mean)_rollmean_10,Pump outlet pressure (Mean)_rollstd_10,Pump outlet pressure (Mean)_rollmax_10,Pump outlet pressure (Mean)_rollstd_30,Pump outlet pressure (Mean)_rollmin_30,Pump outlet pressure (Mean)_rollmax_30,Pump outlet pressure (Mean)_rollslope_30,Pump outlet pressure (Mean)_rollmean_60,Pump outlet pressure (Mean)_rollstd_60,Pump outlet pressure (Mean)_rollmin_60,Pump outlet pressure (Mean)_rollmax_60,Temperature TS outlet (Mean)_rollmean_10,Temperature TS outlet (Mean)_rollstd_10,Temperature TS outlet (Mean)_rollmax_10,Temperature TS outlet (Mean)_rollslope_10,Temperature TS outlet (Mean)_rollmean_30,Temperature TS outlet (Mean)_rollmin_30,Temperature TS outlet (Mean)_rollmax_30,Temperature TS outlet (Mean)_rollslope_30,Temperature TS outlet (Mean)_rollmean_60,Temperature TS outlet (Mean)_rollstd_60,Temperature TS outlet (Mean)_rollmin_60,Temperature TS outlet (Mean)_rollmax_60,Tank temperature (Mean)_rollmean_10,Tank temperature (Mean)_rollmin_10,Tank temperature (Mean)_rollslope_10,Tank temperature (Mean)_rollmean_30,Tank temperature (Mean)_rollstd_30,Tank temperature (Mean)_rollmin_30,Tank temperature (Mean)_rollmax_30,Tank temperature (Mean)_rollslope_30,Tank temperature (Mean)_rollmean_60,Tank temperature (Mean)_rollstd_60,Tank temperature (Mean)_rollmin_60,Tank temperature (Mean)_rollmax_60,Temperature TS inlet (Mean)_rollmean_10,Temperature TS inlet (Mean)_rollstd_10,Temperature TS inlet (Mean)_rollmin_10,Temperature TS inlet (Mean)_rollmax_10,Temperature TS inlet (Mean)_rollmean_30,Temperature TS inlet (Mean)_rollmin_30,Temperature TS inlet (Mean)_rollmax_30,Temperature TS inlet (Mean)_rollmean_60,Temperature TS inlet (Mean)_rollmin_60,Temperature TS inlet (Mean)_rollmax_60,Temperature TS inlet (Mean)_rollslope_60,Bypass temperature (Mean)_rollmean_10,Bypass temperature (Mean)_rollmin_10,Bypass temperature (Mean)_rollmax_10,Bypass temperature (Mean)_rollmean_30,Bypass temperature (Mean)_rollstd_30,Bypass temperature (Mean)_rollmin_30,Bypass temperature (Mean)_rollmax_30,Bypass temperature (Mean)_rollslope_30,Bypass temperature (Mean)_rollmean_60,Bypass temperature (Mean)_rollstd_60,Bypass temperature (Mean)_rollmin_60,Bypass temperature (Mean)_rollmax_60,Flow rate (Mean)_diff5,TS inlet pressure (Mean)_diff1,TS inlet pressure (Mean)_diff10,Pump outlet pressure (Mean)_diff1,Pump outlet pressure (Mean)_diff10,Temperature TS outlet (Mean)_diff10,Tank temperature (Mean)_diff1,Tank temperature (Mean)_diff10,Temperature TS inlet (Mean)_diff1,Temperature TS inlet (Mean)_diff5,Temperature TS inlet (Mean)_diff10,Bypass temperature (Mean)_diff5,Flow rate (Mean)_to_TS inlet pressure (Mean)_ratio,Flow rate (Mean)_to_Pump outlet pressure (Mean)_ratio,Flow rate (Mean)_to_Temperature TS outlet (Mean)_ratio,Flow rate (Mean)_to_Tank temperature (Mean)_ratio,Flow rate (Mean)_to_Bypass temperature (Mean)_ratio,TS outlet pressure (Mean)_to_TS inlet pressure (Mean)_ratio,TS outlet pressure (Mean)_to_Tank temperature (Mean)_ratio,TS outlet pressure (Mean)_to_Temperature TS inlet (Mean)_ratio,TS inlet pressure (Mean)_to_Temperature TS outlet (Mean)_ratio,TS inlet pressure (Mean)_to_Tank temperature (Mean)_ratio,TS inlet pressure (Mean)_to_Temperature TS inlet (Mean)_ratio,Pump outlet pressure (Mean)_to_Temperature TS inlet (Mean)_ratio,Temperature TS outlet (Mean)_to_Tank temperature (Mean)_ratio,Tank temperature (Mean)_to_Temperature TS inlet (Mean)_ratio,Temperature TS inlet (Mean)_to_Bypass temperature (Mean)_ratio


'''