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


def create_future_target(df, shift=-200):
    '''Create target column 'Plug_future' by shifting 'Plug' column'''

    # Predict plug event 10 steps ahead
    df['Plug_future'] = df['Plug'].shift(shift)
    df.dropna(subset=['Plug_future'], inplace=True)


def add_logId_column(df, log_id):
    '''Add a LogId column to the dataframe to specify which log the data comes from'''
    df = df.copy()
    df['LogId'] = int(log_id[4:])
    return df


def split_data(X,y,test_size=0.01, test_logid=False):
    '''Split data into train, validation, and test sets without shuffling'''
    if test_logid: # instead of normal splitting, leave out one dataset
        X_test = X.loc[X['LogId'] == X['LogId'].max()]
        y_test = y.loc[y.index.isin(X_test.index)]
        X_train = X.loc[X['LogId'] != X['LogId'].max()]
        y_train = y.loc[y.index.isin(X_train.index)]
    else:
        n = len(X)
        test_start_idx = int((1 - test_size) * n)
        val_start_idx = int((1 - 2 * test_size) * n)

        X_train = X.iloc[:val_start_idx]
        y_train = y.iloc[:val_start_idx]

        X_test = X.iloc[test_start_idx:]
        y_test = y.iloc[test_start_idx:]

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
    numeric_cols = [col for col in numeric_cols if col not in ['Plug', 'Plug_future', 'Anomaly', 'LogId']]

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


def feature_engineering_windowing(df, log_id):
    '''Basic preprocessing pipeline without train/test split'''
    df = add_logId_column(df, log_id)

    print("Adding time derivative features...")
    df = fe.add_time_derivative_features(df)

    df = fe.feature_engineering_pipeline(df) 

    df_feat = df.select_dtypes(include=['number']).copy()

    X, y = w.prep_window(df, [x for x in df_feat.columns.tolist() if x not in ['Plug', 'Plug_future', 'LogId']])

    return X, y

def preprocess_data(df, dataset_name, additional_data = None, additional_data_name = None, BASE_PATH = ""):
    '''Full preprocessing pipeline for model selection and training'''

    X, y = feature_engineering_windowing(df, dataset_name)
    if additional_data is not None and additional_data_name is not None:
        for add_df, add_name in zip(additional_data, additional_data_name):
            print(f"Processing additional data: {add_name}") 
            X_add, y_add = feature_engineering_windowing(add_df, add_name)
            X = pd.concat([X, X_add], ignore_index=True)
            y = pd.concat([y, y_add], ignore_index=True)

    X_train, X_test, y_train, y_test = split_data(X, y, test_logid=True) # Splits data into X and y, then into train/test sets. Removes Plug and Plug_future from X
    print_distribution(y_train, "Training set")
    print_distribution(y_test, "Test set")

    X_train, selected = fr.shap_feature_importance(X_train, y_train, shap_subset_size=50)
    X_test = fr.remove_shap_low_importance_features(X_test, selected)

    X_train, to_drop = fr.remove_correlated_features(X_train, threshold=0.9)
    X_test = X_test.drop(columns=to_drop)
    
    #drop any remaining NaN values
    X_train.fillna(0,inplace=True)
    X_test.fillna(0,inplace=True) 
    y_train = y_train.loc[X_train.index]
    y_test = y_test.loc[X_test.index]

    X_train, X_test = scale_features(X_train, X_test)

    # print("Visualizing before augmentation")
    # from . import data_visualization as dv
    # all = X_train.copy()
    # all['Plug_future'] = y_train
    # dv.visualize_plug_event(all, name="Training Set Before Augmentation")
    # X_train, y_train = fe.augment_minority_continuous_timeseries(X_train, y_train)
    # all_after = X_train.copy()
    # all_after['Plug_future'] = y_train
    # dv.visualize_plug_event(all_after, name="Training Set After Augmentation")
    
    data_to_save = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    basePath = BASE_PATH + "data/processed_data/"
    dl.save_data(data_to_save, dataset_name, base_path=basePath)

    joblib.dump(X_train.columns.tolist(), FEATURES_PATH)

    return X_train, X_test, y_train, y_test


def preprocess_data_predict(df, dataset_name):
    '''Full preprocessing pipeline for prediction'''

    FEATURES = joblib.load(open(FEATURES_PATH, "rb"))

    X, y = feature_engineering_windowing(df, dataset_name)

    X = align_features(X, FEATURES)

    scalar = joblib.load(scaler_path)

    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['Plug', 'Plug_future', 'Anomaly', 'LogId']]
    unscaled_X = X.copy()
    X[numeric_cols] = scalar.transform(X[numeric_cols])

    X = X.drop(columns=['LogId'])

    return X, y, unscaled_X
