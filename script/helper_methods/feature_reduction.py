import os
import shap
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', '..', 'models')
MODELS_DIR = os.path.abspath(MODELS_DIR)

shap_path = os.path.join(MODELS_DIR, 'shap_selected_mask.pkl')
shap_path = os.path.abspath(shap_path)


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

    always_keep = ['Flow rate (Mean)', 'Pump outlet pressure (Mean)', 'Anomaly', 'LogId']
    always_keep_idx = [X_train.columns.get_loc(col) for col in always_keep if col in X_train.columns]
    selected[always_keep_idx] = True

    X_train_reduced = X_train.loc[:, selected]

    print("Original features:", X_train.shape[1])
    print("Reduced features:", X_train_reduced.shape[1])

    # save selected mask for later use
    joblib.dump(selected, shap_path)

    return X_train_reduced, selected


def remove_shap_low_importance_features(X, selected):
    '''Remove low importance features based on SHAP selection mask'''

    return X.loc[:, selected]


def remove_correlated_features(X, threshold=0.9):
    '''Remove highly correlated features based on a correlation threshold'''

    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]

    X_reduced = X.drop(columns=to_drop)

    print(f"⚙️  Removed {len(to_drop)} correlated features with threshold > {threshold}")
    print("Remaining features:", X_reduced.shape[1])

    return X_reduced, to_drop
