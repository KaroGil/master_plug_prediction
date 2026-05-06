"""
Helper methods for feature reduction: SHAP-based feature selection and correlation filtering.
"""

import os
import shap
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from script.helper_methods.config import get_config

# Load config
cfg = get_config()
seed = cfg["experiment"]["random_state"]
always_keep_columns = cfg["data"]["always_keep_columns"]

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', '..', 'models')
MODELS_DIR = os.path.abspath(MODELS_DIR)

shap_path = os.path.join(MODELS_DIR, 'shap_selected_mask.pkl')
shap_path = os.path.abspath(shap_path)

 
def shap_feature_importance(X_train, y_train, shap_subset_size=100):
    ''' 
    Calculate SHAP feature importance for the given model and training data, and
    remove features with low importance. 
    '''    
    # Train a baseline model for SHAP value calculation
    baseline = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    baseline.fit(X_train, y_train)
    print("🛠️ SHAP baseline model trained.")
    
    # Use a subset of the training data for SHAP value calculation to speed up the process
    shap_idx = np.arange(max(0, len(X_train) - shap_subset_size), len(X_train))
    X_shap = X_train.iloc[shap_idx]
    print("Calculating SHAP values on subset of size:", X_shap.shape)

    explainer = shap.TreeExplainer(baseline) 

    raw_shap = explainer.shap_values(X_shap, check_additivity=False)

    # Handle both binary and multi-class cases, check to ensure correct class's SHAP values
    if isinstance(raw_shap, list):
        shap_values = raw_shap[1]
    else:
        if raw_shap.ndim == 3:
            shap_values = raw_shap[..., 1] 
        else:
            shap_values = raw_shap

    importance = np.mean(np.abs(shap_values), axis=0)  
    
    # Print top 10 important features
    print("-"*20)
    print("Top 10 important features:\n", "\n".join(list(X_train.columns[np.argsort(importance)[-10:]])))
    print("-"*20)

    # Set threshold for feature selection
    threshold = np.percentile(importance, 30)         
    selected = importance > threshold    

    if selected.sum() == 0: # Safety check
        print("Warning: No features selected based on SHAP importance. Keeping top 10 features.")            
        idx = np.argsort(importance)[-10:]
        selected = np.zeros_like(importance, dtype=bool)
        selected[idx] = True

    # Ensure always keep columns are not discarded
    always_keep_idx = [X_train.columns.get_loc(col) for col in always_keep_columns if col in X_train.columns]
    selected[always_keep_idx] = True

    X_train_reduced = X_train.loc[:, selected] # Keep only selected features

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

    corr_matrix = X.corr().abs() # Calculate absolute correlation matrix

    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)) # Get upper triangle of the correlation matrix

    # Identify features to drop based on correlation threshold, keep always_keep_columns
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    to_drop = [col for col in to_drop if col not in always_keep_columns]

    X_reduced = X.drop(columns=to_drop)

    print(f"⚙️  Removed {len(to_drop)} correlated features with threshold > {threshold}")

    return X_reduced, to_drop
