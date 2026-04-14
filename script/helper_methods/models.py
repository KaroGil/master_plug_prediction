import numpy as np
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from script.helper_methods.config import get_config

# Load config
cfg = get_config()
seed = cfg["experiment"]["random_state"]

def get_models_and_params(y):
    ''' Define models and hyperparameters for tuning '''
    
    if len(np.unique(y)) == 1: #Fallback
        print("⚠️  Only one class present.")
        return (
            {"Dummy": DummyClassifier()},
            {"Dummy": {
                "strategy": ["most_frequent", "stratified"]
            }}
        )
    else:
        models = {
            "Dummy": DummyClassifier(strategy="most_frequent", random_state=seed),
            "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=seed),
            "XGBoost": XGBClassifier(eval_metric='logloss', scale_pos_weight= np.sum(y == 0) / np.sum(y == 1))
        }

        hyperparameters = {
            "Random Forest": {
                'n_estimators': [100],
                'max_depth': [5, 10, 20],
                'min_samples_split': [20, 50, 100],
                'min_samples_leaf': [1, 2, 4],
                'max_leaf_nodes': [None, 100, 300, 500],
                'max_features': ['sqrt', 'log2', 0.2, 0.5],
            },
            "XGBoost": {
                'n_estimators': [2000],
                'min_child_weight': [3, 7, 10],
                'max_depth': [3, 4, 6],
                'learning_rate': [0.01, 0.05, 0.1],
                'colsample_bytree': [0.6, 0.75, 0.9],
                'base_score': [0.5],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [1, 1.5, 2.0]
            }
        }

    return models, hyperparameters
