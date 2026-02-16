import yaml
import numpy as np
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

# Load config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

seed = cfg["experiment"]["random_state"]

def get_models_and_params(y):
    if len(np.unique(y)) == 1:
        print("⚠️  Only one class present.")
        return (
            {"Dummy": DummyClassifier()},
            {"Dummy": {
                "strategy": ["most_frequent", "stratified"]
            }}
        )
    else:
        models = {
            "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=seed),
            "XGBoost": XGBClassifier(eval_metric='logloss', scale_pos_weight= np.sum(y == 0) / np.sum(y == 1)),
        }

        hyperparameters = {
            "Random Forest": {
                'base_model__n_estimators': [100],
                'base_model__max_depth': [5, 10, 20],
                'base_model__min_samples_split': [20, 50, 100],
                'base_model__min_samples_leaf': [1, 2, 4],
                'base_model__max_leaf_nodes': [None, 100, 300, 500],
                'base_model__max_features': ['sqrt', 'log2', 0.2, 0.5],
            },
            "XGBoost": {
                'base_model__n_estimators': [2000],
                'base_model__min_child_weight': [3, 7, 10],
                'base_model__max_depth': [3, 4, 6],
                'base_model__learning_rate': [0.01, 0.05, 0.1],
                'base_model__colsample_bytree': [0.6, 0.75, 0.9],
                'base_model__base_score': [0.5],
                'base_model__reg_alpha': [0, 0.1, 0.5],
                'base_model__reg_lambda': [1, 1.5, 2.0]
            }
        }

    return models, hyperparameters
