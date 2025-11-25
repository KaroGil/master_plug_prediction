import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from xgboost import XGBClassifier
from sklearn.svm import SVC, OneClassSVM
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from . import data_visualization as dv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# MODEL SAVING AND LOADING
def save_model(model, model_name, base_path="models/"):
    '''Save model / pipeline to disk'''
    path = Path(base_path)
    path.mkdir(parents=True, exist_ok=True)
    path = path / f"{model_name}.joblib"
    joblib.dump(model, path)
    print(f"Saved model to {path}")


def load_model(path):
    '''Load model / pipeline from disk'''

    return joblib.load(path)


# BASELINE MODEL
def baseline_model(X_train, y_train, X_val, y_val, method="most_frequent"):
    baseline = DummyClassifier(strategy=method)
    baseline.fit(X_train, y_train)

    y_val_pred = baseline.predict(X_val)

    val_score = f1_score(y_val, y_val_pred, average='weighted')
    print(f"Validation F1 Score: {val_score:.3f}")

    print("Validation set results (baseline):")
    print(classification_report(y_val, y_val_pred, digits=3))

    return baseline


# HYPERPARAMETER TUNING AND MODEL COMPARISON

def check_anomaly_model(model):
    '''Check if the model is an anomaly detection model'''

    return isinstance(model, OneClassSVM) or isinstance(model, IsolationForest)


def time_series_hyperparameter_search(model, param_grid, X_train, y_train, X_val, y_val, verbose_level=1):
    '''Perform time-series aware hyperparameter search'''

    keys = list(param_grid.keys())
    combinations = list(product(*param_grid.values()))
    
    best_score = -1
    best_params = None
    best_model = None

    print(f"Searching over {len(combinations)} hyperparameter combinations..." if verbose_level > 0 else "")

    if check_anomaly_model(model):
        X_train = X_train[y_train == 0]
        y_train = y_train[y_train == 0]

    for combo in combinations:
        params = dict(zip(keys, combo))
        model.set_params(**params)


        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        if check_anomaly_model(model):
            val_pred = np.where(val_pred == -1, 1, 0)  # Convert to anomaly labels
        val_acc = f1_score(y_val, val_pred, average='weighted')
        train_acc = f1_score(y_train, train_pred, average='weighted')

        print(f"Params: {params} | Validation F1 Score = {val_acc:.4f} | Training F1 Score = {train_acc:.4f}" if verbose_level > 1 else "")

        if val_acc > best_score:
            best_score = val_acc
            best_train_score = train_acc
            best_params = params
            best_model = model

    print("\n✅ Best parameters found:", best_params)
    print(f"Best validation F1 Score: {best_score:.4f} | Training F1 Score = {best_train_score:.4f}" if verbose_level > 0 else "")


    return best_model, best_params, best_score


def tune_random_search(model, X_train, y_train, params, n_iter=40):
    '''Perform Randomized Search with Time Series Cross-Validation'''
    if check_anomaly_model(model):
        X_train = X_train[y_train == 0]
        y_train = y_train[y_train == 0]

    tscv = TimeSeriesSplit(n_splits=5)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=n_iter,
        scoring='f1_weighted',
        cv=tscv,            
        n_jobs=-1,
        verbose=2,
        random_state=42,
        error_score='raise'
    )

    search.fit(X_train, y_train)

    print("\nBest parameters:", search.best_params_)
    print("Best F1 score:", search.best_score_)
    print("Best Training F1 score:", f1_score(y_train, search.predict(X_train), average='weighted'))

    return search.best_estimator_, search.best_params_, search.best_score_


def get_models_and_params(data):

    if len(np.unique(data)) == 1:
        print("⚠️  Only one class present.")
        return (
            {"Dummy": DummyClassifier(strategy="most_frequent"),
             "Isolation Forest": IsolationForest(n_estimators=100, random_state=42, n_jobs=-1)
             },
            {"Dummy": {
                "strategy": ["most_frequent", "stratified"]
            },
             "Isolation Forest": {
                    'n_estimators': [100, 200],
                    'max_samples': ['auto', 0.8],
                    'contamination': [0.1, 0.2]
             }}
        )
    else:
        models = {
            "Random Forest": RandomForestClassifier(class_weight='balanced'),
            "SVM": SVC(probability=True, class_weight='balanced'),
            "XGBoost": XGBClassifier(eval_metric='logloss'),
        }

        hyperparameters = {
            "Random Forest": {
                'n_estimators': [200, 400, 800],
                'max_depth': [10, 20, 40, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.3, 0.5]
            },
            
            "SVM": {
                'C': [1.0, 0.5],
                'kernel': ['linear'] #commented out ['rbf'] for faster convergence
            },
            "XGBoost": {
                'n_estimators': [200, 500],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'gamma': [0, 1, 5],
                'base_score': [0.5]
            },
        }

    return models, hyperparameters


def find_best_model(X_train, y_train, X_val, y_val, verbose_level=1, visualize=False):
    '''Compare multiple models with hyperparameter tuning and return the best one'''

    models, hyperparameters = get_models_and_params(y_train)

    best_of_all_models = {}

    for name, model in models.items():
        print(f"\n🔍 Tuning {name}...")
        if name == "SVM":
            best_model, best_params, best_score = time_series_hyperparameter_search(
            model,
            hyperparameters[name],
            X_train, y_train,
            X_val, y_val,
            verbose_level
        )
        else:
            best_model, best_params, best_score = tune_random_search(
            model,
            pd.concat([X_train, X_val], axis=0), pd.concat([y_train, y_val], axis=0),
            hyperparameters[name],
            n_iter=20
        )
        best_of_all_models[name] = (best_model, best_params, best_score)

        print(f"Best {name} model with params: {best_params} achieved validation F1 Score: {best_score:.4f}")
        if visualize:
            print(f"Visualizing {name} performance on validation set:")
            y_val_pred = best_model.predict(X_val)
            dv.plot_confusion_matrix(y_val, y_val_pred)

            train_sizes, train_scores, val_scores = get_learning_curve_data(best_model, X_train, y_train)
            dv.learning_curve(train_sizes, train_scores, val_scores)

            y_pred_proba = dv.get_y_scores(best_model, X_val)
            dv.plot_calibration_curve(y_val, y_pred_proba)
            dv.plot_precision_recall_curve(y_val, y_pred_proba)    

    summary = [
        {"Model": name, "Best Validation F1 Score": score}
        for name, (_, _, score) in best_of_all_models.items()
    ]

    print("\nSummary of Best Validation F1 Scores:")
    for item in summary:
        print(f"{item['Model']}: {item['Best Validation F1 Score']:.4f}")

    best_model_name = max(best_of_all_models, key=lambda k: best_of_all_models[k][2])
    print(f"\n🏆 Best overall model: {best_model_name} with validation F1 Score: {best_of_all_models[best_model_name][2]:.4f}")
    return best_of_all_models[best_model_name][0]


# TEST SET EVALUATION
def retrain_final_model(X_train, X_val, y_train, y_val, best_model):
    '''Retrain the best model on the train + val'''

    X_combined = pd.concat([X_train, X_val])
    y_combined = pd.concat([y_train, y_val])

    best_model.fit(X_combined, y_combined)
    return best_model


def evaluate_model_on_test(model, X_test, y_test):
    '''Evaluate the final model on the test dataset'''

    y_test_pred = model.predict(X_test)
    if check_anomaly_model(model):
        y_test_pred = np.where(y_test_pred == -1, 1, 0)  # Convert to anomaly labels
    print(y_test_pred)
    print("Test set results:")
    print(classification_report(y_test, y_test_pred, digits=3))
    print("F1 Score:", f1_score(y_test, y_test_pred, average='weighted'))
    return y_test_pred


# LEARNING CURVE DATA
def get_learning_curve_data(model, X_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)):
    '''Get data for learning curve plotting'''

    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=cv, train_sizes=train_sizes, scoring='f1_weighted', n_jobs=-1
    )

    return train_sizes, train_scores, val_scores


def model_data(X_train, y_train, X_val, y_val, X_test, y_test):
    '''Full modeling pipeline: find best model, retrain on train+val, evaluate on test'''

    best_model = find_best_model(X_train, y_train, X_val, y_val, verbose_level=2)
    save_model(best_model, model_name=type(best_model).__name__)
    final_model = retrain_final_model(X_train, X_val, y_train, y_val, best_model)
    y_test_pred = evaluate_model_on_test(final_model, X_test, y_test)
    return final_model, y_test_pred