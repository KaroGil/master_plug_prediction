"""
This module contains the main data modeling pipeline, including hyperparameter tuning with randomized search and time series cross-validation. 
"""

import os
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, fbeta_score, make_scorer

from . import data_visualization as dv
from .model_io import save_model, save_scores
from .models import get_models_and_params
from .model_evaluation import evaluate_model_on_test, evaluate_all_models
from script.helper_methods.config import get_config

# Load config
cfg = get_config()
seed = cfg["experiment"]["random_state"]
n_iter = cfg["experiment"]["n_iter"]
HORIZON = cfg["experiment"]["horizon"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def leave_one_run_out_cv(groups):
    """
    Custom cross-validation generator that implements a leave-one-run-out strategy for time series data. 
    Each unique group (run) is left out as the test set while the remaining groups are used for training in each fold.
    """

    groups = np.asarray(groups) # Ensure groups is a numpy array for indexing
    
    # Loop through each unique group and yield train/test indices for that group
    for g in np.unique(groups):
        test_idx  = np.where(groups == g)[0]
        train_idx = np.where(groups != g)[0]
        yield train_idx, test_idx


def tune_random_search(model, X_train, y_train, params, n_iter=40):
    """Perform Randomized Search with definde CV function and return best model, params and score"""

    cv = list(leave_one_run_out_cv(X_train["LogId"])) # Splitting into folds
    
    # Scoring metrics
    scoring = {
        "F1-score": "f1_weighted",
        "F2-score": make_scorer(fbeta_score, beta=2, average='weighted')
    }

    X_train = X_train.drop(columns=['LogId']) # drop LogId for modeling

    # Print class distribution in each fold
    for tr, te in cv:
        print(
            "train pos:", (y_train.iloc[tr] == 1).sum(),
            "test pos:",  (y_train.iloc[te] == 1).sum()
        )

    # Run Randomized Search
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=n_iter,
        scoring=scoring,
        refit='F1-score',
        cv=cv,
        n_jobs=-1,
        verbose=3,
        random_state=seed,
        error_score='raise',
        return_train_score=True,
    )

    search.fit(X_train, y_train)

    print("\nBest parameters:", search.best_params_)
    idx = search.best_index_

    print("Train F1:", search.cv_results_['mean_train_F1-score'][idx])
    print("CV F1:", search.cv_results_['mean_test_F1-score'][idx])

    print("Best F1 score:", search.best_score_)

    # Print all CV results for F1-score
    cv_results = pd.DataFrame(search.cv_results_)
    print(cv_results.filter(regex="split"))

    return search.best_estimator_, search.best_params_, search.best_score_


def find_best_model(X_train, y_train):
    """
    Compare multiple models with hyperparameter tuning and return the best one.
    Using Randomized Search with CV (leave-one-run-out) to find the best model and hyperparameters based on validation F1 score.
    Also includes a DummyClassifier as a baseline for comparison.
    Prints the best model, its parameters, and validation F1 score, as well as saves a summary of all models and their best validation scores. 
    Saves the best model and the baseline model for later
    """

    models, hyperparameters = get_models_and_params(y_train) # Get models and their hyperparameter search spaces

    best_of_all_models = {}

    # Loop through each model, perform hyperparameter tuning, and store the best model, its parameters, and validation score
    for name, model in models.items():
        print(f"\n🔍 Tuning {name}...")
        if isinstance(model, DummyClassifier): # No tuning for dummy model, just fit and evaluate as baseline
            print("Dummy model - baseline")
            best_model = model.fit(X_train.drop(columns=['LogId']), y_train)
            best_params = model.get_params()
            best_score = f1_score(y_train, best_model.predict(X_train.drop(columns=['LogId'])), average='weighted')
        else:   
            # Perform hyperparameter tuning for other models
            best_model, best_params, best_score = tune_random_search(
                model,
                X_train, y_train,
                hyperparameters[name],
                n_iter=n_iter
            )

        best_of_all_models[name] = (best_model, best_params, best_score)

        print(f"Best {name} model with params: {best_params} achieved validation F1 Score: {best_score:.4f}")
    
    summary = [
        {"Model": name, "Best Validation F1 Score": score}
        for name, (_, _, score) in best_of_all_models.items()
    ]

    # Save summary to CSV for later analysis
    save_scores(summary)

    print("\nSummary of Best Validation F1 Scores:")
    for item in summary:
        print(f"{item['Model']}: {item['Best Validation F1 Score']:.4f}")

    # Identify the best overall model (excluding Dummy) based on validation F1 score
    best_model_name = max((name for name in best_of_all_models if name != "Dummy"), key=lambda k: best_of_all_models[k][2])
    print(f"\n🏆 Best overall model: {best_model_name} with validation F1 Score: {best_of_all_models[best_model_name][2]:.4f}")
    
    return best_of_all_models, best_model_name



def model_data(X_train, y_train, X_test, y_test, horizon=HORIZON):
    """Full modeling pipeline: find best model, calculate feature importances, evaluate on test"""

    best_of_all_models, best_model_name = find_best_model(X_train, y_train.squeeze()) # squeeze bc loading from csv adds extra dimension
    best_model = best_of_all_models[best_model_name][0]

    # Save best model as well as all models for later use and comparison
    save_model(best_model)
    save_model(best_of_all_models["Dummy"][0], model_name="dummy")
    save_model(best_of_all_models["Random Forest"][0], model_name="rf")
    save_model(best_of_all_models["XGBoost"][0], model_name="xgboost")
    print("Best model name: " + type(best_model).__name__ + " with parameters: " + str(best_model.get_params()))
    
    # Calculate and print feature importances for the best model using permutation importance
    imp = dv.plot_feature_importance(
        best_model,
        X_train,
        y_train,
        method="permutation",
        top_n=20,
        horizon=horizon
    )
    print(imp)

    # Generate SHAP summary plot for the best model
    dv.plot_shap_summary(best_model, X_train.drop(columns=['LogId']), save_path=f"plots/shap_summary_{horizon}.png")
    
    X_test.drop(columns=['LogId'], inplace=True) # drop LogId for evaluation

    # Evaluate the best model on the test set and print the F1 score.
    _, f1_score_value = evaluate_model_on_test(best_model, X_test, y_test)

    if horizon == HORIZON: # Only evaluate all models on test set for the default horizon
        evaluate_all_models(best_of_all_models, X_test, y_test, horizon)

    return best_model, f1_score_value