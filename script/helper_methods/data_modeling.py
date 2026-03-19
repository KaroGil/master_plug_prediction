import os
import numpy as np
import pandas as pd
from imblearn import FunctionSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score, fbeta_score, make_scorer

from . import feature_engineering as fe
from . import data_visualization as dv
from .model_io import save_model, save_scores
from .models import get_models_and_params
from script.helper_methods.config import get_config

# Load config
cfg = get_config()

seed = cfg["experiment"]["random_state"]
n_iter = cfg["experiment"]["n_iter"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def leave_one_run_out_cv(groups):
    '''Leaves out all samples from one group (run) for testing in each fold'''

    groups = np.asarray(groups)
    for g in np.unique(groups):
        test_idx  = np.where(groups == g)[0]
        train_idx = np.where(groups != g)[0]
        yield train_idx, test_idx


def tune_random_search(model, X_train, y_train, params, n_iter=40):
    '''Perform Randomized Search with Time Series CV'''

    cv = list(leave_one_run_out_cv(X_train["LogId"]))
    
    # Scoring metrics
    scoring = {
        "F1-score": "f1_weighted",
        "F2-score": make_scorer(fbeta_score, beta=2, average='weighted')
    }

    X_train = X_train.drop(columns=['LogId']) # drop LogId for modeling

    for tr, te in cv:
        print(
            "train pos:", (y_train.iloc[tr] == 1).sum(),
            "test pos:",  (y_train.iloc[te] == 1).sum()
        )

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

    cv_results = pd.DataFrame(search.cv_results_)
    print(cv_results.filter(regex="split"))

    return search.best_estimator_, search.best_params_, search.best_score_


def find_best_model(X_train, y_train):
    '''Compare multiple models with hyperparameter tuning and return the best one'''

    models, hyperparameters = get_models_and_params(y_train)

    best_of_all_models = {}

    for name, model in models.items():
        print(f"\n🔍 Tuning {name}...")
        if isinstance(model, DummyClassifier):
            print("Dummy model - baseline")
            best_model = model.fit(X_train.drop(columns=['LogId']), y_train)
            best_params = model.get_params()
            best_score = f1_score(y_train, best_model.predict(X_train.drop(columns=['LogId'])), average='weighted')
        else:   
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

    best_model_name = max((name for name in best_of_all_models if name != "Dummy"), key=lambda k: best_of_all_models[k][2])
    print(f"\n🏆 Best overall model: {best_model_name} with validation F1 Score: {best_of_all_models[best_model_name][2]:.4f}")
    
    return best_of_all_models[best_model_name][0]


def evaluate_model_on_test(model, X_test, y_test):
    '''Evaluate the final model on the test dataset'''
    
    X_test = X_test.drop(columns=['LogId'])  # drop LogId for modeling
    y_test_pred = model.predict(X_test)

    print("Test set results:")
    print(classification_report(y_test, y_test_pred, digits=3, zero_division=0))
    f1_score_value = f1_score(y_test, y_test_pred, average='weighted')
    print("F1 Score:", f1_score_value)
    return y_test_pred, f1_score_value


def model_data(X_train, y_train, X_test, y_test):
    '''Full modeling pipeline: find best model, retrain on train+val, evaluate on test'''

    best_model = find_best_model(X_train, y_train.squeeze()) # squeeze bc loading from csv adds extra dimension

    save_model(best_model)
    print("Best model name: " + type(best_model).__name__ + " with parameters: " + str(best_model.get_params()))

    imp = dv.plot_feature_importance(
        best_model,
        X_train,
        y_train,
        method="permutation",
        top_n=20
    )
    print(imp)
   
    _, f1_score_value = evaluate_model_on_test(best_model, X_test, y_test)
    return best_model, f1_score_value