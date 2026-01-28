import os
from imblearn import FunctionSampler
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report, f1_score, fbeta_score, make_scorer

from . import feature_engineering as fe
from .model_io import save_model, save_scores
from .models import get_models_and_params

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def per_logid_time_cv(groups, n_splits=5, gap=0):
    groups = np.array(groups)
    unique_logids = np.unique(groups)

    for run in unique_logids:
        idx = np.where(groups == run)[0]
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        for train_index, val_index in tscv.split(idx):
            yield idx[train_index], idx[val_index]


def drop_one_class_folds(cv, y):
    safe = []
    for tr, va in cv:
        if y.iloc[tr].nunique() < 2:   # train fold
            continue
        safe.append((tr, va))
    if not safe:
        raise ValueError("No valid folds left after dropping one-class folds.")
    return safe


def make_pipeline(model):
    '''Create a sklearn Pipeline from a list of (name, transformer/model) tuples'''
    
    sampler = FunctionSampler(
        func=fe.augment_minority_continuous_timeseries,
        kw_args=dict(n_augmentations=3, noise_frac=0.01, random_state=42)
    )

    pipe = ImbPipeline(steps=[
        ("augmenter", sampler),
        ("base_model", model)
    ])

    return pipe


def tune_random_search(model, X_train, y_train, params, n_iter=40):
    '''Perform Randomized Search with Time Series CV'''

    print(X_train["LogId"].nunique())

    tmp = pd.concat([X_train["LogId"], y_train.squeeze()], axis=1)
    tmp.columns = ["LogId","y"]
    print((tmp.groupby("LogId")["y"].max() == 1).sum())


    # Time-series 
    tscv = per_logid_time_cv(X_train['LogId'], n_splits=5) 

    tscv = drop_one_class_folds(tscv, y_train)
    
    # Scoring metrics
    scoring = {
        "F1-score": "f1_weighted",
        "F2-score": make_scorer(fbeta_score, beta=2, average='weighted')
    }

    X_train = X_train.drop(columns=['LogId'])  # drop LogId for modeling

    pipe = make_pipeline(model)

    # Randomized Search with TSCV
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=params,
        n_iter=n_iter,
        scoring=scoring,
        refit='F1-score',
        cv=tscv,
        n_jobs=-1,
        verbose=3,
        random_state=42,
        error_score='raise',
        return_train_score=True
    )

    search.fit(X_train, y_train)

    print("\nBest parameters:", search.best_params_)
    print("Best Training F1 score:", search.cv_results_['mean_train_F1-score'][search.best_index_])
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
        best_model, best_params, best_score = tune_random_search(
            model,
            X_train, y_train,
            hyperparameters[name],
            n_iter=5
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

    best_model_name = max(best_of_all_models, key=lambda k: best_of_all_models[k][2])
    print(f"\n🏆 Best overall model: {best_model_name} with validation F1 Score: {best_of_all_models[best_model_name][2]:.4f}")
    
    return best_of_all_models[best_model_name][0]


def evaluate_model_on_test(model, X_test, y_test):
    '''Evaluate the final model on the test dataset'''
    X_test = X_test.drop(columns=['LogId'])  # drop LogId for modeling
    y_test_pred = model.predict(X_test)
    print(y_test_pred)
    print("Test set results:")
    print(classification_report(y_test, y_test_pred, digits=3, zero_division=0))
    print("F1 Score:", f1_score(y_test, y_test_pred, average='weighted'))
    return y_test_pred


def model_data(X_train, y_train, X_test, y_test):
    '''Full modeling pipeline: find best model, retrain on train+val, evaluate on test'''

    best_model = find_best_model(X_train, y_train.squeeze()) # squeeze bc loading from csv adds extra dimension
    save_model(best_model)

    print("Best model name: " + type(best_model).__name__ + " with parameters: " + str(best_model.get_params()))
   
    y_test_pred = evaluate_model_on_test(best_model, X_test, y_test)
    return best_model, y_test_pred