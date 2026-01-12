import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.svm import OneClassSVM
from sklearn.dummy import DummyClassifier
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, fbeta_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# MODEL SAVING AND LOADING
def save_model(model, model_name="best_model", base_path="models/"):
    '''Save model / pipeline to disk'''
    path = Path(base_path)
    path.mkdir(parents=True, exist_ok=True)
    path = path / f"{model_name}.joblib"
    joblib.dump(model, path)
    print(f"Saved model to {path}")


def load_model(path):
    '''Load model / pipeline from disk'''
    return joblib.load(path)

def save_scores(summary):
    # Save summary to CSV
    summary_df = pd.DataFrame(summary)
    summary_path = Path("models/model_comparison_summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved model comparison summary to {summary_path}")

# HYPERPARAMETER TUNING AND MODEL COMPARISON
def check_anomaly_model(model):
    return isinstance(model, (OneClassSVM, IsolationForest))


class XGBEarlyStoppingWrapper(BaseEstimator):
    def __init__(self, base_model, early_stopping_rounds=30):
        self.base_model = base_model
        self.early_stopping_rounds = early_stopping_rounds

    def fit(self, X, y, **fit_params):
        """
        X, y already represent the TRAINING portion of the TSCV fold.
        But RandomizedSearchCV calls fit(X_train_fold, y_train_fold)
        without giving us the validation fold.

        HOWEVER: For time series, the best hack is:
            -> Use the LAST part of X,y as validation
            -> But ensure val window has MINIMUM POSITIVES
        """

        # ensure val window contains enough samples
        n = len(X)
        val_size = max(int(n * 0.3), 200)  # more stable

        X_train, X_val = X[:-val_size], X[-val_size:]
        y_train, y_val = y[:-val_size], y[-val_size:]

        model = clone(self.base_model)
        model.set_params(early_stopping_rounds=self.early_stopping_rounds)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        self.model_ = model
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def tune_random_search(model, X_train, y_train, params, n_iter=40):
    '''Perform Randomized Search with Time Series CV + early stopping (XGB only)'''

    # Handle anomaly model logic from your original code
    if check_anomaly_model(model):
        X_train = X_train[y_train == 0]
        y_train = y_train[y_train == 0]

    # Time-series 
    tscv = TimeSeriesSplit(n_splits=5, test_size=1000)

    # Print class distribution in each fold to check splits TODO: remove later
    for i, (_, te) in enumerate(tscv.split(X_train)):
        y_te = y_train[te]
        print(i, "test positives:", (y_te==1).sum(), "test size:", len(te))

    # Scoring metrics
    scoring = {
        "F1-score": "f1_weighted",
        "F2-score": make_scorer(fbeta_score, beta=2, average='weighted')
    }

    # Detect if model is XGBoost → wrap it
    if isinstance(model, XGBClassifier):
        print("⏳ Using early stopping for XGBoost…")
        model = XGBEarlyStoppingWrapper(model, early_stopping_rounds=50)

    else:
        print("🌲 RandomForest/Dummy (baseline) detected → No early stopping (handled normally).")

    # Class imbalance handling via sample weights
    pos = sum(y_train == 1)
    neg = sum(y_train == 0)
    ratio = neg / pos if pos != 0 else 1
    sample_weights = np.where(y_train == 1, ratio, 1)

    # Randomized Search with TSCV
    search = RandomizedSearchCV(
        estimator=model,
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

    search.fit(X_train, y_train, sample_weight=sample_weights)

    print("\nBest parameters:", search.best_params_)
    print("Best Training F1 score:", search.cv_results_['mean_train_F1-score'][search.best_index_])
    print("Best F1 score:", search.best_score_)

    cv_results = pd.DataFrame(search.cv_results_)
    print(cv_results.filter(regex="split"))

    return search.best_estimator_, search.best_params_, search.best_score_


def get_models_and_params(data):

    if len(np.unique(data)) == 1:
        print("⚠️  Only one class present.")
        return (
            {"Dummy": DummyClassifier(strategy="most_frequent")},
            {"Dummy": {
                "strategy": ["most_frequent", "stratified"]
            }}
        )
    else:
        models = {
            "Dummy (baseline)": DummyClassifier(strategy="most_frequent"),
            "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
            "XGBoost": XGBClassifier(eval_metric='logloss', scale_pos_weight=sum(data == 0) / sum(data == 1)),
        }

        hyperparameters = {
            "Dummy (baseline)": {
                "strategy": ["most_frequent", "stratified"]
            },
            "Random Forest": {
                'n_estimators': [100],
                'max_depth': [5, 10, 20],
                'min_samples_split': [20, 50, 100],
                'min_samples_leaf': [1, 2, 4],
                'max_leaf_nodes': [None, 100, 300, 500],
                'max_features': ['sqrt', 'log2', 0.2, 0.5],
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

    y_test_pred = model.predict(X_test)
    print(y_test_pred)
    print("Test set results:")
    print(classification_report(y_test, y_test_pred, digits=3, zero_division=0))
    print("F1 Score:", f1_score(y_test, y_test_pred, average='weighted'))
    return y_test_pred


def model_data(X_train, y_train, X_test, y_test):
    '''Full modeling pipeline: find best model, retrain on train+val, evaluate on test'''

    best_model = find_best_model(X_train, y_train)
    save_model(best_model)

    print("Best model name: " + type(best_model).__name__ + " with parameters: " + str(best_model.get_params()))
   
    y_test_pred = evaluate_model_on_test(best_model, X_test, y_test)
    return best_model, y_test_pred