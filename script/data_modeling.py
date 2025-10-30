from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import joblib
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from pathlib import Path
import script.data_visualization as dv
from sklearn.model_selection import learning_curve



# MODEL SAVING AND LOADING

def save_model(model, path_name):
    '''Save model / pipeline to disk'''
    path = Path(path_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Saved model to {path}")


def load_model(path):
    '''Load model / pipeline from disk'''

    return joblib.load(path)


# RESAMPLING TECHNIQUES

def SMOTE_model(X_train, y_train):
    '''Apply SMOTE to balance classes in training data'''

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print('After SMOTE:', y_train_resampled.value_counts())
    print('After SMOTE (%):', y_train_resampled.value_counts(normalize=True))

    return X_train_resampled, y_train_resampled


def undersample_model(X_train, y_train, sampling_strategy=0.5, random_state=42):
    """
    Apply Random Under-Sampling to balance classes in training data
    """
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

    print('After undersampling:', y_resampled.value_counts())
    print('After undersampling (%):', y_resampled.value_counts(normalize=True))

    return X_resampled, y_resampled


# BASELINE MODEL 

def baseline_model(X_train, y_train, X_val, y_val, method="most_frequent"):
    baseline = DummyClassifier(strategy=method)
    baseline.fit(X_train, y_train)

    y_val_pred = baseline.predict(X_val)
    print("Validation set results (baseline):")
    print(classification_report(y_val, y_val_pred, digits=3))

    return baseline


def train_and_evaluate_rf(X_train, X_val, y_train, y_val):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    val_score = model.score(X_val, y_val)
    print(f"Validation accuracy: {val_score:.3f}")
    return model

# HYPERPARAMETER TUNING AND MODEL COMPARISON

def time_series_hyperparameter_search(model, param_grid, X_train, y_train, X_val, y_val, verbose_level=1):
    '''Perform time-series aware hyperparameter search'''

    keys = list(param_grid.keys())
    combinations = list(product(*param_grid.values()))
    
    best_score = -1
    best_params = None
    best_model = None

    print(f"Searching over {len(combinations)} hyperparameter combinations..." if verbose_level > 0 else "")

    for combo in combinations:
        params = dict(zip(keys, combo))
        model.set_params(**params)

        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_acc = f1_score(y_val, val_pred)

        print(f"Params: {params} | Validation F1 Score = {val_acc:.4f}" if verbose_level > 1 else "")

        if val_acc > best_score:
            best_score = val_acc
            best_params = params
            best_model = model

    print("\n✅ Best parameters found:", best_params)
    print(f"Best validation F1 Score: {best_score:.4f}")

    return best_model, best_params, best_score


def get_models_and_params():
    models = {
        "Random Forest": RandomForestClassifier(class_weight='balanced'),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "SVM": SVC(probability=True, class_weight='balanced'),
        "XGBoost": XGBClassifier(eval_metric='logloss'),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "ANN": MLPClassifier(max_iter=500),
        "CNN": MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500)
    }

    hyperparameters = {
        "Random Forest": {
            'n_estimators': [100, 200],
            'max_depth': [10, 20]
        },
        "Logistic Regression": {
            'C': [1.0, 0.5],
            'solver': ['lbfgs', 'liblinear']
        },
        "SVM": {
            'C': [1.0, 0.5],
            'kernel': ['rbf', 'linear']
        },
        "XGBoost": {
            'n_estimators': [100, 200],
            'max_depth': [6, 8],
            'learning_rate': [0.1, 0.01]
        },
        "KNN": {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        },
        "Naive Bayes": {
            'var_smoothing': [1e-9, 1e-8, 1e-7]
        },
        "ANN": {
            'hidden_layer_sizes': [(100,), (100, 50)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd']
        },
        "CNN": {
            'hidden_layer_sizes': [(100, 50), (150, 75)],
            'activation': ['relu'],
            'solver': ['adam']
        }
    }

    return models, hyperparameters


def find_best_model(X_train, y_train, X_val, y_val, verbose_level=1, visualize=False):
    '''Compare multiple models with hyperparameter tuning and return the best one'''

    models, hyperparameters = get_models_and_params()

    best_of_all_models = {}

    for name, model in models.items():
        print(f"\n🔍 Tuning {name}...")
        best_model, best_params, best_score = time_series_hyperparameter_search(
            model,
            hyperparameters[name],
            X_train, y_train,
            X_val, y_val,
            verbose_level
        )
        best_of_all_models[name] = (best_model, best_params, best_score)

        print(f"Best {name} model with params: {best_params} achieved validation F1 Score: {best_score:.4f}")
        if visualize: #TODO: fix visualization for all models (e.g. get_y_scores may fail)
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