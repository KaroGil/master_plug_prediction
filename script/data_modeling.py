from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
import pandas as pd



def baseline_model(X_train, y_train, X_val, y_val):
    baseline = DummyClassifier(strategy="most_frequent")
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


def SMOTE_model(X_train, y_train, X_val, y_val):
    '''Train RandomForest model with SMOTE to handle class imbalance'''

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print('After SMOTE:', y_train_resampled.value_counts())
    print('After SMOTE (%):', y_train_resampled.value_counts(normalize=True))


    # Train model on resampled data
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_resampled, y_train_resampled)

    # Evaluate on validation and test sets
    y_val_pred = model.predict(X_val)
    print('Validation set results:')
    print(classification_report(y_val, y_val_pred, digits=3))
    return model, X_train_resampled, y_train_resampled


def undersample_model(X_train, y_train, X_val, y_val, sampling_strategy='auto', random_state=42):
    """
    Undersample majority class with RandomUnderSampler and train RandomForest.
    sampling_strategy: 'auto' or float / dict per imbalanced-learn docs.
    """
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

    print('After undersampling:', y_resampled.value_counts())
    print('After undersampling (%):', y_resampled.value_counts(normalize=True))

    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_resampled, y_resampled)

    y_val_pred = model.predict(X_val)
    print('Validation set results (undersample):')
    print(classification_report(y_val, y_val_pred, digits=3))

    return model


# TODO: is this used?
def logistic_regression_model(X_train_scaled, y_train, X_val_scaled, y_val):
    # Train the model
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_train_scaled, y_train)

    # Evaluate the model
    val_pred = logreg.predict(X_val_scaled)
    print('Validation set results for logistic regression:')
    print(classification_report(y_val, val_pred, digits=3))
    return logreg


# TODO: is this used?
def svm_model(X_train_scaled, y_train, X_val_scaled, y_val):
    '''Train and evaluate SVM model'''

    # Train the model
    svm = SVC(random_state=42)
    svm.fit(X_train_scaled, y_train)

    # Evaluate the model
    val_pred = svm.predict(X_val_scaled)
    print('Validation set results for SVM:')
    print(classification_report(y_val, val_pred, digits=3))
    return svm



def save_model(model, path):
    """Save model / pipeline to disk."""
    joblib.dump(model, path)
    print(f"Saved model to {path}")


def load_model(path):
    """Load model / pipeline from disk."""
    return joblib.load(path)


def time_series_hyperparameter_search(model, param_grid, X_train, y_train, X_val, y_val):
    keys = list(param_grid.keys())
    combinations = list(product(*param_grid.values()))
    
    best_score = -1
    best_params = None
    best_model = None

    print(f"Searching over {len(combinations)} hyperparameter combinations...")

    for combo in combinations:
        params = dict(zip(keys, combo))
        model.set_params(**params)

        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)

        print(f"Params: {params} | Validation Accuracy = {val_acc:.4f}")

        if val_acc > best_score:
            best_score = val_acc
            best_params = params
            best_model = model

    print("\n✅ Best parameters found:", best_params)
    print(f"Best validation accuracy: {best_score:.4f}")

    return best_model, best_params, best_score


def find_best_model(X_train, y_train, X_val, y_val):
    models = {
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
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
        }
    }

    best_of_all_models = {}

    for name, model in models.items():
        print(f"\n🔍 Tuning {name}...")
        best_model, best_params, best_score = time_series_hyperparameter_search(
            model,
            hyperparameters[name],
            X_train, y_train,
            X_val, y_val
        )
        best_of_all_models[name] = (best_model, best_params, best_score)

        print(f"Best {name} model with params: {best_params} achieved validation accuracy: {best_score:.4f}")

    # Create a summary table of best validation accuracies
    summary = pd.DataFrame([
        {"Model": name, "Best Validation Accuracy": score}
        for name, (_, _, score) in best_of_all_models.items()
    ])

    print("\nSummary of Best Validation Accuracies:")
    print(summary.to_string(index=False))

    best_model_name = max(best_of_all_models, key=lambda k: best_of_all_models[k][2])
    print(f"\n🏆 Best overall model: {best_model_name} with validation accuracy: {best_of_all_models[best_model_name][2]:.4f}")
    return best_of_all_models[best_model_name][0]