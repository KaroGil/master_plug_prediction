from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from typing import Optional
import numpy as np

def IForestClassifier(X: Optional[np.ndarray],
                      contamination: float = 0.01,
                      random_state: int = 42,
                      **kwargs):
    ''' Create and train an Isolation Forest model. '''
    
    iForest = IsolationForest(contamination=contamination, random_state=random_state, **kwargs)
    
    iForest.fit(X)

    print(f"Isolation Forest trained with contamination={contamination}")

    return iForest


def OCSVMClassifier(X: Optional[np.ndarray],
                    nu: float = 0.5,
                    kernel: str = "rbf",
                    gamma: str = "scale",
                    **kwargs):
    """
    Create (and optionally fit) a OneClassSVM.
    nu approximates the fraction of outliers (0 < nu <= 1).
    """

    clf = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma, **kwargs)

    print("Training One-Class SVM...")
    
    clf.fit(X)

    print(f"One-Class SVM trained with nu={nu}, kernel={kernel}, gamma={gamma}")

    return clf



def train_anomaly_models(X: Optional[np.ndarray], y: Optional[np.ndarray]):
    ''' Train and return a dictionary of anomaly detection models. '''

    normal_X = X[y==0]

    models = {
        "IsolationForest": IForestClassifier(normal_X),
        #"OneClassSVM": OCSVMClassifier(normal_X),
        }

    print(f"Trained {len(models)} anomaly detection models.")

    return models


def predict_anomalies(models, X):
    ''' Predict anomalies using the provided models. '''

    results = {}

    for name, model in models.items():
        print(f"Predicting anomalies with {name}...")

        preds = model.predict(X)
        binary_preds = np.where(preds == -1, 1, 0)
        results[name] = binary_preds

        print(f"Anomalies predicted with {name}.")
        print(f"Anomaly counts:\n{np.unique(binary_preds, return_counts=True)}")

    print("Anomaly prediction completed for all models.")

    return results


def get_anomaly_scores(models, X):
    results = {}

    for name, model in models.items():
        try:
            score = model.decision_function(X)
        except AttributeError:
            score = model.score_samples(X)
        results[name] = score

    return results


def hyperparameter_search(model, param_grid, X_train):
    ts_cv = TimeSeriesSplit(
        n_splits=5, 
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=20,
        cv=ts_cv,
        scoring='f1_weighted',
        n_jobs=-1
    )

    print(f"Starting hyperparameter search for {model.__class__.__name__}...")
    search.fit(X_train)
    print("Hyperparameter search completed.")
    print("Best parameters found:", search.best_params_)
    print(f"Best cross-validation F1 Score: {search.best_score_:.4f}")

    return search.best_estimator_, search.best_params_, search.best_score_
