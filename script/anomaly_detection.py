from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from typing import Optional
import numpy as np

def IForestClassifier(X: Optional[np.ndarray] = None,
                      contamination: float = 0.01,
                      random_state: int = 42,
                      **kwargs):
    ''' Create and train an Isolation Forest model. '''
    
    iForest = IsolationForest(contamination=contamination, random_state=random_state, **kwargs)
    if X is not None:
        iForest.fit(X)
    return iForest


def OCSVMClassifier(X: Optional[np.ndarray] = None,
                    nu: float = 0.5,
                    kernel: str = "rbf",
                    gamma: str = "scale",
                    **kwargs):
    """
    Create (and optionally fit) a OneClassSVM.
    nu approximates the fraction of outliers (0 < nu <= 1).

    Returns
    -------
    clf : OneClassSVM
    """
    clf = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma, **kwargs)
    if X is not None:
        clf.fit(X)
    return clf


def RobcovClassifier(X: Optional[np.ndarray] = None,
                     contamination: float = 0.01,
                     support_fraction: Optional[float] = None,
                     random_state: Optional[int] = None,
                     **kwargs):
    """
    Create (and optionally fit) an EllipticEnvelope (robust covariance) detector.
    Good for Gaussian-like anomaly detection.

    Returns
    -------
    clf : EllipticEnvelope
    """
    clf = EllipticEnvelope(contamination=contamination,
                           support_fraction=support_fraction,
                           random_state=random_state,
                           **kwargs)
    if X is not None:
        clf.fit(X)
    return clf

def train_anomaly_models(X: Optional[np.ndarray] = None):
    ''' Train and return a dictionary of anomaly detection models. '''

    models = {
        "IsolationForest": IForestClassifier(X),
        "OneClassSVM": OCSVMClassifier(X),
        "EllipticEnvelope": RobcovClassifier(X)
    }

    return models



def get_anomaly_scores(models, X):
    results = {}

    for name, model in models.items():
        try:
            score = model.decision_function(X)
        except AttributeError:
            score = model.score_samples(X)
        results[name] = score

    return results

