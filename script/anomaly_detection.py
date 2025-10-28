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

