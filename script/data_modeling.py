from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
from imblearn.under_sampling import RandomUnderSampler


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


def logistic_regression_model(X_train_scaled, y_train, X_val_scaled, y_val):
    # Train the model
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_train_scaled, y_train)

    # Evaluate the model
    val_pred = logreg.predict(X_val_scaled)
    print('Validation set results for logistic regression:')
    print(classification_report(y_val, val_pred, digits=3))
    return logreg

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