import time
import tracemalloc
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from script.helper_methods.config import get_config
from script.helper_methods.data_loader import load_preprocessed_dataset

# Load config
cfg = get_config()
seed = cfg["experiment"]["random_state"]

N_RUNS = 5

def benchmark_model(model, X_train, y_train, X_test, name, n_runs=N_RUNS):
    train_times = []
    pred_times = []
    
    # Run multiple times to get average time and memory usage
    for _ in range(n_runs):
        # Time for training
        start = time.time()
        model.fit(X_train, y_train)
        train_times.append(time.time() - start)
        
        # Time for prediction
        start = time.time()
        model.predict(X_test)
        pred_times.append(time.time() - start)
    
    # Memory for training
    tracemalloc.start()
    model.fit(X_train, y_train)
    _, train_peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Memory for prediction
    tracemalloc.start()
    model.predict(X_test)
    _, pred_peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Return results as a dictionary
    return {
        "Model": name,
        "Training time (s)": round(np.mean(train_times), 4),
        "Prediction time (s)": round(np.mean(pred_times), 4),
        "Training memory (MB)": round(train_peak_memory / 1024 / 1024, 2),
        "Prediction memory (MB)": round(pred_peak_memory / 1024 / 1024, 2),
    }

# Load data
X_train, y_train, X_test, y_test = load_preprocessed_dataset()

# Define best hyperparameters defined in the model selection process
best_rf_params = {
    'class_weight': 'balanced',
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 20,
    'min_samples_leaf': 1,
    'max_leaf_nodes': None,
    'max_features': 'log2'
}

best_xgb_params = {
    'eval_metric': 'logloss',
    'scale_pos_weight': np.sum(y_train == 0) / np.sum(y_train == 1),
    'n_estimators': 2000,
    'min_child_weight': 7,
    'max_depth': 6,
    'learning_rate': 0.01,
    'colsample_bytree': 0.6,
    'base_score': 0.5,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0
}

# Define models to benchmark
models = {
    "Dummy": DummyClassifier(strategy="most_frequent", random_state=seed),
    "Random Forest": RandomForestClassifier(**best_rf_params, random_state=seed),
    "XGBoost": XGBClassifier(**best_xgb_params, random_state=seed)
}

# Benchmark models
results = []
for name, model in models.items():
    print(f"Benchmarking {name}...")
    r = benchmark_model(model, X_train, y_train, X_test, name)
    results.append(r)
    print(r)

# Display results
df_results = pd.DataFrame(results)
print("\n=== Benchmarking Results ===")
print(df_results.to_string(index=False))