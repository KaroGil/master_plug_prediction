"""
Helper methods for saving and loading models, as well as saving performance summaries.
"""

import joblib
import pandas as pd
from pathlib import Path


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
    ''' Save performance summary to CSV '''
    summary_df = pd.DataFrame(summary)
    summary_path = Path("models/model_comparison_summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved model comparison summary to {summary_path}")