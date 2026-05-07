"""
This module contains helper methods for visualizing feature importance of the trained models.
"""

import os
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import permutation_importance
from script.helper_methods.config import get_config
from script.helper_methods.feature_reduction import compute_shap_values

# Config for plots
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

# Load config
cfg = get_config()
seed = cfg["experiment"]["random_state"]


def plot_feature_importance(model, X, y, horizon, 
    method = "auto", 
    top_n = 25,
    scoring = "f1",
    n_repeats = 5,
    random_state = seed,
    drop_cols = ("LogId",),
    figsize = (9, 7),
):
    """
    Calculate and plot feature importance using either model-based importance or permutation importance.
    Returns a DataFrame with the top features and their importance scores, 
    and saves a bar plot of feature importance to "plots/feature_importance/{horizon}.png".
    """

    # Drop columns that should not be used for feature importance calculation (e.g. LogId) and get feature names
    Xp = X.drop(columns=[c for c in drop_cols if c in X.columns]).copy()
    feature_names = Xp.columns.to_list()

    # Validate method parameter
    if method not in {"auto", "permutation", "model"}:
        raise ValueError("method must be one of: auto, permutation, model")

    use_model = False
    if method in {"auto", "model"}:
        if hasattr(model, "feature_importances_"):
            importances = np.asarray(model.feature_importances_, dtype=float)
            use_model = True
        elif hasattr(model, "coef_"):
            coef = np.asarray(model.coef_, dtype=float)
            importances = np.abs(coef).ravel()
            use_model = True

    if method == "permutation" or (method == "auto" and not use_model):
        pi = permutation_importance(
            model, Xp, y,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=random_state,
        )
        importances = pi.importances_mean
        title = f"Permutation importance (scoring={scoring})"
    else:
        title = "Model-based feature importance"

    # Build ranking table
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(top_n)

    # Plot
    plt.figure(figsize=figsize)
    plt.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()

    os.makedirs("plots/feature_importance", exist_ok=True)
    plt.savefig(f"plots/feature_importance/{horizon}.png", dpi=300)
    plt.close()

    return imp_df.reset_index(drop=True)


def plot_shap_summary(model, X, shap_subset_size=200, save_path="plots/shap_summary.png"):
    """
    Generate a SHAP summary plot for the given model and data.
    
    - `model`: Trained model
    - `X`: Feature matrix
    - `shap_subset_size`: Number of samples to use for SHAP calculation
    - `save_path`: Path to save the plot
    """
    
    # Compute SHAP values on a subset of the data for efficiency
    shap_values, X_shap = compute_shap_values(model, X, shap_subset_size=shap_subset_size)

    # Save the SHAP summary plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, X_shap, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 SHAP summary plot saved as {save_path}")
