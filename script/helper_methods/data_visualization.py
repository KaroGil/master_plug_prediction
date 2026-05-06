"""
Plotting and visualization helpers for model analysis and result summaries.

Includes utilities for:
- signal inspection and plug-event highlighting
- prediction vs. ground-truth visualization
- feature importance plots
- dataset-level evaluation summaries
- horizon and F1 score comparisons
"""
import os
import math
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance
from script.helper_methods.config import get_config
#TODO: check this file.
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
dataset_nr = cfg['data']['datasets']
target_col = cfg["data"]["target"]
test_sets = cfg["data"]["test_sets"]


def visualize_key_column(data, key="Flow rate (Mean)", name=None):
    """
    Visualize flow rate and pump outlet pressure over time
    Used when labeling data. 
    """

    if key not in data.columns:
        raise ValueError(f"Column '{key}' not found in dataset.")

    plt.figure(figsize=(12,6))
    plt.plot(data["Elapsed_seconds"] if "Elapsed_seconds" in data.columns else data.index, data[key], label=key)

    plt.xlabel("Elapsed_seconds")
    plt.ylabel("Value")
    plt.title(f"Flow rate & Pump outlet pressure over time for {name}" if name else "Flow rate & Pump outlet pressure over time")
    if name == "Labled Dataset":
        import matplotlib.dates as mdates
        plt.xlabel("Timestamp")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.legend()
    plt.show()


def visualize_plug_event(data, plug_column=target_col, key="Flow rate (Mean)", name=None, extra_id=0, ax=None):
    """
    Visualize a key signal (flow rate or pressure) with plug events highlighted, and optionally extra vertical lines for specific times based on extra_id.
    Used when labeling data.
    - `data`: DataFrame containing the data to plot
    - `plug_column`: Column name indicating plug events (default is target_col)
    - `key`: Column name of the signal to plot (default is "Flow rate (Mean)")
    - `name`: Optional name for the plot title
    - `extra_id`: If set to 0, nothing extra is plotted. Other numbers might correspond to specific extra lines to plot (e.g. for dataset 7)
    - `ax`: Optional matplotlib axis to plot on. If None, a new figure and axis will be created   
    """
    possible_keys = [key, f"{key}_mean"]
    actual_key = next((col for col in possible_keys if col in data.columns), None)

    if actual_key is None:
        raise ValueError(f"None of these columns were found: {possible_keys}")

    created_figure = False
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))
        created_figure = True

    x = data["Elapsed_seconds"] if "Elapsed_seconds" in data.columns else data.index

    # Plot signal
    ax.plot(x, data[actual_key], label=actual_key, alpha=0.5)

    # Plug events
    plug_events = data[data[plug_column] == 1]

    if not plug_events.empty:
        x_plug = plug_events["Elapsed_seconds"] if "Elapsed_seconds" in plug_events.columns else plug_events.index

        ax.scatter(
            x_plug,
            plug_events[actual_key],
            color="red",
            label=f"{plug_column}=1",
            zorder=5,
            s=15
        )

    if extra_id == 3:
        # Indicate end of startup noise, first stabilize at 400kg/h
        ax.axvline(pd.to_datetime("2021-09-01 10:59:21.171000"), color="black", linestyle="--", label="Time = 15:37:26")

        #Spike
        ax.axvline(pd.to_datetime("2021-09-01 11:28:25.069000"), color="black", linestyle="--", label="Time = 15:37:26")
        ax.axvline(pd.to_datetime("2021-09-01 11:36:49.038000"), color="black", linestyle="--", label="Time = 15:37:26")

        # End of experiment spike
        ax.axvline(pd.to_datetime("2021-09-01 12:41:33.813000"), color="black", linestyle="--", label="Time = 15:37:26")

        #1 – 3.9% (-003 min), 2 – 5.7% (006 min), 3 – 4% (012 min), 4 – 3.8% (018 min), 5 – 2.5% (025 min), 6 – (033 min)
        base = data.index[0]

        ax.axvline(base + pd.Timedelta(minutes=-3), linestyle="--", color="red", label="-003 min")
        ax.axvline(base + pd.Timedelta(minutes=6), linestyle="--", color="orange", label="006 min")
        ax.axvline(base + pd.Timedelta(minutes=12), linestyle="--", color="yellow", label="012 min")
        ax.axvline(base + pd.Timedelta(minutes=18), linestyle="--", color="green", label="018 min")
        ax.axvline(base + pd.Timedelta(minutes=25), linestyle="--", color="blue", label="025 min")
        ax.axvline(base + pd.Timedelta(minutes=33), linestyle="--", color="purple", label="033 min")

    if extra_id == 4:
        ax.axvline(pd.to_datetime("1900-01-01 15:37:26"), color="purple", linestyle="--", label="Time = 15:37:26")
        ax.axvline(pd.to_datetime("1900-01-01 16:17:41"), color="green", linestyle="--", label="Time = 16:17:41")
        ax.axvline(pd.to_datetime("1900-01-01 16:20:34"), color="pink", linestyle="--", label="Time = 16:20:34")
        ax.axvline(pd.to_datetime("1900-01-01 16:43:20"), color="yellow", linestyle="--", label="Time = 16:43:20")

    if extra_id == 7:
        ax.axvline(pd.to_datetime("1900-01-01 11:22"), color="purple", linestyle="--", label="Time = 11:22")
        ax.axvline(pd.to_datetime("1900-01-01 11:52"), color="green", linestyle="--", label="Time = 11:52")
        ax.axvline(pd.to_datetime("1900-01-01 12:52"), color="pink", linestyle="--", label="Time = 12:52")
        ax.axvline(pd.to_datetime("1900-01-01 14:58"), color="yellow", linestyle="--", label="Time = 14:58")
        ax.axvline(pd.to_datetime("1900-01-01 17:00"), color="gray", linestyle="--", label="Time = 17:00")

    if extra_id == 14:
        # Video starting times
        ax.axvline(pd.to_datetime("1900-01-01 11:28"), color="black", linestyle="--", label="Time = 11:28, video start")
        ax.axvline(pd.to_datetime("1900-01-01 16:26"), color="black", linestyle="--", label="Time = 16:26, video start")
        
        # Reaches target flowrate 1400 kg/h
        ax.axvline(pd.to_datetime("1900-01-01 11:46:21.233000"), color="yellow", linestyle="--", label="Time = 11:46, reaches target flowrate 1400 kg/h")

        # Plummet point, flowrate drops to 0
        ax.axvline(pd.to_datetime("1900-01-01 17:01:44.233000"), color="red", linestyle="--", label="Time = 17:01")

    if extra_id == 17:
        ax.axvline(pd.to_datetime("2021-12-07 10:43:03.843000"), color="yellow", linestyle="--", label="Time = 10:43, reaches target flowrate 1500 kg/h")
        ax.axvline(pd.to_datetime("2021-12-07 11:06:58.754000"), color="yellow", linestyle="--", label="Time = 11:06, reaches target flowrate 1500 kg/h")
        ax.axvline(pd.to_datetime("2021-12-07 12:22:37.470000"), color="red", linestyle="--", label="Time = 12:24, plummet point")

    if extra_id == 18:
        # Video starting times
        ax.axvline(pd.to_datetime("2022-11-03 14:06"), color="black", linestyle="--", label="Time = 14:06, video start")
        
        # Reaches target flowrate 400 kg/h
        ax.axvline(pd.to_datetime("2022-11-03 14:20:05"), color="yellow", linestyle="--", label="Time = 11:46, reaches target flowrate 400 kg/h")

        # End spike 
        ax.axvline(pd.to_datetime("2022-11-03 17:26:46.000"), color="red", linestyle="--", label="Time = 17:26, end spike")
        
        #14:22 – 3.9% 14:54 – 4.4%; 15:54 – 4.3% 17:55 – 3.0% 19:58 – 3.0% 20:38 – 3.0% TODO: remove this or fix frequencing in datast 18
        # ax.axvline(pd.to_datetime("2022-11-03 14:22"), color="purple", linestyle="--", label="Time = 14:22")
        # ax.axvline(pd.to_datetime("2022-11-03 14:54"), color="green", linestyle="--", label="Time = 14:54")
        # ax.axvline(pd.to_datetime("2022-11-03 15:54"), color="pink", linestyle="--", label="Time = 15:54")
        # ax.axvline(pd.to_datetime("2022-11-03 17:55"), color="yellow", linestyle="--", label="Time = 17:55")
        # ax.axvline(pd.to_datetime("2022-11-03 19:58"), color="gray", linestyle="--", label="Time = 19:58")
        # ax.axvline(pd.to_datetime("2022-11-03 20:38"), color="gray", linestyle="--", label="Time = 20:38")

    if extra_id == 24:
        # Video starts
        ax.axvline(pd.to_datetime("1900-01-01 13:43"), color="black", linestyle="--", label="Time = 13:43, video start")
        ax.axvline(pd.to_datetime("1900-01-01 15:17"), color="black", linestyle="--", label="Time = 15:17, video start")
        ax.axvline(pd.to_datetime("1900-01-01 16:28"), color="black", linestyle="--", label="Time = 16:28, video start")
        

        # Reached target flow rate 900 kg/h
        ax.axvline(pd.to_datetime("1900-01-01 13:57:42.474000"), color="yellow", linestyle="--", label="Time = 14:44, reaches target flowrate 900 kg/h")

        # Plummet
        ax.axvline(pd.to_datetime("1900-01-01 16:29:08.474"), color="red", linestyle="--", label="Time = 16:29:08, plummet start")
        ax.axvline(pd.to_datetime("1900-01-01 16:35:50.474"), color="red", linestyle="--", label="Time = 16:29:11, plummet end")
        
        # 13:56 – 13.4% 18:00 – 13.1%
        ax.axvline(pd.to_datetime("1900-01-01 13:56"), color="pink", linestyle="--", label="Time = 13:56")
        ax.axvline(pd.to_datetime("1900-01-01 18:00"), color="pink", linestyle="--", label="Time = 18:00")

    ax.set_title(f"{name}" if name else actual_key)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(fontsize=6)

    if created_figure:
        plt.tight_layout()
        plt.show()


def visualize_predicted_vs_true(df, y_pred, model_name=None, plotLabel=True):
    """Visualize predicted vs true plug events on top of flow rate and pressure signals, with optional vertical lines for specific times based on dataset ID. Used for prediction visualization."""
    plt.figure(figsize=(12,6))

    pressure_col = "Pump outlet pressure (Mean)_std"

    if pressure_col not in df.columns:
        pressure_col = "Pump outlet pressure (Mean)_mean"

        if pressure_col not in df.columns:
            pressure_candidates = df.columns[df.columns.str.contains("press", case=False, na=False)]
            if len(pressure_candidates) > 0:
                pressure_col = pressure_candidates[0]
            else:
                pressure_col = None

    plt.plot(df.index, df[pressure_col], label={pressure_col}, alpha=0.5, color="darkorange") 

    # Highlight true target events
    if plotLabel:
        plug_events = df[df[target_col] == 1]
        plt.scatter(plug_events.index, plug_events[pressure_col], color="blue", label="Plug=1 (Pressure)", zorder=5, marker='x')
        
    # Highlight predicted Plug events
    plug_events = df[y_pred == 1]
    plt.scatter(plug_events.index, plug_events[pressure_col], color="green", label="Predicted plug (Pressure)", zorder=7, marker='.') 
    plt.xlabel("Elapsed_seconds")

    plt.ylabel("Pressure [Pa]")

    # model_name_str = f" by {model_name}" if model_name else ""
    # plt.title(f"Predicted vs True Plug=1 Events{model_name_str}")
    plt.legend()
    plt.show()


## FEATURE IMPORTANCE PLOTS ##

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
    #TODO: make this into a helper method that can be used for both shap instances (removal and feature importances here)
    # Use only a subset of the data for SHAP calculation to speed up the process
    shap_idx = np.arange(max(0, len(X) - shap_subset_size), len(X))
    X_shap = X.iloc[shap_idx].reset_index(drop=True)
    print(f"Calculating SHAP values on subset of size: {X_shap.shape}")

    # Check for feature name mismatches between model and SHAP input, for debugging
    model_features = set(model.get_booster().feature_names)
    xshap_features = set(X_shap.columns.tolist())
    extra_in_xshap = xshap_features - model_features
    missing_in_xshap = model_features - xshap_features
    print(f"Extra in X_shap (not in model): {extra_in_xshap}")
    print(f"Missing in X_shap (not in X_shap): {missing_in_xshap}")

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(X_shap, check_additivity=False)

    # Handle different output formats of SHAP values for binary classification
    if isinstance(raw_shap, list):
        shap_values = raw_shap[1]
    else:
        if raw_shap.ndim == 3:
            shap_values = raw_shap[..., 1]
        else:
            shap_values = raw_shap
    #TODO: end here this is the block

    # Save the SHAP summary plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, X_shap, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 SHAP summary plot saved as {save_path}")


### VISUALIZING RESULTS FOR PREDICT_ALL ###
def plot_one(ax, df, y_pred, figureNum, y, dataset_id = None, pressure_col="Pump outlet pressure (Mean)_std"):
    """
    Plot flow rate and pressure with true and predicted plug events for one dataset on a given axis.
    Used as a helper function for plot_all_predictions to create subplots for each dataset.
    """
    if pressure_col not in df.columns:
        pressure_candidates = df.columns[df.columns.str.contains("press", case=False, na=False)]
        print(f"Dataset {dataset_id}: Pressure column not found, candidates: {pressure_candidates}")
    
        if len(pressure_candidates) == 0:
            pressure_col = None
        elif "TS outlet pressure (Mean)_max" in pressure_candidates:
            pressure_col = "TS outlet pressure (Mean)_max"
            #pressure_col = "Pump outlet pressure (Mean)_std"
        else:
            pressure_col = pressure_candidates[0]

    # Determine if this dataset is part of training or test set for labeling the plot
    shown_id = dataset_id if dataset_id is not None else figureNum
    dataset_nrs = list(dataset_nr)
    test_set_nr = max(dataset_nrs)
    test_set_nrs = test_sets if test_sets else [test_set_nr]
    train_set_nrs = [nr for nr in dataset_nrs if nr not in test_set_nrs]
    
    if pressure_col in df.columns: 
        ax.plot(df.index, df[pressure_col], label={pressure_col}, alpha=0.5, color="steelblue")

    # Highlight true Plug events, on training data
    true_plug_events = df[y == 1]
    if shown_id not in test_set_nrs:
        if pressure_col and pressure_col in df.columns:
            ax.scatter(true_plug_events.index, true_plug_events[pressure_col], color="red", label="True Plug=1 (Pressure)", zorder=6, marker='x', s=50, linewidth=2)
    
    # Highlight predicted Plug events, on all data
    plug_events = df[y_pred == 1]
    if pressure_col and pressure_col in df.columns:
        ax.scatter(plug_events.index, plug_events[pressure_col], color="yellow", label="Predicted plug (Pressure)", zorder=7, marker='.', s=50)  
     
    ax.set_xlabel("Elapsed_seconds")
    ax.set_ylabel("Pressure [Pa]")

    if shown_id in test_set_nrs:
        ax.set_title(f"Data nr {shown_id} [test set]")
    else:
        ax.set_title(f"Data nr {shown_id}" if shown_id not in train_set_nrs else f"Data nr {shown_id} [training set]")


def plot_all_predictions(X_y_list, y_preds, dataset_ids,  runId):
    """
    Plots predicted vs true events for all given datasets in one figure, with subplots for each dataset.
    
    - `X_y_list`: List of tuples (X, y) for each dataset, where X is the feature DataFrame, y is the true target array.
    - `y_preds`: List of predicted target arrays corresponding to each dataset in X_y_list.
    - `dataset_ids`: List of dataset IDs corresponding to each dataset in X_y_list, used for labeling the subplots.
    - `runId`: Unique identifier for the run, used for saving the figure with a specific name.
    """
    n_plots = len(X_y_list)  
    ncols = 3 if n_plots >= 3 else 2                
    nrows = math.ceil(n_plots / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3.2*nrows))
    axes = axes.flatten()

    # Hide any unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    for subplot_idx, ((X, y), y_pred, dataset_id) in enumerate(zip(X_y_list, y_preds, dataset_ids), start=1):
        plot_one(axes[subplot_idx-1], X, y_pred, subplot_idx, y, dataset_id=dataset_id)
   
    # Shared legend: merge unique handles from all subplots
    seen = set()
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                handles.append(h)
                labels.append(l)

    if "test" in runId:
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.100, 0.100), ncol=1, fontsize=12)
        plt.subplots_adjust(hspace=0.7, top=0.90)
    else:
        fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.95, 0.10), ncol=1, fontsize=12)
        plt.subplots_adjust(hspace=0.5, top=0.90)

    # Save the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"plots/{runId}.png", dpi=300, bbox_inches='tight')
    print(f"Plot saved as plots/{runId}.png")
    plt.close()


def plot_test_data_summary(X_y_list, y_preds, dataset_ids, runId):
    n_plots = len(X_y_list)
    ncols = n_plots
    nrows = 2  

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3.2 * nrows))

    if n_plots == 1:
        axes = [[axes[0]], [axes[1]]]
    else:
        axes = [list(axes[0]), list(axes[1])]

    for col_idx, ((X, y), y_pred, dataset_id) in enumerate(zip(X_y_list, y_preds, dataset_ids)):
        X = X.copy()
        pressure_col="Pump outlet pressure (Mean)_std"
        if pressure_col not in X.columns:
            pressure_candidates = X.columns[X.columns.str.contains("press", case=False, na=False)]
            if len(pressure_candidates) == 0:
                pressure_col = None
            elif "TS outlet pressure (Mean)_std" in pressure_candidates:
                pressure_col = "TS outlet pressure (Mean)_std"
            else:
                pressure_col = pressure_candidates[0]

        title_base = f"Data nr {dataset_id} [test set]"

        # --- Row 1: True labels ---
        ax_true = axes[0][col_idx]
        if pressure_col and pressure_col in X.columns:
            ax_true.plot(X.index, X[pressure_col], label=pressure_col, alpha=0.5, color="darkorange")

        true_plug_events = X[y == 1]
        if pressure_col and pressure_col in X.columns:
            ax_true.scatter(true_plug_events.index, true_plug_events[pressure_col], color="blue", label="True Plug=1 (Pressure)", zorder=6, marker='x')

        ax_true.set_xlabel("Elapsed_seconds")
        ax_true.set_ylabel("Pressure [Pa]")
        ax_true.set_title(f"{title_base}\n[True labels]")

        # --- Row 2: Predictions ---
        ax_pred = axes[1][col_idx]
        if pressure_col and pressure_col in X.columns:
            ax_pred.plot(X.index, X[pressure_col], label=pressure_col, alpha=0.5, color="darkorange")

        plug_events = X[y_pred == 1]
        if pressure_col and pressure_col in X.columns:
            ax_pred.scatter(plug_events.index, plug_events[pressure_col], color="green", label="Predicted plug (Pressure)", zorder=7, marker='.')

        ax_pred.set_xlabel("Elapsed_seconds")
        ax_pred.set_ylabel("Pressure [Pa]")
        ax_pred.set_title(f"{title_base}\n[Predictions]")

    # Shared legend
    seen = set()
    all_handles, all_labels = [], []
    for ax in axes[0] + axes[1]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                all_handles.append(h)
                all_labels.append(l)

    axes[0][0].annotate("True labels", xy=(-0.35, 0.5), xycoords='axes fraction',
                        fontsize=12, fontweight='bold', ha='center', va='center', rotation=90)
    axes[1][0].annotate("Predictions", xy=(-0.35, 0.5), xycoords='axes fraction',
                        fontsize=12, fontweight='bold', ha='center', va='center', rotation=90)

    for col_idx, dataset_id in enumerate(dataset_ids):
        axes[0][col_idx].set_title(f"Dataset {dataset_id}")
        axes[1][col_idx].set_title("") 

    fig.legend(all_handles, all_labels, loc='lower center', bbox_to_anchor=(0.5, -0.05),
               ncol=3, fontsize=11)

    plt.subplots_adjust(hspace=0.4, top=0.90, bottom=0.12)
    plt.tight_layout(rect=[0.05, 0.08, 1, 0.95])

    os.makedirs("plots/test_sets", exist_ok=True)
    plt.savefig(f"plots/test_sets/{runId}.png", dpi=300, bbox_inches='tight')
    print(f"Plot saved as plots/test_sets/{runId}.png")
    plt.close()

## False alarm
def plot_false_alarm_rates(dataset_ids, X_y_list, y_preds, runId, threshold=0.05):
    """
    Calculates and plots the false alarm rate for datasets with no plug events.
    
    - `dataset_ids`: List of dataset IDs
    - `X_y_list`: List of tuples (X, y) for each dataset
    - `y_preds`: List of predicted target arrays
    - `runId`: Unique identifier for the run, used for saving the figure
    - `threshold`: False alarm rate threshold to display as a reference line (default: 0.05)
    """
    no_plug_dataset_ids = []
    false_alarm_rates = []

    for dataset_id, (X, y), y_pred in zip(dataset_ids, X_y_list, y_preds):
        if y.sum() == 0:
            tn, fp, _, _ = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
            far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            no_plug_dataset_ids.append(dataset_id)
            false_alarm_rates.append(far)
            print(f"Dataset {dataset_id} - False Alarm Rate: {far:.4f} ({fp} false alarms out of {fp+tn} timesteps)")

    if not no_plug_dataset_ids:
        print("No no-plug datasets found, skipping false alarm rate plot.")
        return

    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(len(no_plug_dataset_ids)), false_alarm_rates, color="steelblue")
    plt.xticks(range(len(no_plug_dataset_ids)), [str(d) for d in no_plug_dataset_ids])
    plt.axhline(y=threshold, color="red", linestyle="--", label=f"{int(threshold*100)}% threshold")
    plt.xlabel("Dataset ID")
    plt.ylabel("False Alarm Rate")
    plt.title("False Alarm Rate on No-Plug Datasets")
    plt.ylim(0, max(false_alarm_rates) * 1.2 + 0.01)
    plt.legend()

    for bar, far in zip(bars, false_alarm_rates):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.001,
                 f"{far:.3f}",
                 ha="center", va="bottom", fontsize=9)

    os.makedirs("plots/false_alarm_rates", exist_ok=True)
    plt.savefig(f"plots/false_alarm_rates/{runId}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 False alarm rate plot saved as plots/false_alarm_rates/{runId}.png")



## F1 score bar plots

def f1_score_bar_plot(dataset_ids, f1scores, runId):
    """
    Plots a bar plot of F1 scores for each dataset ID, comparing predicted vs true events where target_col=1.
    Plots the data from one model. 
    """
    plt.figure(figsize=(8, 5))
    bar  = plt.bar([str(id) for id in dataset_ids], f1scores, color='skyblue')
    for rect, score in zip(bar, f1scores):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height / 2, f"{score:.3f}", ha='center', va='center', color='black', fontsize=10, rotation=90)
    plt.xlabel('Dataset ID')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1)
    os.makedirs("plots/f1_scores", exist_ok=True)
    plt.savefig(f"plots/f1_scores/{runId}.png", dpi=300)
    print(f"F1 score plot saved as plots/f1_scores/{runId}.png")
    plt.close()


def f1_score_bar_plot_comparison(dataset_ids, f1scores_dict, name=None):
    """
    Plots a bar plot comparing F1 scores for each dataset ID across multiple models.
    f1scores_dict should be a dictionary where keys are model names and values are lists of F1 scores corresponding to dataset_ids.
    """
    x = np.arange(len(dataset_ids))  
    width = 0.25  

    plt.figure(figsize=(10, 6))

    for i, (model_name, f1scores) in enumerate(f1scores_dict.items()):
        bars = plt.bar(x + i*width, f1scores, width=width, label=model_name)

        if len(next(iter(f1scores_dict.values()))) <= 4: # Only add text if there are 4 or fewer datasets to avoid clutter
            for bar, score in zip(bars, f1scores):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f"{score:.3f}", ha='center', va='center', color='black', fontsize=8)

    plt.xlabel('Dataset ID')
    plt.ylabel('F1 Score')
    plt.xticks(x + width * (len(f1scores_dict) - 1) / 2, [str(id) for id in dataset_ids])
    plt.ylim(0, 1)
    plt.legend()
    os.makedirs("plots/f1_scores", exist_ok=True)
    if name:
        plt.savefig(f"plots/f1_scores/comparison_{name}.png", dpi=300)
        print(f"F1 score comparison plot saved as plots/f1_scores/comparison_{name}.png")
    else:
        plt.savefig("plots/f1_scores/comparison.png", dpi=300)
        print("F1 score comparison plot saved as plots/f1_scores/comparison.png")
    plt.close()


def f1_score_line_plot_comparison(dataset_ids, f1scores_dict):
    """
    Plots a line plot of F1 scores for each dataset ID, comparing predicted vs true events where target_col=1.
    f1scores_dict should be a dictionary where keys are model names and values are lists of F1 scores corresponding to dataset_ids.
    """

    plt.figure(figsize=(10, 6))

    for model_name, f1scores in f1scores_dict.items():
        plt.plot([str(id) for id in dataset_ids], f1scores, marker='o', label=model_name)

    plt.xlabel('Dataset ID')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1)
    plt.legend()
    os.makedirs("plots/f1_scores", exist_ok=True)
    plt.savefig("plots/f1_scores/comparison_line_plot.png", dpi=300)
    print("F1 score comparison line plot saved as plots/f1_scores/comparison_line_plot.png")
    plt.close()


## -------- Horizon -----------
def plot_test_f1_vs_horizon(
    horizons,
    scores,
    invert_xaxis=False,
    figsize=(8,5),
    window_size=None,
    test_or_val="Test",
):
    """
    Plots a line plot of F1 scores across different prediction horizons, comparing their performance on either the test or validation set.
        - `horizons`: List of prediction horizons (in seconds) corresponding to the F1 scores.
        - `scores`: List of F1 scores corresponding to each horizon.
        - `invert_xaxis`: If True, the x-axis will be inverted to show longer horizons on the left and shorter horizons on the right.
        - `figsize`: Tuple specifying the size of the figure (width, height) in inches.
        - `window_size`: Optional integer specifying the window size used in the model, included in the filename when saving the plot.
        - `test_or_val`: String indicating whether the scores are from the "Test" set or "Validation" set, used for labeling the axes and filename when saving the plot.
    """
    horizons = np.array(horizons)
    scores = np.array(scores)

    plt.figure(figsize=figsize)

    horizon_seconds = [f"{h} ({h}s)" for h in horizons]

    plt.plot(horizon_seconds, scores,
             marker="o", linewidth=2, label="Test F1" if test_or_val == "Test" else "Validation F1")

    plt.xlabel("Prediction Horizon samples (seconds)")
    plt.ylabel(f"F1 Score ({test_or_val} Set)")
    plt.legend()
    plt.grid(True)

    if invert_xaxis:
        plt.gca().invert_xaxis()

    plt.tight_layout()
    os.makedirs("plots/horizon_test", exist_ok=True)
    if window_size:
        plt.savefig(f"plots/horizon_test/f1_scores_line_plot_{window_size}window_{test_or_val}.png", dpi=300)
        print(f"F1 score line plot saved as plots/horizon_test/f1_scores_line_plot_{window_size}window_{test_or_val}.png")
    else:
        plt.savefig(f"plots/horizon_test/f1_scores_line_plot_{test_or_val}_{horizons}.png", dpi=300)
        print(f"F1 score line plot saved as plots/horizon_test/f1_scores_line_plot_{test_or_val}_{horizons}.png")
    plt.close()


def plot_test_f1_vs_horizon_bar(
    horizons,
    scores,
    invert_xaxis=False,
    figsize=(8,5),
    window_size=None,
    test_or_val="Test",
):
    """
    Plots a bar plot of F1 scores across different prediction horizons, comparing their performance on either the test or validation set.
        - `horizons`: List of prediction horizons (in seconds) corresponding to the F1 scores.
        - `scores`: List of F1 scores corresponding to each horizon.
        - `invert_xaxis`: If True, the x-axis will be inverted to show longer horizons on the left and shorter horizons on the right.
        - `figsize`: Tuple specifying the size of the figure (width, height) in inches.
        - `window_size`: Optional integer specifying the window size used in the model, included in the filename when saving the plot.
        - `test_or_val`: String indicating whether the scores are from the "Test" set or "Validation" set, used for labeling the axes and filename when saving the plot.
    """
    horizons = np.array(horizons)
    scores = np.array(scores)

    plt.figure(figsize=figsize)

    x = np.arange(len(horizons)) 

    plt.bar(x, scores, width=0.6, alpha=0.8)

    best_score = np.max(scores)
    plt.axhline(best_score, color="red", linestyle="--", linewidth=2)
    plt.text(len(x)-0.5, best_score + 0.005,
             f"Best: {best_score:.3f}",
             color="red", ha="right")
    
    for i in range(len(x)):
        plt.text(i, scores[i] // 2, f"{scores[i]:.3f}", ha='center')

    plt.xlabel("Prediction Horizon samples (seconds)")
    plt.ylabel(f"F1 Score ({test_or_val} Set)")
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    horizon_seconds = [f"{h} ({h}s)" for h in horizons]
    plt.xticks(x, horizon_seconds)

    if invert_xaxis:
        plt.gca().invert_xaxis()

    plt.tight_layout()
    os.makedirs("plots/horizon_test", exist_ok=True)
    if window_size:
        plt.savefig(f"plots/horizon_test/f1_scores_bar_plot_{window_size}window_{test_or_val}_{horizons}.png", dpi=300)
        print(f"F1 score bar plot saved as plots/horizon_test/f1_scores_bar_plot_{window_size}window_{test_or_val}_{horizons}.png")
    else:
        plt.savefig(f"plots/horizon_test/f1_scores_bar_plot_{test_or_val}_{horizons}.png", dpi=300)
        print(f"F1 score bar plot saved as plots/horizon_test/f1_scores_bar_plot_{test_or_val}_{horizons}.png")
    plt.close()
