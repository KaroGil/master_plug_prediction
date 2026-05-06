"""
Plotting and visualization helpers for visualizing predictions.
"""
import os
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from script.helper_methods.config import get_config

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

dataset_nr = cfg['data']['datasets']
target_col = cfg["data"]["target"]
test_sets = cfg["data"]["test_sets"]

def check_pressure_col(df, pressure_col="Pump outlet pressure (Mean)_std"):
    """Helper function to check for the presence of a pressure column in the dataframe, and return its name if found. Used for flexible plotting of pressure signals in prediction visualization."""
    if pressure_col not in df.columns:
        pressure_col = "Pump outlet pressure (Mean)_mean"
        if pressure_col not in df.columns:
            pressure_candidates = df.columns[df.columns.str.contains("press", case=False, na=False)]
            if len(pressure_candidates) > 0:
                return pressure_candidates[0]
            else:
                return None
    return pressure_col

### VISUALIZING RESULTS FOR PREDICT ONE DATASET ###
def visualize_predicted_vs_true(df, y_pred, plotLabel=True):
    """Visualize predicted vs true plug events on top of flow rate and pressure signals, with optional vertical lines for specific times based on dataset ID. Used for prediction visualization."""
    plt.figure(figsize=(12,6))

    pressure_col = check_pressure_col(df)

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

    plt.legend()
    plt.show()


### VISUALIZING RESULTS FOR PREDICT_ALL ###
# Used in plot_all_predictions.py
def plot_one(ax, df, y_pred, figureNum, y, dataset_id = None, pressure_col="Pump outlet pressure (Mean)_std"):
    """
    Plot flow rate and pressure with true and predicted plug events for one dataset on a given axis.
    Used as a helper function for plot_all_predictions to create subplots for each dataset.
    """
    pressure_col = check_pressure_col(df, pressure_col)

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

# Used in plot()^
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

# Used in predict_all.py
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
        
        pressure_col = check_pressure_col(X)
        
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
# Used in predict_all.py
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

