"""
Plotting and visualization helpers for visualizing predictions.
"""
import os
import math
import matplotlib.pyplot as plt
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


def _check_pressure_col(df, pressure_col="Pump outlet pressure (Mean)_std"):
    """Helper function to check for the presence of a pressure column in the dataframe, and return its name if found. Used for flexible plotting of pressure signals in prediction visualization."""
    
    if pressure_col not in df.columns: # If selected column is not found, try some common alternatives
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

    pressure_col = _check_pressure_col(df)
    y_true = df[target_col] if target_col else None

    _plot_pressure_and_events(plt.gca(), df, y_true, y_pred, pressure_col, show_true=plotLabel)

    plt.legend()
    plt.show()


### VISUALIZING RESULTS FOR PREDICT ALL ###
def _plot_pressure_and_events(ax, X, y_true, y_pred, pressure_col, show_true=True):
    """Helper function to plot pressure signal with true and predicted plug events highlighted, used for prediction visualization."""
    if pressure_col in X.columns: 
        ax.plot(X.index, X[pressure_col], label={pressure_col}, alpha=0.5, color="steelblue")

    # Highlight true Plug events, on training data
    if show_true and y_true is not None:
        true_plug_events = X[y_true == 1]   
        if pressure_col and pressure_col in X.columns:
            ax.scatter(true_plug_events.index, true_plug_events[pressure_col], color="red", label="True Plug=1 (Pressure)", zorder=6, marker='x', s=50, linewidth=2)
    
    # Highlight predicted Plug events, on all data
    if y_pred is not None:
        plug_events = X[y_pred == 1]
        if pressure_col and pressure_col in X.columns:
            ax.scatter(plug_events.index, plug_events[pressure_col], color="yellow", label="Predicted plug (Pressure)", zorder=7, marker='.', s=50)  
     
    ax.set_xlabel("Elapsed_seconds")
    ax.set_ylabel("Pressure [Pa]")


def _collect_unique_legend_handles(axes):
    """Helper function to collect unique legend handles and labels from a list of axes, used for creating a shared legend in prediction visualization."""
    seen, handles, labels = set(), [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in seen:
                seen.add(label)
                handles.append(handle)
                labels.append(label)
    return handles, labels



def _plot_one(ax, df, y_pred, figureNum, y, dataset_id = None, pressure_col="Pump outlet pressure (Mean)_std"):
    """
    Plot flow rate and pressure with true and predicted plug events for one dataset on a given axis.
    Used as a helper function for plot_all_predictions to create subplots for each dataset.
    """

    # Check for pressure column
    pressure_col = _check_pressure_col(df, pressure_col)

    # Determine if this dataset is part of training or test set for labeling the plot
    shown_id = dataset_id if dataset_id is not None else figureNum
    dataset_nrs = list(dataset_nr)
    test_set_nr = max(dataset_nrs)
    test_set_nrs = test_sets if test_sets else [test_set_nr]
    train_set_nrs = [nr for nr in dataset_nrs if nr not in test_set_nrs]
    
    # Plot pressure and events
    _plot_pressure_and_events(ax, df, y, y_pred, pressure_col, show_true={shown_id not in test_set_nrs})

    # Set title based on dataset ID and whether it's in the test set or training set
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
        _plot_one(axes[subplot_idx-1], X, y_pred, subplot_idx, y, dataset_id=dataset_id)
    
    # Collect unique legend handles and labels for a shared legend
    handles, labels = _collect_unique_legend_handles(axes)

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
    """
    Plots a summary of the test data with true vs predicted events for all test datasets as seperate plots.
    """

    n_plots = len(X_y_list)
    fig, axes = plt.subplots(nrows=2, ncols=n_plots, figsize=(4 * n_plots, 3.2 * 2))

    # Ensure axes is a 2D array for consistent indexing
    if n_plots == 1:
        axes = [[axes[0]], [axes[1]]]
    else:
        axes = [list(axes[0]), list(axes[1])]

    # For each dataset, plot true labels in the first row and predictions in the second row
    for col_idx, ((X, y), y_pred, dataset_id) in enumerate(zip(X_y_list, y_preds, dataset_ids)):
        X = X.copy()
        
        pressure_col = _check_pressure_col(X) # Check for pressure column
        
        _plot_pressure_and_events(axes[0][col_idx], X, y, None, pressure_col, show_true=True)  # True labels on first row
        axes[0][col_idx].set_title(f"Dataset {dataset_id}") # Set title for true labels row

        _plot_pressure_and_events(axes[1][col_idx], X, None, y_pred, pressure_col, show_true=False)  # Predictions on second row
        axes[1][col_idx].set_title("") # No title for predictions row to avoid redundancy, as the dataset ID is already in the true labels row

    # Add shared y-axis label for pressure and x-axis label for time
    axes[0][0].annotate("True labels", xy=(-0.35, 0.5), xycoords='axes fraction',
                        fontsize=12, fontweight='bold', ha='center', va='center', rotation=90)
    axes[1][0].annotate("Predictions", xy=(-0.35, 0.5), xycoords='axes fraction',
                        fontsize=12, fontweight='bold', ha='center', va='center', rotation=90)

    # Shared legend
    handles, labels = _collect_unique_legend_handles(axes[0] + axes[1])
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05),
               ncol=3, fontsize=11)

    plt.subplots_adjust(hspace=0.4, top=0.90, bottom=0.12)
    plt.tight_layout(rect=[0.05, 0.08, 1, 0.95])

    # Save
    os.makedirs("plots/test_sets", exist_ok=True)
    plt.savefig(f"plots/test_sets/{runId}.png", dpi=300, bbox_inches='tight')
    print(f"Plot saved as plots/test_sets/{runId}.png")
    plt.close()