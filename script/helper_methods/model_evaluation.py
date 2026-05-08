"""
Utility functions for evaluating binary classification models on test data.

Includes:
- metric computation (precision, recall, F1, ROC-AUC, PR-AUC)
- formatted metric printing
- full multi-model test evaluation
- confusion matrix, ROC, and PR visualizations
- per-dataset F1 summary statistics
- false-alarm / missed-detection timeline analysis
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from .config import get_config
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, roc_auc_score
)

# Load config
cfg = get_config()
HORIZON = cfg["experiment"]["horizon"]
dataset_ids = cfg["data"]["datasets"]


def evaluate_model_on_test(model, X_test, y_test):
    """Evaluate the final model on the test dataset"""
    
    y_test_pred = model.predict(X_test) # Get predicted labels for the test set

    # Evaluate and print classification report and F1 score
    print("Test set results:")
    print(classification_report(y_test, y_test_pred, digits=3, zero_division=0))
    f1_score_value = f1_score(y_test, y_test_pred, average='weighted')
    print("F1 Score:", f1_score_value)

    return y_test_pred, f1_score_value



def get_metrics(y_true, y_pred, y_prob=None, model_name="Dummy Classifier"):
    """
    Returns a dict of all key metrics for one model.
    y_prob: predicted probability for class 1 (optional, needed for AUC scores).
    """
    # Evaluate precision, recall, F1 for each class and weighted average
    metrics = {
        "model": model_name,
        "precision_0":        round(precision_score(y_true, y_pred, pos_label=0, zero_division=0), 4),
        "precision_1":        round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "recall_0":           round(recall_score(y_true, y_pred, pos_label=0, zero_division=0), 4),
        "recall_1":           round(recall_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "f1_0":               round(f1_score(y_true, y_pred, pos_label=0, zero_division=0), 4),
        "f1_1":               round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "weighted_f1":        round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }
    
    # Compute ROC-AUC and PR-AUC if predicted probabilities are available
    if y_prob is not None:
        metrics["roc_auc"]  = round(roc_auc_score(y_true, y_prob), 4)
        metrics["pr_auc"]   = round(average_precision_score(y_true, y_prob), 4)
    else:
        metrics["roc_auc"]  = None
        metrics["pr_auc"]   = None
 
    return metrics
 
 
def print_metrics(metrics: dict):
    """Pretty-prints the metrics dict."""
    print(f"\n{'─'*45}")
    print(f"  {metrics['model']}")
    print(f"{'─'*45}")
    print(f"  Weighted F1    : {metrics['weighted_f1']}")
    print(f"  ROC-AUC        : {metrics['roc_auc']}")
    print(f"  PR-AUC         : {metrics['pr_auc']}")
    print(f"\n  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'─'*42}")
    for cls, label in [(0, 'No Plug (0)'), (1, 'Plug (1)')]:
        print(f"  {label:<12} {metrics[f'precision_{cls}']:>10} "
              f"{metrics[f'recall_{cls}']:>10} {metrics[f'f1_{cls}']:>10}")
    print(f"{'─'*45}\n")


def print_class_distribution_across_datasets(dataset_ids, X_y_list):
    """Prints the class distribution for each dataset."""

    print("\n⚖️  Class distribution in datasets:")
    print(f"{'Dataset':<12} {'Samples':>10} {'Plug (1)':>12} {'No Plug (0)':>14}")
    print("-" * 55)
    for  ds_id in dataset_ids:
        y = X_y_list[dataset_ids.index(ds_id)][1]
        n_samples  = len(y)
        n_plug     = (y == 1).sum()
        n_no_plug  = (y == 0).sum()
        plug_ratio = n_plug / n_samples
        print(f"{ds_id:<12} {n_samples:>10} {n_plug:>7} ({plug_ratio*100:.1f}%) {n_no_plug:>7} ({(1-plug_ratio)*100:.1f}%)")


def evaluate_all_models(best_of_all_models, X_test, y_test, horizon, save_dir="figures"):
    """
    Takes the best_of_all_models dict from find_best_model and runs
    full evaluation on the held-out test set for every model.
    """
    os.makedirs(save_dir, exist_ok=True) # Ensure the save directory exists

    all_metrics = {}

    # Loop through each model, evaluate on test set, and save dashboard plots
    for name, (model, _, _) in best_of_all_models.items():
        print(f"\nEvaluating {name}...")
        print(f"Model parameters: {model.get_params()}")

        y_pred = model.predict(X_test) # Get predicted labels for the test set

        # Get predicted probabilities for class 1 if available (needed for ROC and PR curves)
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]

        # Define save path for the dashboard plot, organized by horizon if not default
        if horizon != HORIZON:
            os.makedirs(f"{save_dir}/horizon_test", exist_ok=True)
            path = f"{save_dir}/horizon_test/{horizon}_{name.lower().replace(' ', '_')}_dashboard.png"
        else:
            path = f"{save_dir}/{horizon}_{name.lower().replace(' ', '_')}_dashboard.png"

        # Generate and save the dashboard with all metrics and plots
        metrics = plot_dashboard(
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            model_name=name,
            save_path=path
        )

        all_metrics[name] = metrics

    return all_metrics


def plot_confusion_matrix(y_true, y_pred, model_name="Dummy Classifier",
                           normalize=True, ax=None, save_path=None):
    """
    Plots a confusion matrix.
    normalize=True shows recall per class 
    """

    standalone = ax is None
    if standalone: # Create a new figure if no axis is provided (standalone mode)
        _, ax = plt.subplots(figsize=(5, 4))
 
    cm = confusion_matrix(y_true, y_pred, normalize="true" if normalize else None) # Get confusion matrix 
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Plug (0)", "Plug (1)"]
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format=".2f" if normalize else "d")
    ax.set_title(f"{model_name}\nConfusion Matrix{'  (normalized)' if normalize else ''}")
 
    if standalone: # Only show/save the plot if we're in standalone mode (not part of a larger dashboard)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved → {save_path}")
        plt.show()
 
 
def plot_roc_curve(y_true, y_prob, model_name="Dummy Classifier",
                   ax=None, save_path=None):
    """
    Plots ROC curve for one model.
    """

    standalone = ax is None
    if standalone: # Create a new figure if no axis is provided (standalone mode)
        _, ax = plt.subplots(figsize=(5, 4))
    
    # Compute false positive rate, true positive rate, and AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot the ROC curve and the random baseline
    ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
 
    if standalone: # Only show/save the plot if we're in standalone mode (not part of a larger dashboard)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved → {save_path}")
        plt.show()
 
    return fpr, tpr, roc_auc
 
 
def plot_pr_curve(y_true, y_prob, model_name="Dummy Classifier",
                  ax=None, save_path=None):
    """
    Plots Precision-Recall curve for one model
    """

    standalone = ax is None
    if standalone: # Create a new figure if no axis is provided (standalone mode)
        _, ax = plt.subplots(figsize=(5, 4))
    
    # Compute precision, recall, and average precision (PR-AUC)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    baseline = np.mean(y_true) 
    
    # Plot the Precision-Recall curve and the random baseline
    ax.plot(recall, precision, lw=2, label=f"{model_name} (AP = {pr_auc:.3f})")
    ax.axhline(y=baseline, color="k", linestyle="--", lw=1,
               label=f"Random baseline ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
 
    if standalone: # Only show/save the plot if we're in standalone mode (not part of a larger dashboard)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved → {save_path}")
        plt.show()
 
    return precision, recall, pr_auc
 
 
def plot_dashboard(y_true, y_pred, y_prob=None, model_name="Dummy Classifier",
                   save_path=None):
    """
    One-call function: prints metrics + plots confusion matrix, ROC, and PR
    in a single figure. y_prob is optional but strongly recommended.
    """

    # Compute and print all metrics
    metrics = get_metrics(y_true, y_pred, y_prob, model_name)
    print_metrics(metrics)
    
    # Plot confusion matrix, ROC curve, and PR curve 
    n_plots = 3 if y_prob is not None else 1
    _, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]
 
    plot_confusion_matrix(y_true, y_pred, model_name=model_name, ax=axes[0])
 
    if y_prob is not None:
        plot_roc_curve(y_true, y_prob, model_name=model_name, ax=axes[1])
        plot_pr_curve(y_true, y_prob, model_name=model_name, ax=axes[2])
 
    plt.tight_layout()
    
    # Save the dashboard figure if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
 
    return metrics


def per_dataset_statistics(f1_scores_dict):
    """
    Prints mean, std, min, max F1 across datasets.
    """

    print(f"\n{'─'*55}")
    print("  Per-dataset F1 statistics")
    print(f"{'─'*55}")
    print(f"  {'Model':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'─'*52}")

    # Loop through each model and its list of F1 scores, compute statistics, and print them in a formatted way
    for model_name, scores in f1_scores_dict.items():
        scores = np.array(scores)
        print(f"  {model_name:<20} "
              f"{scores.mean():>8.4f} "
              f"{scores.std():>8.4f} "
              f"{scores.min():>8.4f} "
              f"{scores.max():>8.4f}")


def false_alarm_analysis(y_true, y_pred, dataset_id,
                         sample_rate_hz=2, model_name="Model"):
    """
    Plots where false alarms (FP) and missed detections (FN)
    occur over time in a single dataset.
    sample_rate_hz: sampling frequency of your sensor (2Hz = 0.5s per sample)
    """
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    time_axis = np.arange(len(y_true)) / sample_rate_hz  # seconds

    fp_mask = (y_pred == 1) & (y_true == 0)  # false alarms
    fn_mask = (y_pred == 0) & (y_true == 1)  # missed plugs
    tp_mask = (y_pred == 1) & (y_true == 1)  # correct plug detections

    _, ax = plt.subplots(figsize=(14, 3))

    # Highlight true plug regions in the background
    plug_regions = np.where(y_true == 1)[0]
    if len(plug_regions) > 0:
        ax.fill_between(time_axis, 0, 1,
                        where=(y_true == 1),
                        alpha=0.15, color="#2196F3",
                        label="True plug region")

    # Plot false alarms and missed plugs as vertical lines with different colors and labels
    ax.scatter(time_axis[tp_mask],
               np.ones(tp_mask.sum()) * 0.7,
               marker="|", color="#4CAF50", s=60,
               label=f"True positive ({tp_mask.sum()})", alpha=0.7)
    ax.scatter(time_axis[fp_mask],
               np.ones(fp_mask.sum()) * 0.5,
               marker="|", color="#E65100", s=60,
               label=f"False alarm ({fp_mask.sum()})", alpha=0.7)
    ax.scatter(time_axis[fn_mask],
               np.ones(fn_mask.sum()) * 0.3,
               marker="|", color="#9C27B0", s=60,
               label=f"Missed plug ({fn_mask.sum()})", alpha=0.7)

    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0),
               ncol=4, fontsize=10, frameon=False)
    plt.tight_layout(rect=[0, 0.12, 1, 1.0])
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    plt.tight_layout()
    # Save the false alarm figure
    plt.savefig(f"figures/false_alarm_timeline_"
                f"{model_name.replace(' ', '_')}_ds{dataset_id}.png",
                bbox_inches="tight")
    print(f"\nSaved → figures/false_alarm_timeline_{model_name.replace(' ', '_')}_ds{dataset_id}.png")

    # Print summary statistics about the number of true positives, false alarms, and missed plugs for this dataset
    print(f"\nDataset {dataset_id} — {model_name}")
    print(f"  True positives : {tp_mask.sum()}")
    print(f"  False alarms   : {fp_mask.sum()}")
    print(f"  Missed plugs   : {fn_mask.sum()}")


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

    # Calculate false alarm rate for datasets with no plug events
    for dataset_id, (_, y), y_pred in zip(dataset_ids, X_y_list, y_preds):
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
    # Create a bar plot of false alarm rates for no-plug datasets, with a horizontal line indicating the threshold
    bars = plt.bar(range(len(no_plug_dataset_ids)), false_alarm_rates, color="steelblue")
    plt.xticks(range(len(no_plug_dataset_ids)), [str(d) for d in no_plug_dataset_ids])
    plt.axhline(y=threshold, color="red", linestyle="--", label=f"{int(threshold*100)}% threshold")
    plt.xlabel("Dataset ID")
    plt.ylabel("False Alarm Rate")
    plt.title("False Alarm Rate on No-Plug Datasets")
    plt.ylim(0, max(false_alarm_rates) * 1.2 + 0.01)
    plt.legend()

    # Annotate each bar with its false alarm rate value
    for bar, far in zip(bars, false_alarm_rates):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.001,
                 f"{far:.3f}",
                 ha="center", va="bottom", fontsize=9)

    # Save
    os.makedirs("plots/false_alarm_rates", exist_ok=True)
    plt.savefig(f"plots/false_alarm_rates/{runId}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 False alarm rate plot saved as plots/false_alarm_rates/{runId}.png")

