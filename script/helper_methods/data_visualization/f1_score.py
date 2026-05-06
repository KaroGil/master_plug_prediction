"""
F1 Score Visualization Helpers
- Bar plots of F1 scores across datasets for a single model
- Comparison bar plots of F1 scores across datasets for multiple models
- Line plots of F1 scores across datasets for multiple models
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Config for plots
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

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

# Used in compare_model_predictions.py
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

