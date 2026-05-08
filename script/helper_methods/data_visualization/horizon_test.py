"""
This script contains helper functions for visualizing the results of the horizon test experiments.
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

    horizon_seconds = [f"{h} ({h}s)" for h in horizons] # Create labels with seconds for x-axis

    # Plot F1 scores as a line plot with markers
    plt.plot(horizon_seconds, scores,
             marker="o", linewidth=2, label="Test F1" if test_or_val == "Test" else "Validation F1")

    plt.xlabel("Prediction Horizon samples (seconds)")
    plt.ylabel(f"F1 Score ({test_or_val} Set)")
    plt.legend()
    plt.grid(True)

    if invert_xaxis:
        plt.gca().invert_xaxis()

    plt.tight_layout()

    # Save
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

    x = np.arange(len(horizons)) # Create x positions for the bars

    # Plot F1 scores as a bar plot
    plt.bar(x, scores, width=0.6, alpha=0.8)

    # Add horizontal line for best score and annotate it
    best_score = np.max(scores)
    plt.axhline(best_score, color="red", linestyle="--", linewidth=2)
    plt.text(len(x)-0.5, best_score + 0.005,
             f"Best: {best_score:.3f}",
             color="red", ha="right")

    # Annotate each bar with its F1 score
    for i in range(len(x)):
        plt.text(i, scores[i] // 2, f"{scores[i]:.3f}", ha='center')

    plt.xlabel("Prediction Horizon samples (seconds)")
    plt.ylabel(f"F1 Score ({test_or_val} Set)")
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    horizon_seconds = [f"{h} ({h}s)" for h in horizons] # Create labels with seconds for x-axis
    plt.xticks(x, horizon_seconds)

    if invert_xaxis:
        plt.gca().invert_xaxis()

    plt.tight_layout()

    # Save
    os.makedirs("plots/horizon_test", exist_ok=True)
    if window_size:
        plt.savefig(f"plots/horizon_test/f1_scores_bar_plot_{window_size}window_{test_or_val}_{horizons}.png", dpi=300)
        print(f"F1 score bar plot saved as plots/horizon_test/f1_scores_bar_plot_{window_size}window_{test_or_val}_{horizons}.png")
    else:
        plt.savefig(f"plots/horizon_test/f1_scores_bar_plot_{test_or_val}_{horizons}.png", dpi=300)
        print(f"F1 score bar plot saved as plots/horizon_test/f1_scores_bar_plot_{test_or_val}_{horizons}.png")
    plt.close()
