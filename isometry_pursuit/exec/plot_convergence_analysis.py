#!/usr/bin/env python3
"""
Plot convergence analysis results from saved data files.
This script generates plots from convergence data saved by run_convergence_analysis.py.
"""

import argparse
import os
import pickle

from isometry_pursuit.exec.plot_results import (
    plot_coefficient_convergence,
    plot_sparsity_convergence_boxplot,
)
import matplotlib.pyplot as plt
import numpy as np


def plot_support_cardinality_histogram(
    convergence_logs, save_path=None, threshold=1e-4
):
    """
    Plot histogram of support cardinality at the highest iteration number across all replicates.
    Shows the raw number of features with non-zero coefficients.

    Parameters:
    -----------
    convergence_logs : list
        List of convergence data dictionaries from multiple replicates
    save_path : str, optional
        Path to save the plot
    threshold : float
        Threshold for determining non-zero coefficients
    """
    if not convergence_logs or all(log is None for log in convergence_logs):
        print("No valid convergence logs to plot")
        return

    # Extract raw cardinality data at highest iteration from all replicates
    final_cardinalities = []

    for log in convergence_logs:
        if log is not None and "coefficient_norms" in log and "iterations" in log:
            coefficient_norms = log["coefficient_norms"]
            iterations = np.array(log["iterations"])
            sparsities = np.array(log["sparsities"])

            # Filter out failed iterations (sparsities = -1)
            valid_mask = sparsities >= 0
            valid_norms = coefficient_norms[valid_mask]
            valid_iterations = iterations[valid_mask]

            if len(valid_norms) > 0:
                # Get the coefficient norms at the highest iteration (last valid iteration)
                final_norms = valid_norms[-1]

                # Calculate raw cardinality (number of features with non-zero coefficients)
                cardinality = np.sum(final_norms > threshold)
                final_cardinalities.append(cardinality)

    if not final_cardinalities:
        print("No valid cardinality data to plot")
        return

    # Create histogram
    plt.figure(figsize=(10, 6))

    # Use the same purple color as other sparsity plots
    garfield_purple = "#4E2A84"

    # Create proper binning for integer cardinality values
    min_val = min(final_cardinalities)
    max_val = max(final_cardinalities)
    bins = range(min_val, max_val + 2)

    plt.hist(
        final_cardinalities,
        bins=bins,
        color=garfield_purple,
        edgecolor="white",
        align="left",
        alpha=0.8,
    )

    plt.xlabel(
        r"$|\widehat{S}|$ (Number of Features with Non-zero Coefficients)", fontsize=16
    )
    plt.ylabel("Frequency", fontsize=16)
    plt.title(
        "Support Cardinality at Highest Iteration", fontsize=18, fontweight="bold"
    )
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add statistics text
    mean_cardinality = np.mean(final_cardinalities)
    std_cardinality = np.std(final_cardinalities)
    plt.text(
        0.02,
        0.98,
        f"Mean: {mean_cardinality:.1f}\nStd: {std_cardinality:.1f}\nN: {len(final_cardinalities)}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Support cardinality histogram saved to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_convergence_analysis_script(
    convergence_file,
    dataset_name,
    output_dir="figures",
    max_features=20,
):
    """
    Generate plots from saved convergence analysis data.

    Parameters:
    -----------
    convergence_file : str
        Path to the convergence pickle file (all_replicates.pkl)
    dataset_name : str
        Name of the dataset
    output_dir : str
        Directory to save plots
    max_features : int
        Maximum number of features to plot in coefficient convergence
    """
    print(f"Loading convergence data from {convergence_file}")

    # Load convergence logs
    with open(convergence_file, "rb") as f:
        convergence_logs = pickle.load(f)

    if not convergence_logs:
        print("No convergence data found in file")
        return

    print(f"Loaded {len(convergence_logs)} convergence logs")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate plots
    print("Generating convergence plots...")

    # Sparsity convergence boxplot across replicates
    plot_sparsity_convergence_boxplot(
        convergence_logs,
        save_path=os.path.join(
            output_dir, f"{dataset_name}_sparsity_convergence_boxplot.png"
        ),
    )

    # Support cardinality histogram at highest iteration
    plot_support_cardinality_histogram(
        convergence_logs,
        save_path=os.path.join(
            output_dir, f"{dataset_name}_support_cardinality_histogram.png"
        ),
    )

    # Individual plots for the first replicate
    first_replicate = convergence_logs[0]

    # Coefficient convergence plot
    plot_coefficient_convergence(
        first_replicate,
        save_path=os.path.join(
            output_dir, f"{dataset_name}_coefficient_convergence.png"
        ),
        max_features=max_features,
    )

    print(f"Convergence plots completed!")
    print(f"Plots saved to: {output_dir}/{dataset_name}_convergence_*.png")


def main():
    parser = argparse.ArgumentParser(description="Plot convergence analysis results")
    parser.add_argument(
        "--convergence-file",
        required=True,
        help="Path to convergence pickle file (all_replicates.pkl)",
    )
    parser.add_argument("--dataset-name", required=True, help="Name of the dataset")
    parser.add_argument(
        "--output-dir", default="figures", help="Output directory for plots"
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=20,
        help="Maximum number of features to plot",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Plot convergence analysis
    plot_convergence_analysis_script(
        convergence_file=args.convergence_file,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        max_features=args.max_features,
    )


if __name__ == "__main__":
    main()
