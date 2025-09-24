#!/usr/bin/env python3
"""
Plotting module for isometry pursuit experiment results.
Creates various plots from experiment results.
"""

import argparse
import sys
import os
import pickle
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import isometry_pursuit modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from isometry_pursuit.plot import box_plot, ecdf_plot
from isometry_pursuit.plot import plot_convergence
from isometry_pursuit.experiment import analyze_convergence_logs
from isometry_pursuit.algorithm import (
    create_convergence_log_with_coefficients,
    save_convergence_data,
    load_convergence_data,
)


def plot_experiment_results(
    results_file, dataset_name, D=None, power=None, output_dir="figures"
):
    """
    Create plots from experiment results.

    Parameters:
    -----------
    results_file : str
        Path to results CSV file
    dataset_name : str
        Name of the dataset
    D : int
        Target dimension
    power : float
        Power parameter used
    output_dir : str
        Directory to save plots
    """

    print(f"Loading results from {results_file}")
    results_df = pd.read_csv(results_file)

    # Parse D and power from filename if not provided
    if D is None or power is None:
        import re

        filename = os.path.basename(results_file)
        # Pattern: {dataset}_results.csv
        match = re.match(r"(.+)_results\.csv", filename)
        if match:
            dataset_name_from_file = match.group(1)
            print(f"Parsed dataset name from filename: {dataset_name_from_file}")
            # Set defaults if not provided
            if power is None:
                power = 1.0  # Default power
            if D is None:
                # Set default D based on dataset name
                if dataset_name_from_file == "iris":
                    D = 4
                elif dataset_name_from_file == "wine":
                    D = 6
                elif dataset_name_from_file == "word_embeddings":
                    D = 10
                elif dataset_name_from_file == "ethanol":
                    D = 2
                else:
                    raise ValueError(
                        f"Unknown dataset {dataset_name_from_file}, please provide D parameter"
                    )
            print(f"Using defaults: power={power}, D={D}")
        else:
            if D is None or power is None:
                raise ValueError(
                    f"Could not parse dataset name from filename {filename}, and D/power were not provided as arguments"
                )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Main comparison (Greedy vs Two-Stage)
    print("Creating main loss comparison plot (Greedy vs Two-Stage)...")
    if "Losses" not in results_df.columns:
        raise ValueError("Missing required column: 'Losses' (Greedy method)")
    if "Two-Stage Losses" not in results_df.columns:
        raise ValueError("Missing required column: 'Two-Stage Losses'")

    greedy_losses = results_df["Losses"].dropna()
    two_stage_losses = results_df["Two-Stage Losses"].dropna()

    if len(greedy_losses) == 0:
        raise ValueError(
            f"'Losses' column contains no valid data (all NaN) for dataset {dataset_name}. This suggests the greedy method failed to run properly."
        )
    if len(two_stage_losses) == 0:
        raise ValueError(
            f"'Two-Stage Losses' column contains no valid data (all NaN) for dataset {dataset_name}. This suggests the two-stage method failed to run properly."
        )

    plot_filename = os.path.join(
        output_dir, f"{dataset_name}_greedy_vs_twostage_box.png"
    )
    box_plot(
        greedy_losses,
        two_stage_losses,
        D,
        ["Greedy", "Two-Stage"],
        r"$\ell_1(\hat{S}_{g}) - \ell_1(\hat{S}_{TSIP})$",
        plot_filename,
    )
    print(f"Saved plot: {plot_filename}")

    # Plot 2: Optional brute force comparison (only if data exists)
    if (
        "Brute Isometry Losses" in results_df.columns
        and len(results_df["Brute Isometry Losses"].dropna()) > 0
    ):
        print("Creating brute force vs two-stage comparison plot...")
        brute_losses = results_df["Brute Isometry Losses"].dropna()
        brute_plot_filename = os.path.join(
            output_dir, f"{dataset_name}_brute_vs_twostage_box.png"
        )
        box_plot(
            brute_losses,
            two_stage_losses,
            D,
            ["Brute Force", "Two-Stage"],
            r"$\ell_1(\hat{S}_{brute}) - \ell_1(\hat{S}_{TSIP})$",
            brute_plot_filename,
        )
        print(f"Saved plot: {brute_plot_filename}")
    else:
        print("Skipping brute force comparison - no brute force data available")

    # Plot 3: Loss distribution
    print("Creating loss distribution plot...")

    # Check if we have any valid data
    has_valid_data = False
    for col in ["Losses", "Two-Stage Losses", "Random Two-Stage Losses"]:
        if col in results_df.columns and len(results_df[col].dropna()) > 0:
            has_valid_data = True
            break

    if not has_valid_data:
        print("Skipping loss distribution plot - no valid data (all NaN values)")
        return

    plt.figure(figsize=(12, 8))

    if "Losses" in results_df.columns:
        losses = results_df["Losses"].dropna()
        if len(losses) > 0:
            plt.hist(losses, bins=20, alpha=0.7, label="Greedy Losses")
        else:
            print("Skipping Greedy Losses - all NaN values")

    if "Two-Stage Losses" in results_df.columns:
        two_stage_losses = results_df["Two-Stage Losses"].dropna()
        if len(two_stage_losses) > 0:
            plt.hist(two_stage_losses, bins=20, alpha=0.7, label="Two-Stage Losses")
        else:
            print("Skipping Two-Stage Losses - all NaN values")

    if "Random Two-Stage Losses" in results_df.columns:
        random_losses = results_df["Random Two-Stage Losses"].dropna()
        if len(random_losses) > 0:
            plt.hist(random_losses, bins=20, alpha=0.7, label="Random Two-Stage Losses")
        else:
            print("Skipping Random Two-Stage Losses - all NaN values")

    plt.xlabel("Loss Value")
    plt.ylabel("Frequency")
    plt.title(
        f"Loss Distribution - {dataset_name.upper()} Dataset (D={D}, power={power})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    loss_dist_filename = os.path.join(
        output_dir, f"{dataset_name}_loss_distribution.png"
    )
    plt.savefig(
        loss_dist_filename,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()
    print(f"Saved plot: {loss_dist_filename}")

    # Plot 4: Time comparison
    print("Creating time comparison plot...")
    time_columns = [
        col for col in results_df.columns if "Time" in col or "Times" in col
    ]
    if time_columns:
        plt.figure(figsize=(12, 8))

        for col in time_columns:
            plt.hist(
                results_df[col],
                bins=20,
                alpha=0.7,
                label=col.replace(" Times", "").replace(" Times", ""),
            )

        plt.xlabel("Time (seconds)")
        plt.ylabel("Frequency")
        plt.title(
            f"Time Distribution - {dataset_name.upper()} Dataset (D={D}, power={power})"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        time_dist_filename = os.path.join(
            output_dir, f"{dataset_name}_time_distribution.png"
        )
        plt.savefig(
            time_dist_filename,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()
        print(f"Saved plot: {time_dist_filename}")

    # Plot 5: Sparsity analysis
    if "Support Cardinalities (Basis Pursuit)" in results_df.columns:
        print("Creating sparsity analysis plot...")
        plt.figure(figsize=(12, 8))

        # Use the same purple color and styling as Figure2.ipynb example
        garfield_purple = "#4E2A84"
        cardinalities = results_df["Support Cardinalities (Basis Pursuit)"]

        # Use proper binning like in the notebook example
        min_val = min(cardinalities)
        max_val = max(cardinalities)
        bins = range(min_val, max_val + 2)

        plt.hist(
            cardinalities,
            bins=bins,
            color=garfield_purple,
            edgecolor="white",
            align="left",
        )
        plt.axvline(
            D,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Target Dimension (D={D})",
        )

        plt.xlabel(r"$|\widehat{S_{IP}}|$", fontsize=30)
        plt.ylabel("Frequency", fontsize=24)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.title(
            f"Basis Pursuit Sparsity - {dataset_name.upper()} Dataset (D={D}, power={power})"
        )
        plt.legend()
        sparsity_filename = os.path.join(
            output_dir, f"{dataset_name}_sparsity_analysis.png"
        )
        plt.savefig(
            sparsity_filename,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()
        print(f"Saved plot: {sparsity_filename}")

    print(f"Plots saved to {output_dir}")


def plot_convergence_analysis(
    convergence_file, dataset_name, D, power, output_dir="figures"
):
    """
    Create convergence analysis plots.

    Parameters:
    -----------
    convergence_file : str
        Path to convergence pickle file
    dataset_name : str
        Name of the dataset
    D : int
        Target dimension
    power : float
        Power parameter used
    output_dir : str
        Directory to save plots
    """

    print(f"Loading convergence data from {convergence_file}")

    with open(convergence_file, "rb") as f:
        convergence_logs = pickle.load(f)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Plot convergence analysis
    print("Creating convergence analysis plot...")
    convergence_filename = os.path.join(
        output_dir, f"{dataset_name}_convergence_analysis.png"
    )
    analyze_convergence_logs(
        convergence_logs,
        save_path=convergence_filename,
    )
    print(f"Saved plot: {convergence_filename}")


def main():
    parser = argparse.ArgumentParser(description="Plot experiment results")
    parser.add_argument(
        "--results-file", required=True, help="Path to results CSV file"
    )
    parser.add_argument("--dataset-name", required=True, help="Name of the dataset")
    parser.add_argument(
        "--D",
        type=int,
        help="Target dimension (auto-parsed from filename if not provided)",
    )
    parser.add_argument(
        "--power",
        type=float,
        help="Power parameter (auto-parsed from filename if not provided)",
    )
    parser.add_argument("--convergence-file", help="Path to convergence pickle file")
    parser.add_argument(
        "--output-dir", default="figures", help="Output directory for plots"
    )

    args = parser.parse_args()

    # Plot main results
    plot_experiment_results(
        results_file=args.results_file,
        dataset_name=args.dataset_name,
        D=args.D,
        power=args.power,
        output_dir=args.output_dir,
    )

    # Plot convergence analysis if available
    if args.convergence_file and os.path.exists(args.convergence_file):
        plot_convergence_analysis(
            convergence_file=args.convergence_file,
            dataset_name=args.dataset_name,
            D=args.D,
            power=args.power,
            output_dir=args.output_dir,
        )

    print("All plots created successfully!")


def plot_coefficient_convergence(convergence_data, save_path=None, max_features=None):
    """
    Plot coefficient norms over iterations for each feature.

    Parameters:
    -----------
    convergence_data : dict
        Output from create_convergence_log_with_coefficients() or load_convergence_data()
    save_path : str, optional
        Path to save the plot
    max_features : int, optional
        Maximum number of features to plot (for readability). If None, plots all features.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    coefficient_norms = convergence_data["coefficient_norms"]
    iterations = convergence_data["iterations"]
    sparsities = convergence_data["sparsities"]
    D, P = convergence_data["matrix_shape"]

    # Filter out NaN values and failed iterations (sparsities = -1)
    valid_mask = ~np.isnan(coefficient_norms).any(axis=1) & (sparsities >= 0)
    valid_iterations = np.array(iterations)[valid_mask]
    valid_norms = coefficient_norms[valid_mask]

    if len(valid_iterations) == 0:
        print("No valid convergence data to plot")
        return

    # Determine which features are non-zero at the final iteration
    final_norms = valid_norms[-1, :]

    # Use threshold 10e-5 for better visualization
    threshold = 10e-5
    non_zero_features = final_norms > threshold

    # Limit number of features for readability if requested
    if max_features is not None and P > max_features:
        # Select features with highest final norms
        top_features = np.argsort(final_norms)[-max_features:]
        valid_norms = valid_norms[:, top_features]
        non_zero_features = non_zero_features[top_features]
        feature_labels = [f"Feature {i}" for i in top_features]
    else:
        feature_labels = [f"Feature {i}" for i in range(P)]

    plt.figure(figsize=(12, 8))

    # Plot each feature with different colors based on final state
    for i in range(valid_norms.shape[1]):
        if non_zero_features[i]:
            # Non-zero features: black
            color = "black"
            alpha = 0.8
            linewidth = 2.0
        else:
            # Zero features: gray
            color = "gray"
            alpha = 0.4
            linewidth = 1.0

        plt.semilogx(
            valid_iterations,
            valid_norms[:, i],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )

    plt.xlabel("Iterations (log scale)", fontsize=14)
    plt.ylabel("Coefficient Norm ||β[i,:]||", fontsize=14)
    plt.title("Coefficient Norm Convergence", fontsize=16)
    plt.grid(True, alpha=0.3)

    # Add custom legend for color coding
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=2,
            alpha=0.8,
            label="Selected features (non-zero at final)",
        ),
        Line2D(
            [0],
            [0],
            color="gray",
            linewidth=1,
            alpha=0.4,
            label="Unselected features (zero at final)",
        ),
    ]
    plt.legend(handles=legend_elements, loc="upper right", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Coefficient convergence plot saved to {save_path}")
        plt.close()  # Close the figure to free memory
    else:
        plt.show()


def plot_coefficient_convergence_heatmap(convergence_data, save_path=None):
    """
    Plot coefficient norms as a heatmap with iterations on x-axis and features on y-axis.

    Parameters:
    -----------
    convergence_data : dict
        Output from create_convergence_log_with_coefficients() or load_convergence_data()
    save_path : str, optional
        Path to save the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np

    coefficient_norms = convergence_data["coefficient_norms"]
    iterations = convergence_data["iterations"]
    sparsities = convergence_data["sparsities"]
    D, P = convergence_data["matrix_shape"]

    # Filter out NaN values and failed iterations (sparsities = -1)
    valid_mask = ~np.isnan(coefficient_norms).any(axis=1) & (sparsities >= 0)
    valid_iterations = np.array(iterations)[valid_mask]
    valid_norms = coefficient_norms[valid_mask]

    if len(valid_iterations) == 0:
        print("No valid convergence data to plot")
        return

    plt.figure(figsize=(10, 8))

    # Create heatmap
    im = plt.imshow(
        valid_norms.T,
        aspect="auto",
        cmap="viridis",
        extent=[min(valid_iterations), max(valid_iterations), P - 0.5, -0.5],
    )

    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Feature Index", fontsize=14)
    plt.title("Coefficient Norm Heatmap", fontsize=16)
    plt.colorbar(im, label="Coefficient Norm ||β[i,:]||")

    # Set x-axis to log scale
    plt.xscale("log")

    # Set y-axis ticks
    if P <= 20:
        plt.yticks(range(P))
    else:
        # Show every 5th feature for readability
        plt.yticks(range(0, P, max(1, P // 20)))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Coefficient convergence heatmap saved to {save_path}")
        plt.close()  # Close the figure to free memory
    else:
        plt.show()


def plot_convergence_summary(convergence_data, save_path=None):
    """
    Plot a summary of convergence metrics including coefficient norms.

    Parameters:
    -----------
    convergence_data : dict
        Output from create_convergence_log_with_coefficients() or load_convergence_data()
    save_path : str, optional
        Path to save the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np

    coefficient_norms = convergence_data["coefficient_norms"]
    losses = convergence_data["losses"]
    sparsities = convergence_data["sparsities"]
    residuals = convergence_data["residuals"]
    iterations = convergence_data["iterations"]

    # Filter out NaN values and failed iterations (sparsities = -1)
    valid_mask = ~np.isnan(losses) & (sparsities >= 0)
    valid_iterations = np.array(iterations)[valid_mask]
    valid_losses = losses[valid_mask]
    valid_sparsities = sparsities[valid_mask]
    valid_residuals = residuals[valid_mask]
    valid_norms = coefficient_norms[valid_mask]

    # Recalculate sparsity using the same threshold as coefficient convergence plot
    threshold = 10e-5
    recalculated_sparsities = np.sum(valid_norms > threshold, axis=1)

    if len(valid_iterations) == 0:
        print("No valid convergence data to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Convergence Analysis Summary", fontsize=16)

    # Plot 1: Total loss over iterations
    axes[0, 0].semilogx(valid_iterations, valid_losses, "b-o", markersize=4)
    axes[0, 0].set_xlabel("Iterations")
    axes[0, 0].set_ylabel("Total Loss")
    axes[0, 0].set_title("Loss Convergence")
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Sparsity over iterations (recalculated with consistent threshold)
    axes[0, 1].semilogx(valid_iterations, recalculated_sparsities, "r-s", markersize=4)
    axes[0, 1].set_xlabel("Iterations")
    axes[0, 1].set_ylabel("Sparsity (Non-zero Rows)")
    axes[0, 1].set_title("Sparsity Evolution")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Residual over iterations
    axes[1, 0].semilogx(valid_iterations, valid_residuals, "g-^", markersize=4)
    axes[1, 0].set_xlabel("Iterations")
    axes[1, 0].set_ylabel("Constraint Residual")
    axes[1, 0].set_title("Constraint Satisfaction")
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Coefficient norms heatmap (subset for readability)
    max_features_show = min(20, valid_norms.shape[1])
    if max_features_show < valid_norms.shape[1]:
        # Select features with highest final norms
        final_norms = valid_norms[-1, :]
        top_features = np.argsort(final_norms)[-max_features_show:]
        norms_subset = valid_norms[:, top_features]
        feature_labels = [f"F{i}" for i in top_features]
    else:
        norms_subset = valid_norms
        feature_labels = [f"F{i}" for i in range(max_features_show)]

    im = axes[1, 1].imshow(
        norms_subset.T,
        aspect="auto",
        cmap="viridis",
        extent=[
            min(valid_iterations),
            max(valid_iterations),
            max_features_show - 0.5,
            -0.5,
        ],
    )
    axes[1, 1].set_xlabel("Iterations")
    axes[1, 1].set_ylabel("Feature Index")
    axes[1, 1].set_title(f"Coefficient Norms (Top {max_features_show} Features)")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yticks(range(max_features_show))
    axes[1, 1].set_yticklabels(feature_labels)
    plt.colorbar(im, ax=axes[1, 1], label="Coefficient Norm")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Convergence summary plot saved to {save_path}")
        plt.close()  # Close the figure to free memory
    else:
        plt.show()


def plot_sparsity_convergence_boxplot(convergence_logs, save_path=None):
    """
    Plot sparsity convergence as boxplots across replicates.

    Parameters:
    -----------
    convergence_logs : list
        List of convergence data dictionaries from multiple replicates
    save_path : str, optional
        Path to save the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if not convergence_logs or all(log is None for log in convergence_logs):
        print("No valid convergence logs to plot")
        return

    # Extract sparsity data from all replicates
    all_sparsities = []
    all_iterations = []
    threshold = 10e-5  # Use same threshold as other plots

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
                # Recalculate sparsity using consistent threshold
                recalculated_sparsities = np.sum(valid_norms > threshold, axis=1)
                all_sparsities.append(recalculated_sparsities)
                all_iterations.append(valid_iterations)

    if not all_sparsities:
        print("No valid sparsity data to plot")
        return

    # Find common iteration points
    all_unique_iterations = set()
    for iterations in all_iterations:
        all_unique_iterations.update(iterations)
    common_iterations = sorted(list(all_unique_iterations))

    # Organize data by iteration point
    iteration_data = {}
    for i, iterations in enumerate(all_iterations):
        sparsities = all_sparsities[i]
        for iter_val, sparsity in zip(iterations, sparsities):
            if iter_val not in iteration_data:
                iteration_data[iter_val] = []
            iteration_data[iter_val].append(sparsity)

    # Prepare data for boxplot
    boxplot_data = []
    boxplot_labels = []

    for iter_val in common_iterations:
        if iter_val in iteration_data and len(iteration_data[iter_val]) > 0:
            boxplot_data.append(iteration_data[iter_val])
            boxplot_labels.append(str(iter_val))

    if not boxplot_data:
        print("No data to plot")
        return

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Create boxplot
    bp = plt.boxplot(boxplot_data, labels=boxplot_labels, patch_artist=True)

    # Color the boxes
    for patch in bp["boxes"]:
        patch.set_facecolor("#4E2A84")  # Garfield purple
        patch.set_alpha(0.7)

    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Sparsity (Number of Non-zero Features)", fontsize=14)
    plt.title("Sparsity Convergence Across Replicates", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Sparsity convergence boxplot saved to {save_path}")
        plt.close()  # Close the figure to free memory
    else:
        plt.show()


if __name__ == "__main__":
    main()
