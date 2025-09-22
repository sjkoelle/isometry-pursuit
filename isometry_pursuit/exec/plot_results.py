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
from isometry_pursuit.experiment import plot_convergence, analyze_convergence_logs


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

        plt.hist(
            results_df["Support Cardinalities (Basis Pursuit)"], bins=20, alpha=0.7
        )
        plt.axvline(
            D,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Target Dimension (D={D})",
        )

        plt.xlabel("Support Cardinality")
        plt.ylabel("Frequency")
        plt.title(
            f"Basis Pursuit Sparsity - {dataset_name.upper()} Dataset (D={D}, power={power})"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
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


if __name__ == "__main__":
    main()
