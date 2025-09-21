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


def plot_experiment_results(results_file, dataset_name, D, power, output_dir="figures"):
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

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Isometry loss comparison (Brute vs Two-Stage)
    if (
        "Brute Isometry Losses" in results_df.columns
        and "Two-Stage Losses" in results_df.columns
    ):
        # Check if we have valid data (not all NaN)
        brute_losses = results_df["Brute Isometry Losses"].dropna()
        two_stage_losses = results_df["Two-Stage Losses"].dropna()

        if len(brute_losses) > 0 and len(two_stage_losses) > 0:
            print("Creating isometry loss comparison plot...")
            box_plot(
                brute_losses,
                two_stage_losses,
                D,
                ["Brute Isometry", "Two-Stage"],
                r"$\ell_1$ Loss",
                os.path.join(output_dir, f"{dataset_name}_isometry_losses_iso_box.png"),
            )
        else:
            print("Skipping isometry loss comparison plot - insufficient data")

    # Plot 2: Multitask norm comparison (Brute vs Two-Stage)
    if (
        "Brute Losses" in results_df.columns
        and "Greedy Multitask Norms (Two-Stage)" in results_df.columns
    ):
        # Check if we have valid data (not all NaN)
        brute_losses = results_df["Brute Losses"].dropna()
        greedy_multitask_norms = results_df[
            "Greedy Multitask Norms (Two-Stage)"
        ].dropna()

        if len(brute_losses) > 0 and len(greedy_multitask_norms) > 0:
            print("Creating multitask norm comparison plot...")
            box_plot(
                brute_losses,
                greedy_multitask_norms,
                D,
                ["Brute", "Two-Stage"],
                r"$\|w(\cdot)\|_{1,2}$ Loss",
                os.path.join(output_dir, f"{dataset_name}_isometry_losses_ts_box.png"),
            )
        else:
            print("Skipping multitask norm comparison plot - insufficient data")

    # Plot 3: Loss distribution
    print("Creating loss distribution plot...")
    plt.figure(figsize=(12, 8))

    if "Losses" in results_df.columns:
        plt.hist(results_df["Losses"], bins=20, alpha=0.7, label="Greedy Losses")

    if "Two-Stage Losses" in results_df.columns:
        plt.hist(
            results_df["Two-Stage Losses"], bins=20, alpha=0.7, label="Two-Stage Losses"
        )

    if "Random Two-Stage Losses" in results_df.columns:
        plt.hist(
            results_df["Random Two-Stage Losses"],
            bins=20,
            alpha=0.7,
            label="Random Two-Stage Losses",
        )

    plt.xlabel("Loss Value")
    plt.ylabel("Frequency")
    plt.title(
        f"Loss Distribution - {dataset_name.upper()} Dataset (D={D}, power={power})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(output_dir, f"{dataset_name}_loss_distribution.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

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
        plt.savefig(
            os.path.join(output_dir, f"{dataset_name}_time_distribution.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

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
        plt.savefig(
            os.path.join(output_dir, f"{dataset_name}_sparsity_analysis.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

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
    analyze_convergence_logs(
        convergence_logs,
        save_path=os.path.join(output_dir, f"{dataset_name}_convergence_analysis.png"),
    )


def main():
    parser = argparse.ArgumentParser(description="Plot experiment results")
    parser.add_argument(
        "--results-file", required=True, help="Path to results CSV file"
    )
    parser.add_argument("--dataset-name", required=True, help="Name of the dataset")
    parser.add_argument("--D", type=int, required=True, help="Target dimension")
    parser.add_argument("--power", type=float, required=True, help="Power parameter")
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
