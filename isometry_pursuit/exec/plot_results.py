#!/usr/bin/env python3
"""
Plotting module for isometry pursuit experiment results.
Creates various plots from experiment results.
"""

import argparse
import os
import pickle
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from isometry_pursuit.plot import difference_box_plot, ecdf_plot
from isometry_pursuit.exec.plot_convergence import (
    plot_pickle_results,
    plot_convergence_analysis,
)


def plot_experiment_results(
    results_file, dataset_name, power=None, output_dir="figures"
):
    """
    Create plots from experiment results.

    Parameters:
    -----------
    results_file : str
        Path to results CSV file
    dataset_name : str
        Name of the dataset
    power : float
        Power parameter used
    output_dir : str
        Directory to save plots
    """

    print(f"Loading results from {results_file}")
    results_df = pd.read_csv(results_file)

    # Parse power from filename if not provided
    if power is None:
        import re

        filename = os.path.basename(results_file)
        # Pattern: {dataset}_results.csv
        match = re.match(r"(.+)_results\.csv", filename)
        if match:
            dataset_name_from_file = match.group(1)
            print(f"Parsed dataset name from filename: {dataset_name_from_file}")
            # Set default power if not provided
            power = 1.0  # Default power
            print(f"Using default power: {power}")
        else:
            raise ValueError(
                f"Could not parse dataset name from filename {filename}, and power was not provided as argument"
            )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Main comparison (Greedy vs Two-Stage)
    print("Creating main loss comparison plot (Greedy vs Two-Stage)...")
    if "loss" not in results_df.columns:
        raise ValueError("Missing required column: 'loss' (Greedy method)")
    if "two_stage_loss" not in results_df.columns:
        raise ValueError("Missing required column: 'two_stage_loss'")

    # Filter to only include replicates where both methods have valid results
    valid_mask = ~(results_df["loss"].isna() | results_df["two_stage_loss"].isna())

    if valid_mask.sum() == 0:
        raise ValueError(
            f"No valid data found where both greedy and two-stage methods have results for dataset {dataset_name}."
        )

    greedy_losses = results_df.loc[valid_mask, "loss"]
    two_stage_losses = results_df.loc[valid_mask, "two_stage_loss"]

    print(
        f"Using {len(greedy_losses)} replicates where both methods have valid results"
    )

    plot_filename = os.path.join(
        output_dir, f"{dataset_name}_greedy_vs_twostage_box.png"
    )
    difference_box_plot(
        greedy_losses,
        two_stage_losses,
        "Difference",
        "Greedy - Two-Stage",
        r"$\ell_1(\hat{S}_{g}) - \ell_1(\hat{S}_{TSIP})$",
        plot_filename,
    )
    print(f"Saved plot: {plot_filename}")

    # Plot 2: Optional brute force comparison (only if data exists)
    if "brute_gl_loss" in results_df.columns:
        # Filter to only include replicates where both brute force and two-stage methods have valid results
        brute_valid_mask = ~(
            results_df["brute_gl_loss"].isna()
            | results_df["two_stage_loss"].isna()
        )

        if brute_valid_mask.sum() > 0:
            print("Creating brute force vs two-stage comparison plot...")
            brute_losses = results_df.loc[brute_valid_mask, "brute_gl_loss"]
            brute_two_stage_losses = results_df.loc[brute_valid_mask, "two_stage_loss"]
            print(
                f"Using {len(brute_losses)} replicates where both brute force and two-stage methods have valid results"
            )

            brute_plot_filename = os.path.join(
                output_dir, f"{dataset_name}_brute_vs_twostage_box.png"
            )
            difference_box_plot(
                brute_losses,
                brute_two_stage_losses,
                "Difference",
                "Brute Force - Two-Stage",
                r"$\ell_1(\hat{S}_{brute}) - \ell_1(\hat{S}_{TSIP})$",
                brute_plot_filename,
            )
            print(f"Saved plot: {brute_plot_filename}")
        else:
            print(
                "Skipping brute force comparison - no valid data where both methods have results"
            )
    else:
        print("Skipping brute force comparison - no brute force data available")

    # Plot 3: Two-stage greedy vs Greedy comparison
    if "two_stage_greedy_loss" in results_df.columns:
        # Filter to only include replicates where both two-stage greedy and regular greedy methods have valid results
        two_stage_greedy_valid_mask = ~(
            results_df["two_stage_greedy_loss"].isna() | results_df["loss"].isna()
        )

        if two_stage_greedy_valid_mask.sum() > 0:
            print("Creating two-stage greedy vs greedy comparison plot...")
            two_stage_greedy_losses = results_df.loc[
                two_stage_greedy_valid_mask, "two_stage_greedy_loss"
            ]
            greedy_losses_for_comparison = results_df.loc[
                two_stage_greedy_valid_mask, "loss"
            ]
            print(
                f"Using {len(two_stage_greedy_losses)} replicates where both two-stage greedy and greedy methods have valid results"
            )

            two_stage_greedy_plot_filename = os.path.join(
                output_dir, f"{dataset_name}_twostage_greedy_vs_greedy_box.png"
            )
            difference_box_plot(
                two_stage_greedy_losses,
                greedy_losses_for_comparison,
                "Difference",
                "Two-Stage Greedy - Greedy",
                r"$\ell_1(\hat{S}_{TSG}) - \ell_1(\hat{S}_{G})$",
                two_stage_greedy_plot_filename,
            )
            print(f"Saved plot: {two_stage_greedy_plot_filename}")
        else:
            print(
                "Skipping two-stage greedy comparison - no valid data where both methods have results"
            )
    else:
        print(
            "Skipping two-stage greedy comparison - no two-stage greedy data available"
        )

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
        plt.title(f"Time Distribution - {dataset_name.upper()} Dataset (power={power})")
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

    # Plot 4: Sparsity analysis
    if "support_cardinality_basis_pursuit" in results_df.columns:
        print("Creating sparsity analysis plot...")
        plt.figure(figsize=(12, 8))

        # Use the same purple color and styling as Figure2.ipynb example
        garfield_purple = "#4E2A84"
        cardinalities = results_df["support_cardinality_basis_pursuit"]

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

        plt.xlabel(r"$|\widehat{S_{IP}}|$", fontsize=30)
        plt.ylabel("Frequency", fontsize=24)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.title(
            f"Basis Pursuit Sparsity - {dataset_name.upper()} Dataset (power={power})"
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


def main():
    parser = argparse.ArgumentParser(description="Plot experiment results")
    parser.add_argument(
        "--results-file", help="Path to results CSV file or pickle file"
    )
    parser.add_argument(
        "--dataset-name", help="Name of the dataset (auto-parsed if not provided)"
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

    # Determine file type and route accordingly
    if args.results_file:
        file_extension = os.path.splitext(args.results_file)[1].lower()

        if file_extension == ".pkl":
            # Handle pickle files
            print("Detected pickle file, using pickle plotting function...")

            # Auto-parse dataset name if not provided
            dataset_name = args.dataset_name
            if not dataset_name:
                import re

                filename = os.path.basename(args.results_file)
                # Pattern: {dataset}_convergence_all_replicates.pkl
                match = re.match(r"(.+)_convergence_all_replicates\.pkl", filename)
                if match:
                    dataset_name = match.group(1)
                    print(f"Auto-parsed dataset name: {dataset_name}")
                else:
                    # Fallback: use filename without extension
                    dataset_name = os.path.splitext(filename)[0]
                    print(f"Using filename as dataset name: {dataset_name}")

            plot_pickle_results(
                pickle_file=args.results_file,
                dataset_name=dataset_name,
                power=args.power,
                output_dir=args.output_dir,
            )

        elif file_extension == ".csv":
            # Handle CSV files
            print("Detected CSV file, using CSV plotting function...")

            if not args.dataset_name:
                raise ValueError("Dataset name is required for CSV files")

            plot_experiment_results(
                results_file=args.results_file,
                dataset_name=args.dataset_name,
                power=args.power,
                output_dir=args.output_dir,
            )
        else:
            raise ValueError(
                f"Unsupported file type: {file_extension}. Supported types: .csv, .pkl"
            )
    else:
        raise ValueError("No results file provided")

    # Plot convergence analysis if available (for CSV files)
    if args.convergence_file and os.path.exists(args.convergence_file):
        if not args.dataset_name:
            raise ValueError("Dataset name is required for convergence analysis")
        plot_convergence_analysis(
            convergence_file=args.convergence_file,
            dataset_name=args.dataset_name,
            power=args.power,
            output_dir=args.output_dir,
        )

    print("All plots created successfully!")


if __name__ == "__main__":
    main()
