#!/usr/bin/env python3
"""
Generate Figure 2: Box plots comparing different methods
This script creates box plots comparing brute force vs two-stage methods.
"""

import argparse
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np

from isometry_pursuit.plot import difference_box_plot


def generate_cardinality_plots(results_dir="results", output_dir="figures"):
    """
    Generate cardinality plots for all datasets.

    Parameters:
    -----------
    results_dir : str
        Directory containing the results CSV files
    output_dir : str
        Directory to save the plots
    """

    print("Generating cardinality plots...")

    # Color for the plots (Garfield purple)
    garfield_purple = "#8B008B"

    # Find all results files
    results_files = []
    for file in os.listdir(results_dir):
        if file.endswith("_results.csv"):
            results_files.append(os.path.join(results_dir, file))

    if not results_files:
        print(f"No results files found in {results_dir}")
        return

    # Process each results file
    for results_file in results_files:
        print(f"Processing cardinalities for {results_file}...")

        # Load results
        results_df = pd.read_csv(results_file)

        # Extract dataset name from filename
        filename = os.path.basename(results_file)
        dataset_name = filename.split("_")[0]

        # Check if we have the required column
        if "Support Cardinalities (Basis Pursuit)" not in results_df.columns:
            print(f"Warning: No cardinality data found in {results_file}")
            continue

        # Create histogram
        plt.figure(figsize=(10, 7))

        cardinalities = results_df["Support Cardinalities (Basis Pursuit)"]

        if dataset_name == "ethanol":
            # Special handling for ethanol with 25-wide bins
            min_val = min(cardinalities)
            max_val = max(cardinalities)
            bins = range(min_val, max_val + 26, 25)
            plt.hist(cardinalities, bins=bins, color=garfield_purple, edgecolor="white")
        else:
            # Standard handling for other datasets
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

        # Formatting
        plt.xlabel(r"$|\widehat{S_{IP}}|$", fontsize=30)
        plt.ylabel("Frequency", fontsize=24)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        # Save plot
        output_file = os.path.join(output_dir, f"{dataset_name}_cardinalities.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Cardinality plot saved to {output_file}")

    print("Cardinality plots generation completed!")


def generate_figure2(results_dir="results", output_dir="figures"):
    """
    Generate Figure 2 box plots from experiment results.

    Parameters:
    -----------
    results_dir : str
        Directory containing experiment results
    output_dir : str
        Directory to save the plots
    """

    print("Generating Figure 2: Box plots comparing methods")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Look for results files
    results_files = []
    for file in os.listdir(results_dir):
        if file.endswith("_results.csv"):
            results_files.append(os.path.join(results_dir, file))

    if not results_files:
        print(f"No results files found in {results_dir}")
        return

    print(f"Found {len(results_files)} results files")

    for results_file in results_files:
        print(f"Processing {results_file}")

        # Load results
        results_df = pd.read_csv(results_file)

        # Extract dataset info from filename
        filename = os.path.basename(results_file)
        parts = filename.replace("_results.csv", "").split("_")

        # Handle different filename formats
        if "word_embeddings" in filename:
            # Format: word_embeddings_standardized_1.0_10_results.csv
            # Parts: ['word', 'embeddings', 'standardized', '1.0', '10']
            dataset_name = "word_embeddings"
            power = float(parts[3])
            D = int(parts[4])
        else:
            # Format: {dataset}_standardized_{power}_{D}_results.csv
            # Parts: ['wine', 'standardized', '1.0', '6']
            dataset_name = parts[0]
            power = float(parts[2])
            D = int(parts[3])

        print(f"Dataset: {dataset_name}, Power: {power}, D: {D}")

        # Create plots if the required columns exist
        if (
            "Brute Isometry Losses" in results_df.columns
            and "Two-Stage Losses" in results_df.columns
        ):
            # Use difference box plot for wine standardized (replaces ECDF plot)
            if dataset_name == "wine" and "standardized" in filename:
                print(
                    "Creating wine standardized difference box plot (replaces ECDF)..."
                )
                difference_box_plot(
                    results_df["Brute Isometry Losses"],
                    results_df["Two-Stage Losses"],
                    "Wine Standardized",
                    r"$\ell_1(\hat{S}_{g}) - \ell_1(\hat{S}_{TSIP})$",
                    "Difference in Loss",
                    os.path.join(
                        output_dir,
                        f"{dataset_name}_standardized_isometry_losses_ecdf.png",
                    ),
                )
            else:
                print("Creating isometry loss comparison plot...")
                difference_box_plot(
                    results_df["Brute Isometry Losses"],
                    results_df["Two-Stage Losses"],
                    "Difference",
                    "Brute Isometry - Two-Stage",
                    r"$\ell_1$ Loss",
                    os.path.join(
                        output_dir, f"{dataset_name}_isometry_losses_iso_box.png"
                    ),
                )

        if (
            "Brute Losses" in results_df.columns
            and "Greedy Multitask Norms (Two-Stage)" in results_df.columns
        ):
            print("Creating multitask norm comparison plot...")
            difference_box_plot(
                results_df["Brute Losses"],
                results_df["Greedy Multitask Norms (Two-Stage)"],
                "Difference",
                "Brute - Two-Stage",
                r"$\|w(\cdot)\|_{1,2}$ Loss",
                os.path.join(output_dir, f"{dataset_name}_isometry_losses_ts_box.png"),
            )

    print("Figure 2 generation completed!")

    # Generate cardinality plots
    generate_cardinality_plots(results_dir, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate Figure 2 box plots")
    parser.add_argument(
        "--results-dir", default="results", help="Directory containing results files"
    )
    parser.add_argument(
        "--output-dir", default="figures", help="Output directory for plots"
    )

    args = parser.parse_args()

    generate_figure2(results_dir=args.results_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
