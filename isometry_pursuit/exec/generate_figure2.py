#!/usr/bin/env python3
"""
Generate Figure 2: Box plots comparing different methods
This script creates box plots comparing brute force vs two-stage methods.
"""

import argparse
import sys
import os
import pandas as pd
import json

# Add parent directory to path to import isometry_pursuit modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from isometry_pursuit.plot import box_plot


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
            # Skip generating wine_standardized_isometry_losses_box.png as requested
            if not (dataset_name == "wine" and "standardized" in filename):
                print("Creating isometry loss comparison plot...")
                box_plot(
                    results_df["Brute Isometry Losses"],
                    results_df["Two-Stage Losses"],
                    D,
                    ["Brute Isometry", "Two-Stage"],
                    r"$\ell_1$ Loss",
                    os.path.join(
                        output_dir, f"{dataset_name}_isometry_losses_iso_box.png"
                    ),
                )
            else:
                print("Skipping wine standardized isometry loss box plot as requested")

        if (
            "Brute Losses" in results_df.columns
            and "Greedy Multitask Norms (Two-Stage)" in results_df.columns
        ):
            print("Creating multitask norm comparison plot...")
            box_plot(
                results_df["Brute Losses"],
                results_df["Greedy Multitask Norms (Two-Stage)"],
                D,
                ["Brute", "Two-Stage"],
                r"$\|w(\cdot)\|_{1,2}$ Loss",
                os.path.join(output_dir, f"{dataset_name}_isometry_losses_ts_box.png"),
            )

    print("Figure 2 generation completed!")


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
