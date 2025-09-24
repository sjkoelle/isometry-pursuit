#!/usr/bin/env python3
"""
Plot difference boxplots across multiple datasets.
Creates a single plot with datasets on x-axis and method differences on y-axis.
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Add parent directory to path to import isometry_pursuit modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from isometry_pursuit.plot import box_plot


def plot_difference_boxplot(
    result_files,
    baseline_column,
    comparison_column,
    dataset_names=None,
    output_file="difference_boxplot.png",
    title="Method Comparison Across Datasets",
    ylabel=None,
):
    """
    Create a box plot comparing two methods across multiple datasets.

    Parameters:
    -----------
    result_files : list
        List of paths to results CSV files
    baseline_column : str
        Name of the baseline method column (e.g., "Losses" for greedy)
    comparison_column : str
        Name of the comparison method column (e.g., "Two-Stage Losses")
    dataset_names : list, optional
        Names for datasets (auto-parsed from filenames if not provided)
    output_file : str
        Output filename for the plot
    title : str
        Plot title
    ylabel : str, optional
        Y-axis label (auto-generated if not provided)
    """

    if len(result_files) < 2:
        raise ValueError("Need at least 2 result files to compare")

    # Parse dataset names from filenames if not provided
    if dataset_names is None:
        dataset_names = []
        for file_path in result_files:
            filename = os.path.basename(file_path)
            # Pattern: {dataset}_results.csv
            match = re.match(r"(.+)_results\.csv", filename)
            if match:
                dataset_names.append(match.group(1))
            else:
                # Fallback to filename without extension
                dataset_names.append(os.path.splitext(filename)[0])

    if len(dataset_names) != len(result_files):
        raise ValueError("Number of dataset names must match number of result files")

    # Load data and compute differences
    differences = []
    valid_datasets = []

    for i, (result_file, dataset_name) in enumerate(zip(result_files, dataset_names)):
        print(f"Loading {dataset_name} from {result_file}...")

        try:
            df = pd.read_csv(result_file)

            # Check if required columns exist
            if baseline_column not in df.columns:
                print(
                    f"Warning: {baseline_column} not found in {result_file}, skipping {dataset_name}"
                )
                continue
            if comparison_column not in df.columns:
                print(
                    f"Warning: {comparison_column} not found in {result_file}, skipping {dataset_name}"
                )
                continue

            # Get data and check for valid values
            baseline_data = df[baseline_column].dropna()
            comparison_data = df[comparison_column].dropna()

            if len(baseline_data) == 0 or len(comparison_data) == 0:
                print(
                    f"Warning: No valid data in {result_file}, skipping {dataset_name}"
                )
                continue

            # Compute difference (baseline - comparison)
            diff = np.array(baseline_data) - np.array(comparison_data)
            differences.append(diff)
            valid_datasets.append(dataset_name)

            print(
                f"  {dataset_name}: {len(diff)} samples, mean diff = {np.mean(diff):.3f}"
            )

        except Exception as e:
            print(f"Error loading {result_file}: {e}")
            continue

    if len(differences) == 0:
        raise ValueError("No valid data found in any result files")

    if len(differences) == 1:
        print("Warning: Only one dataset has valid data, creating single box plot")

    # Create the plot using the box_plot function
    if ylabel is None:
        # Use the same LaTeX formula as in plot_results.py
        ylabel = r"$\ell_1(\hat{S}_{g}) - \ell_1(\hat{S}_{TSIP})$"

    box_plot(
        data_list=differences,
        labels=valid_datasets,
        title=title,
        xlabel="Dataset",
        ylabel=ylabel,
        filename=output_file,
        figsize=(max(8, len(valid_datasets) * 2), 8),
        show_zero_line=True,
    )

    print(f"Saved plot: {output_file}")
    print(f"Compared {len(valid_datasets)} datasets: {', '.join(valid_datasets)}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot difference boxplots across multiple datasets"
    )
    parser.add_argument(
        "--result-files",
        nargs="+",
        required=True,
        help="List of result CSV files to compare",
    )
    parser.add_argument(
        "--baseline-column",
        default="loss",
        help="Name of baseline method column (default: 'loss')",
    )
    parser.add_argument(
        "--comparison-column",
        default="two_stage_loss",
        help="Name of comparison method column (default: 'two_stage_loss')",
    )
    parser.add_argument(
        "--dataset-names",
        nargs="*",
        help="Names for datasets (auto-parsed from filenames if not provided)",
    )
    parser.add_argument(
        "--output-file",
        default="difference_boxplot.png",
        help="Output filename for the plot",
    )
    parser.add_argument(
        "--title", default="Method Comparison Across Datasets", help="Plot title"
    )
    parser.add_argument(
        "--ylabel", help="Y-axis label (auto-generated if not provided)"
    )

    args = parser.parse_args()

    plot_difference_boxplot(
        result_files=args.result_files,
        baseline_column=args.baseline_column,
        comparison_column=args.comparison_column,
        dataset_names=args.dataset_names,
        output_file=args.output_file,
        title=args.title,
        ylabel=args.ylabel,
    )


if __name__ == "__main__":
    main()
