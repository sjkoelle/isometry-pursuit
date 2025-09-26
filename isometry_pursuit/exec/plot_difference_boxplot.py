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
    cardinality_column="Support Cardinalities (Basis Pursuit)",
):
    """
    Create a box plot comparing two methods across multiple datasets.

    Parameters:
    -----------
    result_files : list
        List of paths to results CSV files
    baseline_column : str
        Name of the baseline method column (e.g., "loss" for greedy)
    comparison_column : str
        Name of the comparison method column (e.g., "two_stage_loss")
    dataset_names : list, optional
        Names for datasets (auto-parsed from filenames if not provided)
    output_file : str
        Output filename for the plot
    title : str
        Plot title
    ylabel : str, optional
        Y-axis label (auto-generated if not provided)
    cardinality_column : str, optional
        Name of the cardinality column in CSV files (default: "Support Cardinalities (Basis Pursuit)")
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

    # Log detailed statistics for each dataset
    print("\n" + "="*60)
    print("DETAILED STATISTICS SUMMARY")
    print("="*60)
    
    from scipy.stats import ttest_rel
    
    # Helper for 2 significant figures
    def sigfig(x, n=2):
        return f"{x:.{n}g}"
    
    for i, (dataset_name, diff) in enumerate(zip(valid_datasets, differences)):
        # Reload the data to get original values for t-test
        result_file = result_files[i]
        df = pd.read_csv(result_file)
        baseline_data = df[baseline_column].dropna()
        comparison_data = df[comparison_column].dropna()
        
        # Calculate statistics
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        n_samples = len(diff)
        
        # Calculate probability estimates
        prob_greater = np.mean(diff > 0)  # P(baseline > comparison)
        prob_equal = np.mean(diff == 0)  # P(baseline = comparison)

        # Calculate support cardinality statistics from CSV data
        mean_cardinality = std_cardinality = np.nan
        
        # Check if cardinality column exists in the CSV
        if cardinality_column in df.columns:
            cardinality_data = df[cardinality_column].dropna()
            if len(cardinality_data) > 0:
                mean_cardinality = np.mean(cardinality_data)
                std_cardinality = np.std(cardinality_data)
            else:
                print(f"  Warning: No valid cardinality data found in {cardinality_column}")
        else:
            print(f"  Warning: {cardinality_column} column not found in {result_file}")

        # Calculate paired t-test p-value
        try:
            t_stat, p_value = ttest_rel(baseline_data, comparison_data)
        except Exception as e:
            print(f"Warning: Could not compute t-test for {dataset_name}: {e}")
            p_value = np.nan

        # Log statistics in table format
        print(f"\n{dataset_name}:")
        print(f"  Sample size (R): {n_samples}")
        print(
            f"  {baseline_column}: {sigfig(np.mean(baseline_data))} ± {sigfig(np.std(baseline_data))}"
        )
        print(
            f"  {comparison_column}: {sigfig(np.mean(comparison_data))} ± {sigfig(np.std(comparison_data))}"
        )
        if not np.isnan(mean_cardinality):
            print(
                f"  Support Cardinality (|Ŝ|): {sigfig(mean_cardinality)} ± {sigfig(std_cardinality)}"
            )
        print(f"  Mean difference: {sigfig(mean_diff)} ± {sigfig(std_diff)}")
        print(f"  P({baseline_column} > {comparison_column}): {sigfig(prob_greater)}")
        print(f"  P({baseline_column} = {comparison_column}): {sigfig(prob_equal)}")
        print(f"  Paired t-test p-value: {p_value:.1e}")

    print("=" * 60)

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
        figsize=(12, 7),  # Match the sparsity convergence plot size
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
        help="List of results CSV files to compare",
    )
    parser.add_argument(
        "--baseline-column",
        required=True,
        help="Name of the baseline method column (e.g., 'loss' for greedy)",
    )
    parser.add_argument(
        "--comparison-column",
        required=True,
        help="Name of the comparison method column (e.g., 'two_stage_loss')",
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
    parser.add_argument(
        "--cardinality-column",
        default="Support Cardinalities (Basis Pursuit)",
        help="Name of the cardinality column in CSV files",
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
        cardinality_column=args.cardinality_column,
    )


if __name__ == "__main__":
    main()
