#!/usr/bin/env python3
"""
Plot convergence analysis results from saved data files.
This script generates plots from convergence data saved by run_convergence_analysis.py.
"""

import argparse
import sys
import os
import pickle

# Add parent directory to path to import isometry_pursuit modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from isometry_pursuit.exec.plot_results import (
    plot_coefficient_convergence,
    plot_sparsity_convergence_boxplot,
)


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
