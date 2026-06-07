#!/usr/bin/env python3
"""
Plot convergence analysis results from saved data files.
This script generates plots from convergence data saved by run_convergence_analysis.py.
"""

import argparse
import os

from isometry_pursuit.exec.plot_convergence import (
    plot_coefficient_convergence,
    plot_sparsity_convergence_boxplot,
    plot_support_cardinality_histogram,
    plot_convergence_analysis_script,
)


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
