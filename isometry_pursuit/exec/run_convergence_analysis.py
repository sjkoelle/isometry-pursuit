#!/usr/bin/env python3
"""
Run convergence analysis on a dataset.
This script runs convergence logging independently of the main experiments.
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path to import isometry_pursuit modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from isometry_pursuit.algorithm import (
    create_convergence_log_with_coefficients,
    save_convergence_data,
    load_convergence_data,
)
from isometry_pursuit.exec.plot_results import (
    plot_coefficient_convergence,
    plot_coefficient_convergence_heatmap,
    plot_convergence_summary,
)


def run_convergence_analysis(
    data_file,
    dataset_name,
    D,
    power=1.0,
    iteration_points=None,
    output_dir="results",
):
    """
    Run convergence analysis on a dataset.

    Parameters:
    -----------
    data_file : str
        Path to the data CSV file
    dataset_name : str
        Name of the dataset
    D : int
        Target dimension
    power : float
        Power parameter for transformation
    iteration_points : list, optional
        List of iteration counts to test
    output_dir : str
        Directory to save results
    """
    print(f"Loading data from {data_file}")
    data = pd.read_csv(data_file)

    # Convert to numpy array and transpose to get (D, P) format
    X = data.values.T

    print(f"Running convergence analysis on {dataset_name} dataset...")
    print(f"Data shape: {X.shape}")
    print(f"Parameters: D={D}, power={power}")

    # Run convergence analysis
    convergence_data = create_convergence_log_with_coefficients(
        X, threshold=1e-6, iteration_points=iteration_points
    )

    # Save convergence data
    convergence_file = os.path.join(
        output_dir, f"{dataset_name}_convergence_analysis.npz"
    )
    save_convergence_data(convergence_data, convergence_file)

    # Generate plots
    print("Generating convergence plots...")

    # Coefficient convergence plot
    plot_coefficient_convergence(
        convergence_data,
        save_path=os.path.join(
            output_dir, f"{dataset_name}_coefficient_convergence.png"
        ),
        max_features=20,
    )

    # Heatmap plot
    plot_coefficient_convergence_heatmap(
        convergence_data,
        save_path=os.path.join(output_dir, f"{dataset_name}_convergence_heatmap.png"),
    )

    # Summary plot
    plot_convergence_summary(
        convergence_data,
        save_path=os.path.join(output_dir, f"{dataset_name}_convergence_summary.png"),
    )

    print(f"Convergence analysis completed!")
    print(f"Data saved to: {convergence_file}")
    print(f"Plots saved to: {output_dir}/{dataset_name}_convergence_*.png")


def main():
    parser = argparse.ArgumentParser(description="Run convergence analysis")
    parser.add_argument("--data-file", required=True, help="Path to data file (CSV)")
    parser.add_argument("--dataset-name", required=True, help="Name of the dataset")
    parser.add_argument("--D", type=int, required=True, help="Target dimension")
    parser.add_argument("--power", type=float, default=1.0, help="Power parameter")
    parser.add_argument(
        "--iteration-points",
        nargs="+",
        type=int,
        default=[1, 10, 100, 1000, 10000],
        help="Iteration points to test",
    )
    parser.add_argument("--output-dir", default="results", help="Output directory")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Run convergence analysis
    run_convergence_analysis(
        data_file=args.data_file,
        dataset_name=args.dataset_name,
        D=args.D,
        power=args.power,
        iteration_points=args.iteration_points,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
