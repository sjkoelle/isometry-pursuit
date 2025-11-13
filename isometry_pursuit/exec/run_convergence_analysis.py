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


def run_convergence_analysis_script(
    data_file,
    dataset_name,
    power=1.0,
    iteration_points=None,
    output_dir="results",
    n_replicates=10,
):
    """
    Run convergence analysis on a dataset with multiple replicates.

    Parameters:
    -----------
    data_file : str
        Path to the data CSV file
    dataset_name : str
        Name of the dataset
    power : float
        Power parameter for transformation
    iteration_points : list, optional
        List of iteration counts to test
    output_dir : str
        Directory to save results
    n_replicates : int
        Number of replicates to run
    """
    print(f"Loading data from {data_file}")

    # Load data tensor for all datasets
    if data_file.endswith(".npy"):
        print(f"Loading data tensor from {data_file}")
        data_tensor = np.load(data_file)
        print(f"Data tensor shape: {data_tensor.shape}")

        # Extract data for each replicate
        convergence_logs = []
        for i in range(min(n_replicates, data_tensor.shape[2])):
            print(f"Running replicate {i+1}/{n_replicates}...")
            X_replicate = data_tensor[:, :, i]
            print(f"  Replicate data shape: {X_replicate.shape}")

            convergence_data = create_convergence_log_with_coefficients(
                X_replicate, threshold=1e-6, iteration_points=iteration_points
            )
            convergence_logs.append(convergence_data)

            # Save individual replicate data
            convergence_file = os.path.join(
                output_dir, f"{dataset_name}_convergence_replicate_{i+1}.npz"
            )
            save_convergence_data(convergence_data, convergence_file)
    else:
        print(f"Error: Please provide a .npy tensor file")
        print(f"Expected file ending with .npy, got: {data_file}")
        return

    # Save all replicates data
    all_convergence_file = os.path.join(
        output_dir, f"{dataset_name}_convergence_all_replicates.pkl"
    )
    import pickle

    with open(all_convergence_file, "wb") as f:
        pickle.dump(convergence_logs, f)

    print(f"Convergence analysis completed!")
    print(
        f"Individual replicate data saved to: {output_dir}/{dataset_name}_convergence_replicate_*.npz"
    )
    print(f"All replicates data saved to: {all_convergence_file}")
    print(f"Use plot_convergence_analysis.py to generate plots from this data")


def main():
    parser = argparse.ArgumentParser(description="Run convergence analysis")
    parser.add_argument(
        "--data-file",
        required=True,
        help="Path to data tensor file (.npy)",
    )
    parser.add_argument("--dataset-name", required=True, help="Name of the dataset")
    parser.add_argument("--power", type=float, default=1.0, help="Power parameter")
    parser.add_argument(
        "--iteration-points",
        nargs="+",
        type=int,
        default=[1, 10, 100, 1000, 10000],
        help="Iteration points to test",
    )
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument(
        "--n-replicates", type=int, default=10, help="Number of replicates to run"
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Run convergence analysis
    run_convergence_analysis_script(
        data_file=args.data_file,
        dataset_name=args.dataset_name,
        power=args.power,
        iteration_points=args.iteration_points,
        output_dir=args.output_dir,
        n_replicates=args.n_replicates,
    )


if __name__ == "__main__":
    main()
