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
    plot_sparsity_convergence_boxplot,
)


def run_convergence_analysis_script(
    data_file,
    dataset_name,
    D,
    power=1.0,
    iteration_points=None,
    output_dir="results",
    n_replicates=10,
    frac=0.5,
):
    """
    Run convergence analysis on a dataset with multiple replicates.

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
    n_replicates : int
        Number of replicates to run
    frac : float
        Fraction of data to use in each resampling (for non-ethanol datasets)
    """
    print(f"Loading data from {data_file}")

    # For non-ethanol datasets, load the resampled data tensor directly
    if dataset_name != "ethanol":
        if data_file.endswith(".npy"):
            print(f"Loading resampled data tensor from {data_file}")
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
            print(f"Error: For non-ethanol datasets, please provide a .npy tensor file")
            print(f"Expected file ending with .npy, got: {data_file}")
            return
    else:
        # For ethanol, load CSV and use individual data points
        data = pd.read_csv(data_file)
        X = data.values.T
        print(f"Running convergence analysis on {dataset_name} dataset...")
        print(f"Data shape: {X.shape}")

        convergence_logs = []
        for i in range(n_replicates):
            print(f"Running replicate {i+1}/{n_replicates}...")
            # For ethanol, each replicate is a different data point
            sample_data = data.iloc[i % len(data)].values
            P = len(sample_data) // D
            X_replicate = sample_data.reshape(D, P)
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

    # Save all replicates data
    all_convergence_file = os.path.join(
        output_dir, f"{dataset_name}_convergence_all_replicates.pkl"
    )
    import pickle

    with open(all_convergence_file, "wb") as f:
        pickle.dump(convergence_logs, f)

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
        max_features=20,
    )

    # Summary plot
    plot_convergence_summary(
        first_replicate,
        save_path=os.path.join(output_dir, f"{dataset_name}_convergence_summary.png"),
    )

    print(f"Convergence analysis completed!")
    print(
        f"Individual replicate data saved to: {output_dir}/{dataset_name}_convergence_replicate_*.npz"
    )
    print(f"All replicates data saved to: {all_convergence_file}")
    print(f"Plots saved to: {output_dir}/{dataset_name}_convergence_*.png")


def main():
    parser = argparse.ArgumentParser(description="Run convergence analysis")
    parser.add_argument(
        "--data-file",
        required=True,
        help="Path to data file (CSV for ethanol, .npy tensor for others)",
    )
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
    parser.add_argument(
        "--n-replicates", type=int, default=10, help="Number of replicates to run"
    )
    parser.add_argument(
        "--frac",
        type=float,
        default=0.5,
        help="Fraction of data to use in each resampling",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Run convergence analysis
    run_convergence_analysis_script(
        data_file=args.data_file,
        dataset_name=args.dataset_name,
        D=args.D,
        power=args.power,
        iteration_points=args.iteration_points,
        output_dir=args.output_dir,
        n_replicates=args.n_replicates,
        frac=args.frac,
    )


if __name__ == "__main__":
    main()
