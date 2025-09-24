#!/usr/bin/env python3
"""
Unified experiment runner for isometry pursuit experiments.
This script runs resampling experiments with configurable parameters.
"""

import argparse
import sys
import os
import pickle
import json
from pathlib import Path

# Add parent directory to path to import isometry_pursuit modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from isometry_pursuit.experiment import run_tensor_experiment


def main():
    parser = argparse.ArgumentParser(description="Run isometry pursuit experiment")
    parser.add_argument(
        "--data-file",
        required=True,
        help="Path to data tensor file (.npy)",
    )
    parser.add_argument("--dataset-name", required=True, help="Name of the dataset")
    parser.add_argument(
        "--D",
        type=int,
        help="Target dimension (optional, will be extracted from tensor if not provided)",
    )
    parser.add_argument("--power", type=float, default=1.0, help="Power parameter")
    parser.add_argument(
        "--compute-brute", action="store_true", help="Compute brute force solutions"
    )
    parser.add_argument(
        "--log-convergence", action="store_true", help="Log convergence information"
    )
    parser.add_argument("--output-dir", default="results", help="Output directory")

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data_file}")

    # Load data tensor for all datasets
    if args.data_file.endswith(".npy"):
        print(f"Loading data tensor from {args.data_file}")
        data_tensor = np.load(args.data_file)
        print(f"Data tensor shape: {data_tensor.shape}")

        # Validate tensor dimension if D is provided
        if args.D is not None:
            if data_tensor.shape[0] != args.D:
                print(
                    f"Error: Tensor dimension {data_tensor.shape[0]} does not match provided D={args.D}"
                )
                return
            print(f"Tensor dimension validated: D={args.D}")
        else:
            print(f"Using tensor dimension: D={data_tensor.shape[0]}")

        # Run experiment using tensor
        if args.log_convergence:
            (
                results_df,
                co_occurrence_matrix,
                greedy_co_occurrence_matrix,
                convergence_logs,
            ) = run_tensor_experiment(
                data_tensor=data_tensor,
                compute_brute=args.compute_brute,
                power=args.power,
                log_convergence=args.log_convergence,
            )
        else:
            results_df, co_occurrence_matrix, greedy_co_occurrence_matrix = (
                run_tensor_experiment(
                    data_tensor=data_tensor,
                    compute_brute=args.compute_brute,
                    power=args.power,
                    log_convergence=args.log_convergence,
                )
            )
            convergence_logs = None
    else:
        print(f"Error: Please provide a .npy tensor file")
        print(f"Expected file ending with .npy, got: {args.data_file}")
        return

    # Save results (common for both tensor and traditional experiments)
    os.makedirs(args.output_dir, exist_ok=True)

    # Save results
    results_file = os.path.join(args.output_dir, f"{args.dataset_name}_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")

    # Save co-occurrence matrices
    co_occurrence_file = os.path.join(
        args.output_dir, f"{args.dataset_name}_co_occurrence.pkl"
    )
    with open(co_occurrence_file, "wb") as f:
        pickle.dump(
            {
                "co_occurrence_matrix": co_occurrence_matrix,
                "greedy_co_occurrence_matrix": greedy_co_occurrence_matrix,
            },
            f,
        )
    print(f"Co-occurrence matrices saved to {co_occurrence_file}")

    # Save convergence logs if available
    if convergence_logs is not None:
        convergence_file = os.path.join(
            args.output_dir,
            f"{args.dataset_name}_standardized_{args.power}_{args.D}_convergence.pkl",
        )
        with open(convergence_file, "wb") as f:
            pickle.dump(convergence_logs, f)
        print(f"Convergence logs saved to {convergence_file}")

    print("Experiment completed successfully!")
    print(f"Results shape: {results_df.shape}")
    print(f"Co-occurrence matrix shape: {co_occurrence_matrix.shape}")


if __name__ == "__main__":
    main()
