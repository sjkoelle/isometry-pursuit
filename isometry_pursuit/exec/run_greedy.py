#!/usr/bin/env python3
"""
Greedy algorithm experiment runner.
This script runs greedy feature selection experiments on data tensors or existing results.
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
from isometry_pursuit.algorithm import greedy
from isometry_pursuit.loss import isometry_loss
from isometry_pursuit.transformation import exponential_transformation


def run_greedy_on_tensor(data_tensor, power=1.0, target_dimension=None):
    """
    Run greedy algorithm on data tensor.

    Parameters:
    -----------
    data_tensor : numpy.ndarray
        Data tensor of shape (D, n_samples_per_replicate, n_replicates)
    power : float
        Power parameter for isometry loss
    target_dimension : int, optional
        Target dimension (if None, uses tensor dimension)

    Returns:
    --------
    results_df : pandas.DataFrame
        Results with greedy metrics for each replicate
    """
    D = data_tensor.shape[0]
    R = data_tensor.shape[2]

    if target_dimension is None:
        target_dimension = D

    print(
        f"Running greedy algorithm on {R} replicates with target dimension {target_dimension}"
    )

    isometry_loss_power = lambda x, indices=None: isometry_loss(x, power, indices)

    losses = []
    greedy_times = []
    greedy_solutions = []

    for i in range(R):
        print(f"Processing replicate {i+1}/{R}")
        X = data_tensor[:, :, i]

        import time

        start_time = time.time()

        # Run greedy algorithm
        greedy_solution = greedy(
            X, isometry_loss_power, target_dimension, [], verbose=False
        )

        end_time = time.time()
        greedy_time = end_time - start_time

        # Calculate loss
        loss = isometry_loss_power(X[:, greedy_solution])

        losses.append(loss)
        greedy_times.append(greedy_time)
        greedy_solutions.append(greedy_solution)

        print(f"  Loss: {loss:.4f}, Time: {greedy_time:.4f}s")

    # Create results DataFrame
    results_data = {
        "loss": losses,
        "greedy_time": greedy_times,
        "target_dimension": [target_dimension] * R,
        "replicate": list(range(R)),
    }

    results_df = pd.DataFrame(results_data)
    return results_df, greedy_solutions


def main():
    parser = argparse.ArgumentParser(description="Run greedy algorithm experiment")
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
    parser.add_argument("--output-dir", default="results", help="Output directory")

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data_file}")

    if args.data_file.endswith(".npy"):
        print(f"Loading data tensor from {args.data_file}")
        data_tensor = np.load(args.data_file)
        print(f"Data tensor shape: {data_tensor.shape}")

        # Extract tensor dimension
        tensor_dimension = data_tensor.shape[0]
        print(f"Tensor dimension: {tensor_dimension}")

        # Run greedy experiment
        results_df, greedy_solutions = run_greedy_on_tensor(
            data_tensor=data_tensor,
            power=args.power,
            target_dimension=args.D,
        )

    else:
        print(f"Error: Please provide a .npy tensor file")
        print(f"Expected file ending with .npy, got: {args.data_file}")
        return

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    results_file = os.path.join(
        args.output_dir, f"{args.dataset_name}_greedy_results.csv"
    )
    results_df.to_csv(results_file, index=False)
    print(f"Greedy results saved to {results_file}")

    # Save greedy solutions
    if greedy_solutions is not None:
        solutions_file = os.path.join(
            args.output_dir, f"{args.dataset_name}_greedy_solutions.pkl"
        )
        with open(solutions_file, "wb") as f:
            pickle.dump(greedy_solutions, f)
        print(f"Greedy solutions saved to {solutions_file}")

    print("Greedy experiment completed successfully!")
    print(f"Results shape: {results_df.shape}")


if __name__ == "__main__":
    main()
