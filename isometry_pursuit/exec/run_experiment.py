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
from isometry_pursuit.experiment import run_resampling_experiment, run_tensor_experiment


def run_experiment(
    data,
    dataset_name,
    D,
    power=1.0,
    frac=0.5,
    R=25,
    compute_brute=False,
    log_convergence=False,
    output_dir="results",
):
    """
    Run isometry pursuit experiment on given data.

    Parameters:
    -----------
    data : pandas.DataFrame
        Input data matrix
    dataset_name : str
        Name of the dataset (e.g., 'iris', 'wine')
    D : int
        Target dimension
    power : float
        Power parameter for transformation
    frac : float
        Fraction of data to use in each resampling
    R : int
        Number of resampling runs
    compute_brute : bool
        Whether to compute brute force solutions
    log_convergence : bool
        Whether to log convergence information
    output_dir : str
        Directory to save results

    Returns:
    --------
    results_df : pandas.DataFrame
        Results from the experiment
    co_occurrence_matrix : np.ndarray
        Co-occurrence matrix for two-stage solutions
    greedy_co_occurrence_matrix : np.ndarray
        Co-occurrence matrix for greedy solutions
    convergence_logs : list or None
        Convergence logs if log_convergence=True
    """

    print(f"Running experiment on {dataset_name} dataset...")
    print(f"Parameters: D={D}, power={power}, frac={frac}, R={R}")
    print(f"Data shape: {data.shape}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Run the experiment
    if log_convergence:
        (
            results_df,
            co_occurrence_matrix,
            greedy_co_occurrence_matrix,
            convergence_logs,
        ) = run_resampling_experiment(
            data,
            D,
            frac=frac,
            R=R,
            power=power,
            compute_brute=compute_brute,
            log_convergence=log_convergence,
        )
    else:
        results_df, co_occurrence_matrix, greedy_co_occurrence_matrix = (
            run_resampling_experiment(
                data,
                D,
                frac=frac,
                R=R,
                power=power,
                compute_brute=compute_brute,
                log_convergence=log_convergence,
            )
        )
        convergence_logs = None

    # Save results
    results_file = os.path.join(output_dir, f"{dataset_name}_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")

    # Save co-occurrence matrices
    co_occurrence_file = os.path.join(output_dir, f"{dataset_name}_co_occurrence.pkl")
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
            output_dir, f"{dataset_name}_standardized_{power}_{D}_convergence.pkl"
        )
        with open(convergence_file, "wb") as f:
            pickle.dump(convergence_logs, f)
        print(f"Convergence logs saved to {convergence_file}")

    # Save experiment metadata
    metadata = {
        "dataset_name": dataset_name,
        "D": D,
        "power": power,
        "frac": frac,
        "R": R,
        "compute_brute": compute_brute,
        "log_convergence": log_convergence,
        "data_shape": data.shape,
        "results_file": results_file,
        "co_occurrence_file": co_occurrence_file,
    }

    if convergence_logs is not None:
        metadata["convergence_file"] = convergence_file

    metadata_file = os.path.join(output_dir, f"{dataset_name}_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_file}")

    return (
        results_df,
        co_occurrence_matrix,
        greedy_co_occurrence_matrix,
        convergence_logs,
    )


def main():
    parser = argparse.ArgumentParser(description="Run isometry pursuit experiment")
    parser.add_argument(
        "--data-file",
        required=True,
        help="Path to data file (CSV for ethanol, .npy tensor for others)",
    )
    parser.add_argument("--dataset-name", required=True, help="Name of the dataset")
    parser.add_argument("--D", type=int, required=True, help="Target dimension")
    parser.add_argument("--power", type=float, default=1.0, help="Power parameter")
    parser.add_argument(
        "--frac", type=float, default=0.5, help="Fraction of data to use"
    )
    parser.add_argument("--R", type=int, default=25, help="Number of resampling runs")
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

    # For non-ethanol datasets, load the resampled data tensor directly
    if args.dataset_name != "ethanol":
        if args.data_file.endswith(".npy"):
            print(f"Loading resampled data tensor from {args.data_file}")
            data_tensor = np.load(args.data_file)
            print(f"Data tensor shape: {data_tensor.shape}")

            # Run experiment using tensor
            if args.log_convergence:
                (
                    results_df,
                    co_occurrence_matrix,
                    greedy_co_occurrence_matrix,
                    convergence_logs,
                ) = run_tensor_experiment(
                    data_tensor=data_tensor,
                    D=args.D,
                    R=args.R,
                    compute_brute=args.compute_brute,
                    power=args.power,
                    log_convergence=args.log_convergence,
                )
            else:
                results_df, co_occurrence_matrix, greedy_co_occurrence_matrix = (
                    run_tensor_experiment(
                        data_tensor=data_tensor,
                        D=args.D,
                        R=args.R,
                        compute_brute=args.compute_brute,
                        power=args.power,
                        log_convergence=args.log_convergence,
                    )
                )
                convergence_logs = None
        else:
            print(f"Error: For non-ethanol datasets, please provide a .npy tensor file")
            print(f"Expected file ending with .npy, got: {args.data_file}")
            return
    else:
        # For ethanol, load CSV as before
        data = pd.read_csv(args.data_file)

        # Run experiment using traditional resampling
        if args.log_convergence:
            (
                results_df,
                co_occurrence_matrix,
                greedy_co_occurrence_matrix,
                convergence_logs,
            ) = run_experiment(
                data=data,
                dataset_name=args.dataset_name,
                D=args.D,
                power=args.power,
                frac=args.frac,
                R=args.R,
                compute_brute=args.compute_brute,
                log_convergence=args.log_convergence,
                output_dir=args.output_dir,
            )
        else:
            results_df, co_occurrence_matrix, greedy_co_occurrence_matrix = (
                run_experiment(
                    data=data,
                    dataset_name=args.dataset_name,
                    D=args.D,
                    power=args.power,
                    frac=args.frac,
                    R=args.R,
                    compute_brute=args.compute_brute,
                    log_convergence=args.log_convergence,
                    output_dir=args.output_dir,
                )
            )
            convergence_logs = None

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
