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


def line_search_constraint_violation(
    data_tensor,
    target_sparsity,
    compute_brute=False,
    power=1.0,
    log_convergence=False,
    max_iterations=20,
    tolerance=0.5,
    initial_bounds=(0.0, 2.0),
    collect_path_data=False,
):
    """
    Perform line search to find constraint violation that achieves target sparsity.

    Parameters:
    -----------
    data_tensor : numpy.ndarray
        Data tensor for the experiment
    target_sparsity : int
        Target number of features to select
    compute_brute : bool
        Whether to compute brute force solutions
    power : float
        Power parameter
    log_convergence : bool
        Whether to log convergence information
    max_iterations : int
        Maximum number of line search iterations
    tolerance : float
        Tolerance for sparsity difference
    initial_bounds : tuple
        Initial bounds for constraint violation search
    collect_path_data : bool
        Whether to collect regularization path data during search

    Returns:
    --------
    best_violation : float
        Best constraint violation found
    best_sparsity : float
        Actual sparsity achieved
    results : tuple
        Results from the best experiment
    path_data : dict, optional
        Regularization path data if collect_path_data=True
    """
    print(f"Performing line search for target sparsity: {target_sparsity}")

    # Binary search bounds
    low, high = initial_bounds
    best_violation = None
    best_sparsity = None
    best_results = None

    # Collect path data if requested
    if collect_path_data:
        path_data = {
            "constraint_violations": [],
            "sparsities": [],
            "losses": [],
            "iterations": [],
        }

    for iteration in range(max_iterations):
        # Test midpoint
        mid = (low + high) / 2
        print(f"Iteration {iteration + 1}: Testing constraint violation = {mid:.4f}")

        try:
            # Run experiment with current constraint violation
            if log_convergence:
                (
                    results_df,
                    co_occurrence_matrix,
                    greedy_co_occurrence_matrix,
                    convergence_logs,
                    coefficient_matrices,
                ) = run_tensor_experiment(
                    data_tensor=data_tensor,
                    compute_brute=compute_brute,
                    power=power,
                    log_convergence=log_convergence,
                    constraint_violation=mid,
                )
            else:
                (
                    results_df,
                    co_occurrence_matrix,
                    greedy_co_occurrence_matrix,
                    coefficient_matrices,
                ) = run_tensor_experiment(
                    data_tensor=data_tensor,
                    compute_brute=compute_brute,
                    power=power,
                    log_convergence=log_convergence,
                    constraint_violation=mid,
                )
                convergence_logs = None

            # Calculate average sparsity
            avg_sparsity = results_df["support_cardinality_basis_pursuit"].mean()
            avg_loss = results_df["loss"].mean()
            print(f"  Average sparsity: {avg_sparsity:.2f}")

            # Collect path data if requested
            if collect_path_data:
                path_data["constraint_violations"].append(mid)
                path_data["sparsities"].append(avg_sparsity)
                path_data["losses"].append(avg_loss)
                path_data["iterations"].append(iteration + 1)

            # Check if we're close enough to target
            sparsity_diff = abs(avg_sparsity - target_sparsity)
            if sparsity_diff <= tolerance:
                print(f"  Target sparsity achieved! (diff: {sparsity_diff:.2f})")
                if collect_path_data:
                    return (
                        mid,
                        avg_sparsity,
                        (
                            results_df,
                            co_occurrence_matrix,
                            greedy_co_occurrence_matrix,
                            convergence_logs,
                            coefficient_matrices,
                        ),
                        path_data,
                    )
                else:
                    return (
                        mid,
                        avg_sparsity,
                        (
                            results_df,
                            co_occurrence_matrix,
                            greedy_co_occurrence_matrix,
                            convergence_logs,
                            coefficient_matrices,
                        ),
                    )

            # Update best result if this is closer to target
            if best_sparsity is None or sparsity_diff < abs(
                best_sparsity - target_sparsity
            ):
                best_violation = mid
                best_sparsity = avg_sparsity
                best_results = (
                    results_df,
                    co_occurrence_matrix,
                    greedy_co_occurrence_matrix,
                    convergence_logs,
                )

            # Update search bounds
            if avg_sparsity < target_sparsity:
                # Need more sparsity, decrease constraint violation (tighter constraint)
                high = mid
            else:
                # Need less sparsity, increase constraint violation (looser constraint)
                low = mid

        except Exception as e:
            print(f"  Failed with violation {mid:.4f}: {e}")
            # If experiment fails, try a smaller constraint violation
            high = mid
            continue

    print(
        f"Line search completed. Best constraint violation: {best_violation:.4f} (sparsity: {best_sparsity:.2f})"
    )
    if collect_path_data:
        return best_violation, best_sparsity, best_results, path_data
    else:
        return best_violation, best_sparsity, best_results


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
        help="Target sparsity (number of features to select). If provided, will perform line search to find constraint violation that achieves this sparsity.",
    )
    parser.add_argument("--power", type=float, default=1.0, help="Power parameter")
    parser.add_argument(
        "--compute-brute", action="store_true", help="Compute brute force solutions"
    )
    parser.add_argument(
        "--log-convergence", action="store_true", help="Log convergence information"
    )
    parser.add_argument(
        "--constraint-violation",
        type=float,
        default=0.0,
        help="Constraint violation tolerance for group basis pursuit (default: 0.0)",
    )
    parser.add_argument(
        "--try-multiple-violations",
        action="store_true",
        help="Try multiple constraint violation values when D is not provided",
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

        # Extract tensor dimension
        tensor_dimension = data_tensor.shape[0]
        print(f"Tensor dimension: {tensor_dimension}")

        # Run experiment using tensor
        if args.D is not None:
            # Use line search to find constraint violation for target sparsity
            print(f"Target sparsity: {args.D}")
            best_violation, best_sparsity, best_results = (
                line_search_constraint_violation(
                    data_tensor=data_tensor,
                    target_sparsity=args.D,
                    compute_brute=args.compute_brute,
                    power=args.power,
                    log_convergence=args.log_convergence,
                )
            )

            if best_results is not None:
                (
                    results_df,
                    co_occurrence_matrix,
                    greedy_co_occurrence_matrix,
                    convergence_logs,
                    coefficient_matrices,
                ) = best_results
                print(
                    f"Final constraint violation: {best_violation:.4f}, achieved sparsity: {best_sparsity:.2f}"
                )
            else:
                print(
                    "Line search failed, falling back to default constraint violation"
                )
                if args.log_convergence:
                    (
                        results_df,
                        co_occurrence_matrix,
                        greedy_co_occurrence_matrix,
                        convergence_logs,
                        coefficient_matrices,
                    ) = run_tensor_experiment(
                        data_tensor=data_tensor,
                        compute_brute=args.compute_brute,
                        power=args.power,
                        log_convergence=args.log_convergence,
                        constraint_violation=args.constraint_violation,
                    )
                else:
                    (
                        results_df,
                        co_occurrence_matrix,
                        greedy_co_occurrence_matrix,
                        coefficient_matrices,
                    ) = run_tensor_experiment(
                        data_tensor=data_tensor,
                        compute_brute=args.compute_brute,
                        power=args.power,
                        log_convergence=args.log_convergence,
                        constraint_violation=args.constraint_violation,
                    )
                    convergence_logs = None
        else:
            # Automatically search for constraint violation that gives D sparsity
            target_sparsity = tensor_dimension
            print(
                f"No --D specified, automatically searching for constraint violation that achieves sparsity = {target_sparsity}"
            )

            # Calculate max column norm for adaptive search bounds
            max_col_norm = np.max(np.linalg.norm(data_tensor, axis=0))
            print(f"Max column norm: {max_col_norm:.4f}")

            best_violation, best_sparsity, best_results = (
                line_search_constraint_violation(
                    data_tensor=data_tensor,
                    target_sparsity=target_sparsity,
                    compute_brute=args.compute_brute,
                    power=args.power,
                    log_convergence=args.log_convergence,
                    initial_bounds=(0.0, max_col_norm),
                )
            )

            if best_results is not None:
                (
                    results_df,
                    co_occurrence_matrix,
                    greedy_co_occurrence_matrix,
                    convergence_logs,
                    coefficient_matrices,
                ) = best_results
                print(
                    f"Final constraint violation: {best_violation:.4f}, achieved sparsity: {best_sparsity:.2f}"
                )
            else:
                print(
                    "Automatic search failed, falling back to default constraint violation"
                )
                if args.log_convergence:
                    (
                        results_df,
                        co_occurrence_matrix,
                        greedy_co_occurrence_matrix,
                        convergence_logs,
                        coefficient_matrices,
                    ) = run_tensor_experiment(
                        data_tensor=data_tensor,
                        compute_brute=args.compute_brute,
                        power=args.power,
                        log_convergence=args.log_convergence,
                        constraint_violation=args.constraint_violation,
                    )
                else:
                    (
                        results_df,
                        co_occurrence_matrix,
                        greedy_co_occurrence_matrix,
                        coefficient_matrices,
                    ) = run_tensor_experiment(
                        data_tensor=data_tensor,
                        compute_brute=args.compute_brute,
                        power=args.power,
                        log_convergence=args.log_convergence,
                        constraint_violation=args.constraint_violation,
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

    # Save convergence logs if available
    if convergence_logs is not None:
        convergence_file = os.path.join(
            args.output_dir,
            f"{args.dataset_name}_standardized_{args.power}_{args.D}_convergence.pkl",
        )
        with open(convergence_file, "wb") as f:
            pickle.dump(convergence_logs, f)
        print(f"Convergence logs saved to {convergence_file}")

    # Save coefficient matrices
    if coefficient_matrices is not None:
        # Create coefficient results structure
        coefficient_results = {}
        D = data_tensor.shape[0]  # Number of features
        R = data_tensor.shape[2]  # Number of replicates

        # Get the constraint violation used (from line search or default)
        constraint_violation = getattr(args, "constraint_violation", 0.0)
        if hasattr(args, "D") and args.D is not None:
            # If D was specified, we used line search, so get the best violation
            constraint_violation = (
                best_violation
                if "best_violation" in locals()
                else args.constraint_violation
            )

        for replicate_idx in range(R):
            coefficient_results[replicate_idx] = {
                constraint_violation: coefficient_matrices[replicate_idx]
            }

        coefficient_file = os.path.join(
            args.output_dir,
            f"{args.dataset_name}_coefficient_matrices.pkl",
        )
        with open(coefficient_file, "wb") as f:
            pickle.dump(coefficient_results, f)
        print(f"Coefficient matrices saved to {coefficient_file}")

    print("Experiment completed successfully!")
    print(f"Results shape: {results_df.shape}")
    print(f"Co-occurrence matrix shape: {co_occurrence_matrix.shape}")


if __name__ == "__main__":
    main()
