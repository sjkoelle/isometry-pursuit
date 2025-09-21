#!/usr/bin/env python3
"""
Example script demonstrating convergence logging functionality.
"""

import numpy as np
import matplotlib.pyplot as plt
from isometry_pursuit.experiment import (
    run_resampling_experiment,
    plot_convergence,
    analyze_convergence_logs,
)
from isometry_pursuit.algorithm import (
    group_basis_pursuit,
    create_convergence_log,
)
from isometry_pursuit.transformation import exponential_transformation


def example_single_run():
    """Example of convergence logging for a single run."""
    print("=== Single Run Convergence Example ===")

    # Create a simple test matrix
    np.random.seed(42)
    D, P = 3, 8
    X = np.random.randn(D, P)
    X_transformed = exponential_transformation(X, power=1.0)

    print("\n--- CVXPY Implementation (5 Logarithmic Points) ---")
    # Run with CVXPY implementation
    beta, iterations = group_basis_pursuit(X_transformed)

    # Create convergence log by calling group_basis_pursuit multiple times
    convergence_log = create_convergence_log(X_transformed)

    print(f"Final iterations: {iterations}")
    print(f"Convergence log keys: {convergence_log.keys()}")
    print(f"Number of logged points: {len(convergence_log['iterations'])}")
    print(f"Iteration points: {convergence_log['iterations']}")
    print(f"Loss values: {convergence_log['losses']}")
    print(f"Sparsity values: {convergence_log['sparsities']}")

    # Plot convergence
    plot_convergence(
        convergence_log,
        title="CVXPY Implementation Convergence (5 Logarithmic Points)",
        save_path="/Users/samsonkoelle/isometry-pursuit/figures/cvxpy_convergence.png",
    )


def example_multiple_runs():
    """Example of convergence logging for multiple runs."""
    print("\n=== Multiple Runs Convergence Example ===")

    # Create a simple test dataset
    np.random.seed(42)
    data = np.random.randn(10, 8)  # 10 samples, 8 features
    D = 3

    # Run experiment with convergence logging (CVXPY with 5 logarithmic points)
    print("--- CVXPY Implementation (5 Logarithmic Points) ---")
    results_df, co_occurence_matrix, greedy_co_occurence_matrix, convergence_logs = (
        run_resampling_experiment(
            data,
            D,
            frac=0.5,
            R=3,
            compute_brute=False,
            power=1.0,
            log_convergence=True,
        )
    )

    print(f"Number of runs: {len(convergence_logs)}")
    print(f"Results DataFrame columns: {list(results_df.columns)}")

    # Print convergence summary
    print("\nConvergence Summary:")
    for i, log in enumerate(convergence_logs):
        if log and log["iterations"]:
            print(f"Run {i+1}:")
            print(f"  Iteration points: {log['iterations']}")
            print(f"  Loss progression: {[f'{l:.4f}' for l in log['losses']]}")
            print(f"  Sparsity progression: {log['sparsities']}")

    # Analyze convergence across runs
    analyze_convergence_logs(
        convergence_logs,
        save_path="/Users/samsonkoelle/isometry-pursuit/figures/multi_run_convergence.png",
    )

    return results_df, convergence_logs


def example_convergence_analysis():
    """Example of analyzing convergence data."""
    print("\n=== Convergence Analysis Example ===")

    # Get some convergence data
    results_df, convergence_logs = example_multiple_runs()

    # Print some statistics
    print("\nConvergence Statistics:")
    for i, log in enumerate(convergence_logs):
        if log and log["iterations"]:
            print(f"Run {i+1}:")
            print(f"  Iterations: {len(log['iterations'])}")
            print(f"  Final loss: {log['losses'][-1]:.4f}")
            print(f"  Final sparsity: {log['sparsities'][-1]}")
            print(f"  Final residual: {log['residuals'][-1]:.6f}")
            print()


if __name__ == "__main__":
    print("Running convergence logging examples...\n")

    try:
        example_single_run()
        example_multiple_runs()
        example_convergence_analysis()

        print("\n🎉 All examples completed successfully!")
        print("\nTo use convergence logging in your notebooks:")
        print("1. Set log_convergence=True in run_resampling_experiment()")
        print("2. Use plot_convergence() for single run analysis")
        print("3. Use analyze_convergence_logs() for multi-run analysis")

    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        import traceback

        traceback.print_exc()
