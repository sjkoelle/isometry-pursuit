#!/usr/bin/env python3
"""
Ethanol experiment runner for isometry pursuit experiments.
This script runs the ethanol experiment with the specific parameters used in the paper.
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
from isometry_pursuit.experiment import analyze_data
from tqdm import tqdm


def run_ethanol_experiment(
    data_file,
    output_dir="results",
    D=2,
    power=1.0,
    R=None,  # Will be set to number of data points
    compute_brute=False,
    log_convergence=False,
    seed=1234,
):
    """
    Run ethanol isometry pursuit experiment.

    Parameters:
    -----------
    data_file : str
        Path to the processed ethanol data CSV file
    output_dir : str
        Directory to save results
    D : int
        Target dimension (should be 2 for ethanol)
    power : float
        Power parameter for transformation
    R : int
        Number of data points to process
    compute_brute : bool
        Whether to compute brute force solutions
    log_convergence : bool
        Whether to log convergence information
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    results_df : pandas.DataFrame
        Results from the experiment
    """

    print(f"Running ethanol experiment...")
    print(f"Parameters: D={D}, power={power}, R={R}")
    print(f"Data file: {data_file}")

    # Load data
    data = pd.read_csv(data_file)
    print(f"Data shape: {data.shape}")

    # Set R to number of data points if not specified
    if R is None:
        R = len(data)
    print(f"Using {R} data points for experiments")

    # Set random seed
    np.random.seed(seed)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # For ethanol, we run the experiment on each sample individually
    # as described in the paper (replicas correspond to individual data points)
    losses = []
    support_cardinalities_basis_pursuit = []
    two_stage_losses = []
    random_two_stage_losses = []
    greedy_multitask_norms_two_stage = []
    brute_isometry_losses = []
    brute_losses = []
    basis_pursuit_times = []
    greedy_times = []
    stage_two_times = []

    print(f"Processing {R} data points...")

    for i in tqdm(range(R)):
        # Set seed for each run to ensure reproducibility
        np.random.seed(seed + i)

        # Get the data for this run (each row is a flattened projection)
        sample_data = data.iloc[i % len(data)].values

        # Reshape back to the expected format (D x P)
        # The data was flattened as (dimension * p,) so we need to reshape
        P = len(sample_data) // D
        sample_data_reshaped = sample_data.reshape(D, P)

        try:
            # Run the analysis for this sample
            (
                loss,
                support_cardinality_basis_pursuit,
                two_stage_loss,
                random_two_stage_loss,
                greedy_multitask_norm_two_stage,
                brute_loss,
                brute_isometry_loss,
                basis_pursuit_time,
                stage_two_time,
                greedy_time,
                two_stage_multitask,
                greedy_solution,
                basis_pursuit_iterations,
                convergence_log,
            ) = analyze_data(
                sample_data_reshaped, compute_brute=compute_brute, power=power
            )

            losses.append(loss)
            support_cardinalities_basis_pursuit.append(
                support_cardinality_basis_pursuit
            )
            two_stage_losses.append(two_stage_loss)
            random_two_stage_losses.append(random_two_stage_loss)
            greedy_multitask_norms_two_stage.append(greedy_multitask_norm_two_stage)
            brute_losses.append(brute_loss)
            brute_isometry_losses.append(brute_isometry_loss)
            basis_pursuit_times.append(basis_pursuit_time)
            greedy_times.append(greedy_time)
            stage_two_times.append(stage_two_time)

        except Exception as e:
            print(f"Warning: Data point {i} failed: {e}")
            # Add NaN values for failed data points
            losses.append(np.nan)
            support_cardinalities_basis_pursuit.append(np.nan)
            two_stage_losses.append(np.nan)
            random_two_stage_losses.append(np.nan)
            greedy_multitask_norms_two_stage.append(np.nan)
            brute_losses.append(np.nan)
            brute_isometry_losses.append(np.nan)
            basis_pursuit_times.append(np.nan)
            greedy_times.append(np.nan)
            stage_two_times.append(np.nan)

    # Create results DataFrame
    results_df = pd.DataFrame(
        {
            "Losses": losses,
            "Support Cardinalities (Basis Pursuit)": support_cardinalities_basis_pursuit,
            "Two-Stage Losses": two_stage_losses,
            "Random Two-Stage Losses": random_two_stage_losses,
            "Greedy Multitask Norms (Two-Stage)": greedy_multitask_norms_two_stage,
            "Brute Isometry Losses": brute_isometry_losses,
            "Brute Losses": brute_losses,
            "Basis Pursuit Times": basis_pursuit_times,
            "Stage Two Times": stage_two_times,
            "Greedy Times": greedy_times,
        }
    )

    # Save results
    results_file = os.path.join(output_dir, f"ethanol_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")

    # Save experiment metadata
    metadata = {
        "dataset_name": "ethanol",
        "D": D,
        "power": power,
        "R": R,
        "compute_brute": compute_brute,
        "log_convergence": log_convergence,
        "data_shape": data.shape,
        "results_file": results_file,
        "seed": seed,
    }

    metadata_file = os.path.join(
        output_dir, f"ethanol_standardized_{power}_{D}_metadata.json"
    )
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_file}")

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Run ethanol isometry pursuit experiment"
    )
    parser.add_argument(
        "--data-file", required=True, help="Path to processed ethanol data file (CSV)"
    )
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument(
        "--D", type=int, default=2, help="Target dimension (should be 2 for ethanol)"
    )
    parser.add_argument("--power", type=float, default=1.0, help="Power parameter")
    parser.add_argument(
        "--R",
        type=int,
        default=None,
        help="Number of data points to process (defaults to number of rows in data)",
    )
    parser.add_argument(
        "--compute-brute", action="store_true", help="Compute brute force solutions"
    )
    parser.add_argument(
        "--log-convergence", action="store_true", help="Log convergence information"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    args = parser.parse_args()

    # Check if data file exists
    if not os.path.exists(args.data_file):
        print(f"Error: Data file not found: {args.data_file}")
        print("Please run the data loading script first:")
        print(
            "python isometry_pursuit/exec/load_ethanol_exec.py --config-path <config> --positions-path <positions> --angles-path <angles> --indices-path <indices>"
        )
        sys.exit(1)

    # Run experiment
    results_df = run_ethanol_experiment(
        data_file=args.data_file,
        output_dir=args.output_dir,
        D=args.D,
        power=args.power,
        R=args.R,
        compute_brute=args.compute_brute,
        log_convergence=args.log_convergence,
        seed=args.seed,
    )

    print("Ethanol experiment completed successfully!")
    print(f"Results shape: {results_df.shape}")
    print(f"Successful data points: {results_df['Losses'].notna().sum()}")


if __name__ == "__main__":
    main()
