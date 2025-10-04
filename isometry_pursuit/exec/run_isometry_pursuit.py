#!/usr/bin/env python3
"""
Simple isometry pursuit runner.
This script runs isometry_pursuit directly on a single data matrix.
"""

import click
import os
import pickle
import numpy as np
from isometry_pursuit.algorithm import isometry_pursuit


@click.command()
@click.option(
    "--data-file",
    required=True,
    help="Path to .npy data matrix file",
)
@click.option("--output-dir", required=True, help="Output directory name")
@click.option(
    "--constraint-violation",
    type=float,
    default=0.0,
    help="Constraint violation tolerance for group basis pursuit (default: 0.0)",
)
def main(
    data_file,
    output_dir,
    constraint_violation,
):
    """Run isometry pursuit on a single data matrix"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data tensor
    data_tensor = np.load(data_file)

    # Extract first replicate as 2D matrix
    # New format: (n_replicates, D, n_samples) -> (D, n_samples)
    X = data_tensor[0, :, :]

    # Run isometry pursuit
    result = isometry_pursuit(
        X,
        constraint_violation=constraint_violation,
    )

    # Save coefficient matrix
    beta_file = os.path.join(output_dir, "coefficient_matrix.npy")
    np.save(beta_file, result["beta"])

    # Save selected indices
    indices_file = os.path.join(output_dir, "selected_indices.npy")
    np.save(indices_file, result["basis_pursuit_indices"])

    # Print summary
    print(f"Selected {result['basis_pursuit_indices']} features")


if __name__ == "__main__":
    main()
