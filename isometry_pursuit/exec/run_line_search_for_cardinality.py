#!/usr/bin/env python3
"""
Line search for cardinality runner.
This script runs line_search_for_cardinality to find constraint violation for target cardinality.
"""

import click
import os
import pickle
import numpy as np
from isometry_pursuit.algorithm import line_search_for_cardinality


@click.command()
@click.option(
    "--data-file",
    required=True,
    help="Path to .npy data matrix file",
)
@click.option("--output-dir", required=True, help="Output directory name")
@click.option(
    "--target-cardinality",
    type=int,
    required=False,
    help="Target support cardinality",
)
def main(
    data_file,
    output_dir,
    target_cardinality=None,
):
    """Run line search to find constraint violation for target cardinality"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data tensor
    data_tensor = np.load(data_file)

    # Extract first replicate as 2D matrix
    # New format: (n_replicates, D, n_samples) -> (D, n_samples)
    X = data_tensor[0, :, :]
    if target_cardinality is None:
        target_cardinality = X.shape[0]
    # Run line search
    best_violation, best_cardinality, result = line_search_for_cardinality(
        X,
        target_cardinality=target_cardinality,
    )

    # Save coefficient matrix
    beta_file = os.path.join(output_dir, "coefficient_matrix.npy")
    np.save(beta_file, result["beta"])

    # Save selected indices
    indices_file = os.path.join(output_dir, "selected_indices.npy")
    np.save(indices_file, result["basis_pursuit_indices"])

    # Print summary
    print(f"Line search completed successfully!")
    print(f"Target cardinality: {target_cardinality}")
    print(f"Achieved cardinality: {best_cardinality}")
    print(f"Optimal constraint violation: {best_violation:.6f}")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
