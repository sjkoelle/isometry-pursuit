#!/usr/bin/env python3
"""
Compute isometry loss of selected support.
This script computes the isometry loss for a given coefficient matrix and selected indices.
"""

import click
import os
import numpy as np
import warnings
from isometry_pursuit.loss import isometry_loss


def isometry_loss_top_d(matrix: np.ndarray, power=1, d=None) -> float:
    """Compute isometry loss using only the top D singular values"""
    singular_values = np.linalg.svd(
        matrix, compute_uv=False
    )  # no 0s since we are in full rankistan

    # Check for rank deficiency - if any singular values are too small, skip this matrix
    min_singular_value = np.min(singular_values)
    if min_singular_value < 1e-10:  # Threshold for numerical rank deficiency
        return float(
            "inf"
        )  # Return infinity to indicate this matrix should not be selected

    # Use only top D singular values if specified
    if d is not None:
        singular_values = singular_values[:d]

    singular_values = singular_values**power

    # Log relevant arguments when overflow occurs
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        output = np.sum(
            (np.exp(singular_values) + np.exp(singular_values ** (-1))) / (2 * np.e)
        )

        # Check if we got any overflow warnings
        for warning in w:
            if "overflow" in str(warning.message).lower():
                return float(
                    "inf"
                )  # Return infinity to indicate this matrix should not be selected

    return output


@click.command()
@click.option(
    "--coefficient-matrix",
    required=True,
    help="Path to .npy coefficient matrix file",
)
@click.option(
    "--selected-indices",
    required=True,
    help="Path to .npy selected indices file",
)
@click.option(
    "--power",
    type=float,
    default=1.0,
    help="Power parameter for isometry loss (default: 1.0)",
)
@click.option(
    "--D",
    type=int,
    help="Number of top singular values to use (default: use all)",
)
@click.option(
    "--output-file",
    help="Path to save the computed loss (optional)",
)
def main(
    coefficient_matrix,
    selected_indices,
    power,
    d,
    output_file,
):
    """Compute isometry loss of selected support"""

    # Load coefficient matrix
    beta = np.load(coefficient_matrix)
    print(f"Loaded coefficient matrix shape: {beta.shape}")

    # Load selected indices
    indices = np.load(selected_indices)
    print(f"Loaded selected indices: {indices}")
    print(f"Number of selected features: {len(indices)}")

    # Extract the selected support (rows corresponding to selected indices)
    selected_support = beta[indices, :]
    print(f"Selected support shape: {selected_support.shape}")

    # Compute isometry loss
    if d is not None:
        loss = isometry_loss_top_d(selected_support, power=power, d=d)
        print(f"\nIsometry Loss Results (using top {d} singular values):")
    else:
        loss = isometry_loss(selected_support, power=power)
        print(f"\nIsometry Loss Results (using all singular values):")

    print(f"  Power parameter: {power}")
    print(f"  Selected support shape: {selected_support.shape}")
    if d is not None:
        print(f"  Dimension parameter D: {d}")
    print(f"  Isometry loss: {loss:.6f}")

    # Save loss if output file specified
    if output_file:
        results = {
            "isometry_loss": loss,
            "power": power,
            "dimension_d": d,
            "selected_indices": indices,
            "selected_support_shape": selected_support.shape,
            "coefficient_matrix_shape": beta.shape,
        }

        # Save as numpy file for the loss value
        np.save(output_file, loss)

        # Also save metadata
        metadata_file = output_file.replace(".npy", "_metadata.npy")
        np.save(metadata_file, results)

        print(f"  Loss saved to: {output_file}")
        print(f"  Metadata saved to: {metadata_file}")


if __name__ == "__main__":
    main()
