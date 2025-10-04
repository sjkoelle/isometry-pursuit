#!/usr/bin/env python3
"""
Generate simulated data consisting of orthogonal matrices.
Creates 10 8-dimensional orthogonal matrices randomly rotated in R^10,
and one 10-dimensional orthogonal matrix.
"""

import click
import os
import numpy as np
import json
from scipy.stats import ortho_group


def generate_orthogonal_data(
    output_dir="data/orthogonal",
    matrix_specs=[(8, 10), (10, 1)],
    n_replicates=25,
    embedding_dim=20,
    noise_level=0.0,
    seed=42,
):
    """
    Generate simulated data with orthogonal matrices.

    Parameters:
    -----------
    output_dir : str
        Output directory for generated data
    matrix_specs : list of tuples
        List of (dimension, count) tuples specifying orthogonal matrices to generate
        e.g., [(8, 10), (10, 1)] means 10 matrices of dimension 8 and 1 matrix of dimension 10
    n_replicates : int
        Number of replicates to generate
    embedding_dim : int
        Dimension to embed all matrices in (default: 20)
    noise_level : float
        Standard deviation of Gaussian noise to add
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    data_tensor : numpy.ndarray
        Generated data tensor
    metadata : dict
        Metadata about the generated data
    """

    np.random.seed(seed)

    # Generate orthogonal matrices
    print(f"Generating orthogonal matrices...")
    all_matrices = []
    matrix_info = []

    for dim, count in matrix_specs:
        print(f"Generating {count} {dim}-dimensional orthogonal matrices...")

        for i in range(count):
            # Generate random dim x dim orthogonal matrix
            Q_dim = ortho_group.rvs(dim)

            # Embed in embedding_dim-dimensional space (pad with zeros)
            Q_embedded = np.zeros((embedding_dim, dim))
            Q_embedded[:dim, :] = Q_dim

            # Apply random rotation in R^embedding_dim
            rotation_matrix = ortho_group.rvs(embedding_dim)
            Q_rotated = rotation_matrix @ Q_embedded

            all_matrices.append(Q_rotated)
            matrix_info.append((dim, i + 1))
            print(f"  Matrix {i+1}: shape {Q_rotated.shape}")

    # Calculate total samples: dimension samples per matrix
    n_samples_total = sum(Q.shape[1] for Q in all_matrices)
    D = embedding_dim  # All matrices are embedded in embedding_dim-dimensional space

    print(f"Total matrices: {len(all_matrices)}")
    print(f"Total samples: {n_samples_total} (dimension samples per matrix)")
    print(f"Generating {n_replicates} replicates...")

    # Generate data tensor: (n_replicates, D, n_samples_total)
    data_tensor = np.zeros((n_replicates, D, n_samples_total))

    for replicate in range(n_replicates):
        sample_idx = 0

        # Generate samples from each matrix
        for i, Q in enumerate(all_matrices):
            # Generate samples equal to the dimension of this matrix
            n_samples_this_matrix = Q.shape[1]
            # Use the orthonormal matrix directly as samples
            samples = Q

            # Add noise if specified
            if noise_level > 0:
                noise = np.random.normal(0, noise_level, samples.shape)
                samples += noise

            data_tensor[
                replicate, :, sample_idx : sample_idx + n_samples_this_matrix
            ] = samples
            sample_idx += n_samples_this_matrix

    print(f"Generated data tensor shape: {data_tensor.shape}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save data tensor
    tensor_file = os.path.join(output_dir, "orthogonal_data_tensor.npy")
    np.save(tensor_file, data_tensor)
    print(f"Data tensor saved to {tensor_file}")

    # Save metadata
    metadata = {
        "dataset_name": "orthogonal_simulated",
        "matrix_specs": matrix_specs,
        "n_matrices_total": len(all_matrices),
        "n_samples_total": n_samples_total,
        "n_replicates": n_replicates,
        "dimension": D,
        "noise_level": noise_level,
        "seed": seed,
        "data_file": tensor_file,
        "description": f"Simulated data with orthogonal matrices: {matrix_specs}, all embedded in {embedding_dim}D space, dimension samples per matrix",
    }

    metadata_file = os.path.join(output_dir, "orthogonal_data_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_file}")

    return data_tensor, metadata


@click.command()
@click.option(
    "--output-dir",
    default="data/orthogonal",
    help="Output directory for generated data",
)
@click.option(
    "--matrix-specs",
    default="[(8,10),(10,1)]",
    help="List of (dimension, count) tuples as string, e.g., '[(8,10),(10,1)]' (default: [(8,10),(10,1)])",
)
@click.option(
    "--R",
    type=int,
    default=25,
    help="Number of replicates to generate (default: 25)",
)
@click.option(
    "--B",
    type=int,
    default=20,
    help="Embedding dimension for all matrices (default: 20)",
)
@click.option(
    "--noise-level",
    type=float,
    default=0.0,
    help="Standard deviation of Gaussian noise to add (default: 0.0)",
)
@click.option(
    "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
)
def main(
    output_dir,
    matrix_specs,
    r,
    b,
    noise_level,
    seed,
):
    """Generate simulated data with orthogonal matrices"""

    # Parse matrix_specs string to list of tuples
    try:
        matrix_specs_list = eval(matrix_specs)
        if not isinstance(matrix_specs_list, list):
            raise ValueError("matrix_specs must be a list")
        for spec in matrix_specs_list:
            if not isinstance(spec, tuple) or len(spec) != 2:
                raise ValueError("Each spec must be a tuple of (dimension, count)")
    except Exception as e:
        print(f"Error parsing matrix_specs: {e}")
        print("Expected format: '[(8,10),(10,1)]'")
        return

    # Generate data
    data_tensor, metadata = generate_orthogonal_data(
        output_dir=output_dir,
        matrix_specs=matrix_specs_list,
        n_replicates=r,
        embedding_dim=b,
        noise_level=noise_level,
        seed=seed,
    )

    print(f"Generated data tensor shape: {data_tensor.shape}")
    print(f"Matrix specifications: {matrix_specs_list}")
    print(f"Total matrices: {metadata['n_matrices_total']}")
    print(f"Total samples: {metadata['n_samples_total']}")
    print(f"Replicates: {metadata['n_replicates']}")


if __name__ == "__main__":
    main()
