#!/usr/bin/env python3
"""
Data loader for Wine dataset experiments.
Loads and preprocesses the Wine dataset for isometry pursuit experiments.
"""

import click
import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler


def load_wine_data(
    output_dir="data",
    standardize=True,
    max_features=None,
    n_replicates=25,
    frac=0.5,
    D=None,
):
    """
    Load and preprocess Wine dataset with resampling.

    Parameters:
    -----------
    output_dir : str
        Directory to save the processed data
    standardize : bool
        Whether to standardize the data
    max_features : int, optional
        Maximum number of features to use (if None, use all)
    n_replicates : int
        Number of resampling replicates to generate
    frac : float
        Fraction of data to use in each resampling
    D : int, optional
        Number of features to sample for each replicate (default: None, use all features)

    Returns:
    --------
    data_tensor : numpy.ndarray
        Resampled data tensor of shape (D, n_samples_per_replicate, n_replicates)
    data : pandas.DataFrame
        Original processed Wine data
    D : int
        Number of features (dimension)
    P : int
        Number of samples
    """

    print("Loading Wine dataset...")
    wine = load_wine()

    # Get basic info
    P_wine, D_wine_full = wine["data"].shape
    print(f"Wine dataset: {P_wine} samples, {D_wine_full} features")
    print(f"Features: {wine['feature_names']}")

    # Limit features if specified
    if max_features is not None:
        D_wine = min(max_features, D_wine_full)
        print(f"Using first {D_wine} features")
    else:
        D_wine = D_wine_full

    # Set D to use all features if not specified
    if D is None:
        D = D_wine

    # Create DataFrame with selected features
    wine_df = pd.DataFrame(
        data=wine["data"][:, :D_wine], columns=wine["feature_names"][:D_wine]
    )

    if standardize:
        print("Standardizing data...")
        scaler = StandardScaler()
        wine_standardized = pd.DataFrame(
            data=scaler.fit_transform(wine["data"][:, :D_wine]),
            columns=wine["feature_names"][:D_wine],
        )

        # Verify standardization (should be sqrt(P))
        norms = np.linalg.norm(wine_standardized, axis=0)
        print(f"Column norms after standardization: {norms}")

        data = wine_standardized
        data_name = "wine_standardized"
    else:
        data = wine_df
        data_name = "wine"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save data
    output_file = os.path.join(output_dir, f"{data_name}_{D_wine}.csv")
    data.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

    # Save metadata
    metadata = {
        "dataset_name": "wine",
        "standardized": standardize,
        "n_samples": P_wine,
        "n_features": D_wine,
        "n_features_available": D_wine_full,
        "feature_names": list(wine["feature_names"][:D_wine]),
        "target_names": list(wine["target_names"]),
        "data_file": output_file,
    }

    metadata_file = os.path.join(output_dir, f"{data_name}_{D_wine}_metadata.json")
    import json

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_file}")

    # Generate resampled data tensor
    print(f"Generating {n_replicates} resampled replicates with frac={frac}...")
    if D == D_wine:
        print(f"Using all {D} features for each replicate")
    else:
        print(
            f"Sampling {D} features for each replicate from {D_wine} available features"
        )
    n_samples_per_replicate = int(P_wine * frac)
    data_tensor = np.zeros((D, n_samples_per_replicate, n_replicates))

    for i in range(n_replicates):
        np.random.seed(i)
        # Sample rows (samples)
        random_row_indices = np.random.choice(
            range(P_wine), n_samples_per_replicate, replace=False
        )

        if D == D_wine:
            # Use all features (no feature sampling)
            sampled_data = data.to_numpy()[random_row_indices]
        else:
            # Sample columns (features)
            random_col_indices = np.random.choice(range(D_wine), D, replace=False)
            sampled_data = data.to_numpy()[random_row_indices][:, random_col_indices]

        # Transpose to get shape (D, n_samples_per_replicate)
        data_tensor[:, :, i] = sampled_data.T

    print(f"Generated data tensor shape: {data_tensor.shape}")

    # Save resampled data tensor
    tensor_file = os.path.join(output_dir, f"{data_name}_resampled_tensor.npy")
    np.save(tensor_file, data_tensor)
    print(f"Resampled data tensor saved to {tensor_file}")

    return data_tensor, data, D, P_wine


@click.command()
@click.option(
    "--output-dir", default="data", help="Output directory for processed data"
)
@click.option("--no-standardize", is_flag=True, help="Skip standardization")
@click.option("--max-features", type=int, help="Maximum number of features to use")
@click.option(
    "--D",
    type=int,
    default=None,
    help="Number of features to sample for each replicate (default: None, use all features)",
)
@click.option(
    "--R",
    type=int,
    default=25,
    help="Number of replicates to generate (default: 25)",
)
def main(
    output_dir,
    no_standardize,
    max_features,
    d,
    r,
):
    """Load and preprocess Wine dataset"""

    # Load data
    data_tensor, data, D, P = load_wine_data(
        output_dir=output_dir,
        standardize=not no_standardize,
        max_features=max_features,
        n_replicates=r,
        D=d,
    )

    print(f"Loaded Wine dataset: {P} samples, {D} features")
    print(f"Original data shape: {data.shape}")
    print(f"Resampled tensor shape: {data_tensor.shape}")


if __name__ == "__main__":
    main()
