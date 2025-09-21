#!/usr/bin/env python3
"""
Data loader for Wine dataset experiments.
Loads and preprocesses the Wine dataset for isometry pursuit experiments.
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

# Add parent directory to path to import isometry_pursuit modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def load_wine_data(output_dir="data", standardize=True, max_features=None):
    """
    Load and preprocess Wine dataset.

    Parameters:
    -----------
    output_dir : str
        Directory to save the processed data
    standardize : bool
        Whether to standardize the data
    max_features : int, optional
        Maximum number of features to use (if None, use all)

    Returns:
    --------
    data : pandas.DataFrame
        Processed Wine data
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

    return data, D_wine, P_wine


def main():
    parser = argparse.ArgumentParser(description="Load and preprocess Wine dataset")
    parser.add_argument(
        "--output-dir", default="data", help="Output directory for processed data"
    )
    parser.add_argument(
        "--no-standardize", action="store_true", help="Skip standardization"
    )
    parser.add_argument(
        "--max-features", type=int, help="Maximum number of features to use"
    )

    args = parser.parse_args()

    # Load data
    data, D, P = load_wine_data(
        output_dir=args.output_dir,
        standardize=not args.no_standardize,
        max_features=args.max_features,
    )

    print(f"Loaded Wine dataset: {P} samples, {D} features")
    print(f"Data shape: {data.shape}")


if __name__ == "__main__":
    main()
