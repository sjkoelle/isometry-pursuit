#!/usr/bin/env python3
"""
Data loader for Iris dataset experiments.
Loads and preprocesses the Iris dataset for isometry pursuit experiments.
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Add parent directory to path to import isometry_pursuit modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def load_iris_data(output_dir="data", standardize=True):
    """
    Load and preprocess Iris dataset.

    Parameters:
    -----------
    output_dir : str
        Directory to save the processed data
    standardize : bool
        Whether to standardize the data

    Returns:
    --------
    data : pandas.DataFrame
        Processed Iris data
    D : int
        Number of features (dimension)
    P : int
        Number of samples
    """

    print("Loading Iris dataset...")
    iris = load_iris()

    # Get basic info
    P_iris, D_iris = iris["data"].shape
    print(f"Iris dataset: {P_iris} samples, {D_iris} features")
    print(f"Features: {iris['feature_names']}")

    # Create DataFrame
    iris_df = pd.DataFrame(data=iris["data"], columns=iris["feature_names"])

    if standardize:
        print("Standardizing data...")
        scaler = StandardScaler()
        iris_standardized = pd.DataFrame(
            data=scaler.fit_transform(iris["data"]), columns=iris["feature_names"]
        )

        # Verify standardization (should be sqrt(P))
        norms = np.linalg.norm(iris_standardized, axis=0)
        print(f"Column norms after standardization: {norms}")

        data = iris_standardized
        data_name = "iris_standardized"
    else:
        data = iris_df
        data_name = "iris"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save data
    output_file = os.path.join(output_dir, f"{data_name}.csv")
    data.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

    # Save metadata
    metadata = {
        "dataset_name": "iris",
        "standardized": standardize,
        "n_samples": P_iris,
        "n_features": D_iris,
        "feature_names": list(iris["feature_names"]),
        "target_names": list(iris["target_names"]),
        "data_file": output_file,
    }

    metadata_file = os.path.join(output_dir, f"{data_name}_metadata.json")
    import json

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_file}")

    return data, D_iris, P_iris


def main():
    parser = argparse.ArgumentParser(description="Load and preprocess Iris dataset")
    parser.add_argument(
        "--output-dir", default="data", help="Output directory for processed data"
    )
    parser.add_argument(
        "--no-standardize", action="store_true", help="Skip standardization"
    )

    args = parser.parse_args()

    # Load data
    data, D, P = load_iris_data(
        output_dir=args.output_dir, standardize=not args.no_standardize
    )

    print(f"Loaded Iris dataset: {P} samples, {D} features")
    print(f"Data shape: {data.shape}")


if __name__ == "__main__":
    main()
