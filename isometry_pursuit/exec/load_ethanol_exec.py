#!/usr/bin/env python3
"""
Data loader for Ethanol dataset experiments.
Loads and preprocesses the Ethanol dataset for isometry pursuit experiments.
"""

import argparse
import sys
import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.io
from tqdm import tqdm

from isometry_pursuit.vendor import HParams
from isometry_pursuit.atomic_geometry import (
    get_atoms3_full,
    get_atoms4_full,
    get_atoms_4,
)
from isometry_pursuit.manifold import local_pca
import itertools


def get_features(positions, atoms2, atoms3, atoms4):
    """
    Extract geometric features from positions.
    This is a simplified version of the get_features function from montlake.
    """
    from isometry_pursuit.atomic_geometry import get_D_feats_feats

    # For now, we'll use a simplified approach
    # In the full implementation, this would compute all geometric features
    natoms = positions.shape[0]

    # Compute basic geometric features
    features = []

    # Add interpoint distances if atoms2 is not empty
    if len(atoms2) > 0:
        for i, j in atoms2:
            dist = np.linalg.norm(positions[i] - positions[j])
            features.append(dist)

    # Add angles if atoms3 is not empty
    if len(atoms3) > 0:
        for i, j, k in atoms3:
            v1 = positions[j] - positions[i]
            v2 = positions[k] - positions[i]
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            features.append(angle)

    # Add torsions if atoms4 is not empty
    if len(atoms4) > 0:
        for i, j, k, l in atoms4:
            v1 = positions[j] - positions[i]
            v2 = positions[k] - positions[j]
            v3 = positions[l] - positions[k]

            # Compute torsion angle
            n1 = np.cross(v1, v2)
            n2 = np.cross(v2, v3)

            if np.linalg.norm(n1) > 1e-10 and np.linalg.norm(n2) > 1e-10:
                cos_torsion = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
                torsion = np.arccos(np.clip(cos_torsion, -1, 1))
                features.append(torsion)

    return np.array(features)


def data_stream_custom_range(indices):
    """
    Simple data stream function for custom range of indices.
    This is a simplified version from montlake.
    """
    for i in indices:
        yield i


def load_ethanol_data(
    config_path,
    raw_data_path,
    positions_path,
    indices_path,
    angles_path,
    output_dir="data",
    R=25,
    dimension=2,
    power=1.0,
    n_neighbors=252,
    seed=1234,
):
    """
    Load and preprocess Ethanol dataset following the exact pattern from Ethanol.ipynb.

    Parameters:
    -----------
    config_path : str
        Path to the ethanol configuration JSON file
    raw_data_path : str
        Path to the raw ethanol.mat file
    positions_path : str
        Path to the pre-computed positions.npy file
    indices_path : str
        Path to the pre-computed indices.npy file
    angles_path : str
        Path to the pre-computed angles.npy file
    output_dir : str
        Directory to save the processed data
    R : int
        Number of replicates (selected points) for the experiment
    dimension : int
        Target dimension for the experiment
    power : float
        Power parameter for transformation
    n_neighbors : int
        Number of neighbors for local PCA
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    data : pandas.DataFrame
        Processed Ethanol data
    D : int
        Number of features (dimension)
    P : int
        Number of samples
    """

    print("Loading Ethanol dataset...")

    # Set random seeds
    np.random.seed(seed)
    random.seed(seed)

    # Load configuration
    print(f"Loading configuration from {config_path}")
    with open(config_path) as f:
        config = json.load(f)

    hparams = HParams(**config)

    # Load pre-computed data files (following Ethanol.ipynb pattern)
    print(f"Loading positions from {positions_path}")
    positions = np.load(positions_path)

    print(f"Loading indices from {indices_path}")
    randomindices = np.load(indices_path)

    print(f"Loading angles from {angles_path}")
    data = np.load(angles_path)

    print(f"Loading raw data from {raw_data_path}")
    raw_data = scipy.io.loadmat(raw_data_path)

    # Verify data consistency (as in Ethanol.ipynb)
    assert (
        raw_data["R"][randomindices] == positions
    ).all(), "Data consistency check failed"

    # Extract configuration parameters
    d = hparams.d
    atoms2_feat = hparams.atoms2_feat
    atoms3_feat = hparams.atoms3_feat
    atoms4_feat = hparams.atoms4_feat
    atoms2_dict = hparams.atoms2_dict
    atoms3_dict = hparams.atoms3_dict
    atoms4_dict = hparams.atoms4_dict
    diagram = hparams.diagram

    ii = np.asarray(hparams.ii)
    jj = np.asarray(hparams.jj)

    # Load geometric features (following Ethanol.ipynb pattern)
    natoms = positions.shape[1]
    n = positions.shape[0]
    atoms2 = np.asarray(list(itertools.combinations(range(natoms), 2)))
    atoms2full = atoms2
    atoms3 = np.asarray(list(itertools.combinations(range(natoms), 3)))
    atoms4 = np.asarray(list(itertools.combinations(range(natoms), 4)))
    atoms3full = get_atoms3_full(atoms3)
    atoms4full = get_atoms4_full(atoms4)

    if atoms2_feat:
        atoms2_feats = atoms2full
    else:
        atoms2_feats = np.asarray([])

    if atoms3_feat:
        atoms3_feats = atoms3full
    else:
        atoms3_feats = np.asarray([])

    if atoms4_feat:
        atoms4_feats = atoms4full
    else:
        atoms4_feats = np.asarray([])

    atoms4_dicts = get_atoms_4(natoms, ii, jj)[0]
    if atoms2_dict:
        atoms2_dicts = atoms2full
    else:
        atoms2_dicts = np.asarray([])
    if atoms3_dict:
        atoms3_dicts = atoms3full
    else:
        atoms3_dicts = np.asarray([])

    print(f"Ethanol dataset: {n} samples, {natoms} atoms")
    print(f"Number of replicates: {R}")
    print(f"Target dimension: {dimension}")

    # Process the data following Ethanol.ipynb pattern
    p = atoms4full.shape[0]
    selected_points = np.random.choice(range(min(50000, n)), R, replace=False)

    print("Computing geometric features...")
    from isometry_pursuit.atomic_geometry import get_D_feats_feats

    dg_x = np.asarray(
        [
            get_D_feats_feats(
                positions[selected_points[i]],
                atoms2in=atoms2_feats,
                atoms3in=atoms3_feats,
                atoms4in=atoms4_feats,
                atoms2out=atoms2_dicts,
                atoms3out=atoms3_dicts,
                atoms4out=atoms4full,
            )
            for i in tqdm(range(R), desc="Computing geometric features")
        ]
    )

    print("Computing tangent spaces and projections...")
    dg_M = np.zeros((R, dimension, p))
    for i, sp in enumerate(tqdm(selected_points, desc="Computing tangent spaces")):
        tangent_bases = local_pca(data, sp, n_neighbors, dimension)
        dg_M[i] = np.einsum("b p, b d -> d p", dg_x[i], tangent_bases)

    # Convert to DataFrame format for compatibility with run_experiment.py
    # Each row represents one experiment run with the projected data
    processed_data = []
    for i in range(R):
        # Flatten the 2D projection to 1D for each sample
        row_data = dg_M[i].flatten()  # Shape: (dimension * p,)
        processed_data.append(row_data)

    # Create DataFrame
    columns = [f"feature_{j}" for j in range(dimension * p)]
    data_df = pd.DataFrame(processed_data, columns=columns)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save metadata
    metadata = {
        "dataset_name": "ethanol",
        "n_samples": R,
        "n_features": dimension * p,
        "original_n_samples": n,
        "n_atoms": natoms,
        "target_dimension": dimension,
        "power": power,
        "n_neighbors": n_neighbors,
        "seed": seed,
        "selected_points": selected_points.tolist(),
        "config_file": config_path,
        "raw_data_file": raw_data_path,
        "positions_file": positions_path,
        "indices_file": indices_path,
        "angles_file": angles_path,
    }

    metadata_file = os.path.join(output_dir, "ethanol_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_file}")

    # Generate ethanol data tensor
    print(f"Generating ethanol data tensor...")
    # The data_df has R rows (one per selected point) and dimension*p columns
    # We want a tensor of shape (dimension, p, R) where R are the replicates
    data_tensor = np.zeros((dimension, p, R))

    # Reshape each row from (dimension * p,) to (dimension, p)
    for i in range(R):
        sample_data = data_df.iloc[i].values  # Shape: (dimension * p,)
        data_tensor[:, :, i] = sample_data.reshape(dimension, p)

    print(f"Generated ethanol data tensor shape: {data_tensor.shape}")

    # Save ethanol data tensor
    tensor_file = os.path.join(output_dir, "ethanol_tensor.npy")
    np.save(tensor_file, data_tensor)
    print(f"Ethanol data tensor saved to {tensor_file}")

    return data_tensor, data_df, dimension * p, R


def main():
    parser = argparse.ArgumentParser(description="Load and preprocess Ethanol dataset")
    parser.add_argument(
        "--config-path", required=True, help="Path to ethanol configuration JSON file"
    )
    parser.add_argument(
        "--raw-data-path", required=True, help="Path to raw ethanol.mat file"
    )
    parser.add_argument(
        "--positions-path", required=True, help="Path to positions.npy file"
    )
    parser.add_argument(
        "--indices-path", required=True, help="Path to indices.npy file"
    )
    parser.add_argument("--angles-path", required=True, help="Path to angles.npy file")
    parser.add_argument(
        "--output-dir", default="data", help="Output directory for processed data"
    )
    parser.add_argument(
        "--R", type=int, default=25, help="Number of replicates"
    )
    parser.add_argument("--dimension", type=int, default=2, help="Target dimension")
    parser.add_argument(
        "--power", type=float, default=1.0, help="Power parameter for transformation"
    )
    parser.add_argument(
        "--n-neighbors", type=int, default=252, help="Number of neighbors for local PCA"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    args = parser.parse_args()

    # Check if input files exist
    for file_path in [
        args.config_path,
        args.raw_data_path,
        args.positions_path,
        args.indices_path,
        args.angles_path,
    ]:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            print("Please ensure you have the required ethanol data files:")
            print("- ethanol.mat (raw data)")
            print("- positions.npy (pre-computed positions)")
            print("- indices.npy (pre-computed indices)")
            print("- angles.npy (pre-computed angles)")
            print("- ethanol_config.json (configuration file)")
            sys.exit(1)

    # Load data
    data_tensor, data, D, P = load_ethanol_data(
        config_path=args.config_path,
        raw_data_path=args.raw_data_path,
        positions_path=args.positions_path,
        indices_path=args.indices_path,
        angles_path=args.angles_path,
        output_dir=args.output_dir,
        R=args.R,
        dimension=args.dimension,
        power=args.power,
        n_neighbors=args.n_neighbors,
        seed=args.seed,
    )

    print(f"Loaded Ethanol dataset: {P} samples, {D} features")
    print(f"Original data shape: {data.shape}")
    print(f"Data tensor shape: {data_tensor.shape}")


if __name__ == "__main__":
    main()
