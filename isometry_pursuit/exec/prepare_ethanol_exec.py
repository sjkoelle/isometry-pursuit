#!/usr/bin/env python3
"""
Prepare ethanol data from raw .mat file.
Extracts positions, computes planar angles, and saves positions.npy,
indices.npy, and angles.npy needed by load_ethanol_exec.
"""

import argparse
import itertools
import json
import random

import numpy as np
import scipy.io
from tqdm import tqdm

from isometry_pursuit.atomic_geometry import get_atoms3_full, get_features


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ethanol positions and angles from raw .mat file"
    )
    parser.add_argument(
        "--raw-data-path",
        default="data/ethanol.mat",
        help="Path to raw ethanol.mat file",
    )
    parser.add_argument(
        "--config-path",
        default="configs/ethanol/ethanol_config.json",
        help="Path to ethanol config JSON",
    )
    parser.add_argument(
        "--output-dir", default="data", help="Directory to save output .npy files"
    )
    parser.add_argument(
        "--n", type=int, default=50000, help="Number of conformations to subsample"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load raw data
    print(f"Loading {args.raw_data_path}...")
    raw = scipy.io.loadmat(args.raw_data_path)
    R = raw["R"]  # (555092, 9, 3)
    print(f"R shape: {R.shape}")

    # Load config to know which atom sets to use for angles
    with open(args.config_path) as f:
        config = json.load(f)

    atoms3_feat = config.get("atoms3_feat", True)

    # Subsample indices
    n_total = R.shape[0]
    n = min(args.n, n_total)
    randomindices = np.random.choice(n_total, n, replace=False)
    positions = R[randomindices]  # (n, 9, 3)

    natoms = positions.shape[1]
    atoms3 = np.asarray(list(itertools.combinations(range(natoms), 3)))
    atoms3full = get_atoms3_full(atoms3) if atoms3_feat else np.asarray([])
    atoms2_feats = np.asarray([])
    atoms4_feats = np.asarray([])

    print(f"Computing planar angles for {n} conformations ({atoms3full.shape[0]} angle features each)...")
    angles = np.zeros((n, len(atoms3full)))
    for i in tqdm(range(n)):
        _, planarangles, _ = get_features(
            positions[i], atoms2_feats, atoms3full, atoms4_feats
        )
        angles[i] = planarangles

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    pos_path = f"{args.output_dir}/positions.npy"
    idx_path = f"{args.output_dir}/indices.npy"
    ang_path = f"{args.output_dir}/angles.npy"

    np.save(pos_path, positions)
    np.save(idx_path, randomindices)
    np.save(ang_path, angles)

    print(f"Saved positions -> {pos_path}  {positions.shape}")
    print(f"Saved indices   -> {idx_path}  {randomindices.shape}")
    print(f"Saved angles    -> {ang_path}  {angles.shape}")


if __name__ == "__main__":
    main()
