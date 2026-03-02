#!/usr/bin/env python3
"""Run isometry pursuit experiments on a pre-computed data tensor."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from isometry_pursuit.experiment import run_tensor_experiment


def main():
    parser = argparse.ArgumentParser(description="Run isometry pursuit experiment")
    parser.add_argument("--data-file", required=True, help="Path to .npy data tensor")
    parser.add_argument("--output-path", required=True, help="Path for output JSON")
    parser.add_argument(
        "--constraint-violation", type=float, default=0.0,
        help="Constraint violation tolerance (default: 0.0)"
    )
    args = parser.parse_args()

    if not args.data_file.endswith(".npy"):
        print(f"Error: expected a .npy file, got: {args.data_file}")
        return

    print(f"Loading {args.data_file}")
    data_tensor = np.load(args.data_file)
    print(f"Data tensor shape: {data_tensor.shape}")

    replicates, _ = run_tensor_experiment(
        data_tensor=data_tensor,
        constraint_violation=args.constraint_violation,
    )

    output = {
        "metadata": {
            "data_file": args.data_file,
            "constraint_violation": args.constraint_violation,
            "data_tensor_shape": list(data_tensor.shape),
        },
        "replicates": replicates,
    }

    output_dir = str(Path(args.output_path).parent) or "."
    os.makedirs(output_dir, exist_ok=True)

    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
