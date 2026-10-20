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
    parser.add_argument(
        "--iterations", type=int, default=None,
        help="Max solver iterations (default: None, use eps stopping criterion)"
    )
    parser.add_argument(
        "--eps", type=float, default=None,
        help="Solver convergence tolerance (default: 1e-12)"
    )
    parser.add_argument(
        "--run-restricted-bp-backward",
        action="store_true",
        help="Also run backward elimination from the IP support scored by restricted basis pursuit solves",
    )
    parser.add_argument(
        "--restricted-bp-backward-iterations",
        type=int,
        default=None,
        help="SCS iteration cap for each restricted BP backward candidate solve",
    )
    parser.add_argument(
        "--restricted-bp-backward-eps",
        type=float,
        default=None,
        help="Tolerance for each restricted BP backward candidate solve; incompatible with an iteration cap",
    )
    parser.add_argument(
        "--restricted-bp-backward-reconstruction-tol",
        type=float,
        default=1e-3,
        help="Reconstruction tolerance for accepting restricted BP backward candidate solves",
    )
    parser.add_argument(
        "--restricted-bp-backward-max-replicates",
        type=int,
        default=None,
        help="Only run restricted BP backward on the first N replicates",
    )
    parser.add_argument(
        "--verbose-restricted-bp-backward",
        action="store_true",
        help="Print restricted BP backward pruning progress",
    )
    args = parser.parse_args()

    restricted_bp_backward_iterations = args.restricted_bp_backward_iterations
    if (
        restricted_bp_backward_iterations is None
        and args.restricted_bp_backward_eps is None
    ):
        restricted_bp_backward_iterations = 250

    if (
        restricted_bp_backward_iterations is not None
        and args.restricted_bp_backward_eps is not None
    ):
        parser.error(
            "--restricted-bp-backward-iterations and "
            "--restricted-bp-backward-eps cannot both be set"
        )

    if not args.data_file.endswith(".npy"):
        print(f"Error: expected a .npy file, got: {args.data_file}")
        return

    print(f"Loading {args.data_file}")
    data_tensor = np.load(args.data_file)
    print(f"Data tensor shape: {data_tensor.shape}")

    replicates, _ = run_tensor_experiment(
        data_tensor=data_tensor,
        constraint_violation=args.constraint_violation,
        iterations=args.iterations,
        eps=args.eps,
        run_restricted_bp_backward=args.run_restricted_bp_backward,
        restricted_bp_backward_iterations=restricted_bp_backward_iterations,
        restricted_bp_backward_eps=args.restricted_bp_backward_eps,
        restricted_bp_backward_reconstruction_tol=(
            args.restricted_bp_backward_reconstruction_tol
        ),
        restricted_bp_backward_max_replicates=(
            args.restricted_bp_backward_max_replicates
        ),
        verbose_restricted_bp_backward=args.verbose_restricted_bp_backward,
    )

    output = {
        "metadata": {
            "data_file": args.data_file,
            "constraint_violation": args.constraint_violation,
            "iterations": args.iterations,
            "eps": args.eps,
            "run_restricted_bp_backward": args.run_restricted_bp_backward,
            "restricted_bp_backward_iterations": restricted_bp_backward_iterations,
            "restricted_bp_backward_eps": args.restricted_bp_backward_eps,
            "restricted_bp_backward_reconstruction_tol": (
                args.restricted_bp_backward_reconstruction_tol
            ),
            "restricted_bp_backward_max_replicates": (
                args.restricted_bp_backward_max_replicates
            ),
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
