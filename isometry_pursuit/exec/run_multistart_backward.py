#!/usr/bin/env python3
"""Run the randomized multi-start backward-deletion pilot."""

import argparse
import json

from isometry_pursuit.multistart_backward_experiment import (
    run_multistart_backward_experiment,
    save_multistart_backward_experiment,
)


DEFAULT_CONDITIONS = [
    {"D": 2, "corrupting_D": 1, "num_corrupting_matrices": 16},
    {"D": 4, "corrupting_D": 3, "num_corrupting_matrices": 16},
    {"D": 8, "corrupting_D": 7, "num_corrupting_matrices": 8},
    {"D": 16, "corrupting_D": 15, "num_corrupting_matrices": 4},
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=json.loads, default=DEFAULT_CONDITIONS)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--num-starts", type=int, default=8)
    parser.add_argument("--relative-tolerance", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--output-dir", default="results/multistart_backward_pilot")
    parser.add_argument("--figure-dir", default="figures/multistart_backward_pilot")
    args = parser.parse_args()
    records = run_multistart_backward_experiment(
        args.conditions,
        replicates=args.replicates,
        num_starts=args.num_starts,
        relative_tolerance=args.relative_tolerance,
        seed=args.seed,
    )
    metadata = {
        "conditions": args.conditions,
        "replicates": args.replicates,
        "num_starts": args.num_starts,
        "relative_tolerance": args.relative_tolerance,
        "seed": args.seed,
        "randomization": (
            "Start zero uses exact greedy deletion with randomized exact-tie breaking. "
            "Later starts sample uniformly among finite removals whose loss is within "
            "relative_tolerance * abs(best_loss) of the best removal."
        ),
        "arithmetic_proxy": (
            "Backward methods: D^2 sum_{k=D+1}^P k(k-1), multiplied by starts. "
            "IP: reported SCS iterations times D^2 P."
        ),
    }
    paths = save_multistart_backward_experiment(
        records, args.output_dir, args.figure_dir, metadata
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
