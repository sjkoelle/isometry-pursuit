#!/usr/bin/env python3
"""Run the paired beam look-ahead backward-deletion pilot."""

import argparse
import json

from isometry_pursuit.lookahead_backward_experiment import (
    run_lookahead_backward_experiment,
    save_lookahead_backward_experiment,
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
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--num-starts", type=int, default=8)
    parser.add_argument("--relative-tolerance", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--output-dir", default="results/lookahead_backward_pilot")
    parser.add_argument("--figure-dir", default="figures/lookahead_backward_pilot")
    args = parser.parse_args()
    records = run_lookahead_backward_experiment(
        args.conditions,
        replicates=args.replicates,
        depth=args.depth,
        beam_width=args.beam_width,
        num_starts=args.num_starts,
        relative_tolerance=args.relative_tolerance,
        seed=args.seed,
    )
    metadata = {
        "conditions": args.conditions,
        "replicates": args.replicates,
        "depth": args.depth,
        "beam_width": args.beam_width,
        "num_starts": args.num_starts,
        "relative_tolerance": args.relative_tolerance,
        "seed": args.seed,
        "lookahead_rule": (
            "At every accepted deletion, expand all finite immediate removals, "
            "retain the lowest-loss beam, expand to the requested depth, accept "
            "the first deletion on the lowest-loss terminal path, and replan."
        ),
        "arithmetic_proxy": (
            "Backward searches sum D^2*q for every evaluated D-by-q candidate. "
            "IP uses reported SCS iterations times D^2*P."
        ),
    }
    paths = save_lookahead_backward_experiment(
        records, args.output_dir, args.figure_dir, metadata
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
