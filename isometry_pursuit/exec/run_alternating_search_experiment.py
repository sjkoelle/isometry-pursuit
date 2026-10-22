#!/usr/bin/env python3
"""Run the paired expansion-contraction local-search pilot."""

import argparse
import json
import os

from isometry_pursuit.alternating_search_experiment import (
    plot_alternating_search,
    run_alternating_search_experiment,
    summarize_alternating_search,
)


def _conditions(value):
    try:
        result = [tuple(int(part) for part in item.split(":")) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("conditions must use D:d_corr:num_blocks") from exc
    if any(len(item) != 3 or min(item) < 1 for item in result):
        raise argparse.ArgumentTypeError("conditions must be positive D:d_corr:num_blocks triples")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditions", type=_conditions,
        default=_conditions("4:2:4,4:3:16,8:4:8,8:7:16,16:8:8,16:15:8"),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=71513)
    parser.add_argument("--max-cycles", type=int, default=4)
    parser.add_argument("--ip-iterations", type=int, default=100000)
    parser.add_argument("--output-dir", default="results/alternating_search")
    parser.add_argument("--figure-dir", default="figures/alternating_search")
    args = parser.parse_args()
    if min(args.repeats, args.max_cycles, args.ip_iterations) < 1:
        parser.error("repeats, max-cycles, and ip-iterations must be positive")

    records = run_alternating_search_experiment(
        args.conditions,
        repeats=args.repeats,
        seed=args.seed,
        max_cycles=args.max_cycles,
        ip_iterations=args.ip_iterations,
    )
    summary = summarize_alternating_search(records)
    os.makedirs(args.output_dir, exist_ok=True)
    records_path = os.path.join(args.output_dir, "runs.csv")
    summary_path = os.path.join(args.output_dir, "summary.csv")
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    figure_path = os.path.join(args.figure_dir, "alternating_search_comparison.png")
    records.to_csv(records_path, index=False)
    summary.to_csv(summary_path, index=False)
    with open(metadata_path, "w") as handle:
        json.dump(
            {
                "conditions": args.conditions,
                "repeats": args.repeats,
                "seed": args.seed,
                "max_cycles": args.max_cycles,
                "expansion_widths": [2, 4],
                "ip_iteration_cap": args.ip_iterations,
                "arithmetic_proxy": (
                    "D*q*min(D,q) per subset loss or explicit rank check; "
                    "IP contributes solver_iterations*D^2*P."
                ),
            },
            handle,
            indent=2,
        )
    plot_alternating_search(summary, figure_path)
    print(records_path)
    print(summary_path)
    print(figure_path)


if __name__ == "__main__":
    main()
