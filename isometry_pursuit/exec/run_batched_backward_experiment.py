#!/usr/bin/env python3
"""Compare rank-safe batched deletion with backward greedy and IP."""

import argparse
import json
import os

from isometry_pursuit.batched_backward_experiment import (
    plot_batched_backward,
    run_batched_backward_experiment,
    summarize_batched_backward,
)


def _conditions(value):
    parsed = []
    for item in value.split(","):
        try:
            parsed.append(tuple(int(part) for part in item.split(":")))
        except ValueError as exc:
            raise argparse.ArgumentTypeError("conditions must use D:d_corr:num_blocks") from exc
    if any(len(item) != 3 or min(item) < 1 for item in parsed):
        raise argparse.ArgumentTypeError("conditions must use positive D:d_corr:num_blocks triples")
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditions", type=_conditions,
        default=_conditions("4:3:16,8:7:16,16:15:8"),
        help="Comma-separated D:d_corr:num_blocks triples.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=95173)
    parser.add_argument("--ip-iterations", type=int, default=100000)
    parser.add_argument("--output-dir", default="results/batched_backward")
    parser.add_argument("--figure-dir", default="figures/batched_backward")
    args = parser.parse_args()
    if args.repeats < 1 or args.ip_iterations < 1:
        parser.error("--repeats and --ip-iterations must be positive")

    records = run_batched_backward_experiment(
        args.conditions,
        repeats=args.repeats,
        seed=args.seed,
        ip_iterations=args.ip_iterations,
    )
    summary = summarize_batched_backward(records)
    os.makedirs(args.output_dir, exist_ok=True)
    records_path = os.path.join(args.output_dir, "runs.csv")
    summary_path = os.path.join(args.output_dir, "summary.csv")
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    figure_path = os.path.join(args.figure_dir, "batched_backward_comparison.png")
    records.to_csv(records_path, index=False)
    summary.to_csv(summary_path, index=False)
    with open(metadata_path, "w") as handle:
        json.dump(
            {
                "conditions": args.conditions,
                "repeats": args.repeats,
                "seed": args.seed,
                "ip_iterations": args.ip_iterations,
                "batch_rule": (
                    "Rank candidates by stale one-deletion loss; accept in that order "
                    "only while the jointly retained matrix has maximal possible rank."
                ),
                "arithmetic_proxy": (
                    "D*q*min(D,q) per loss evaluation and per explicit rank check; "
                    "IP uses iterations*D^2*P."
                ),
            },
            handle,
            indent=2,
        )
    plot_batched_backward(summary, figure_path)
    print(records_path)
    print(summary_path)
    print(figure_path)


if __name__ == "__main__":
    main()
