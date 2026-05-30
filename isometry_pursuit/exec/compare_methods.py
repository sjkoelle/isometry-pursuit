#!/usr/bin/env python3
"""
Compare convex (basis pursuit) vs greedy approach on experiment results.
"""

import argparse
import json
from isometry_pursuit.experiment import print_experiment_summary


def main():
    parser = argparse.ArgumentParser(description="Compare convex vs greedy results")
    parser.add_argument(
        "--results-file",
        required=True,
        help="Path to results JSON file from run_experiment"
    )
    args = parser.parse_args()

    with open(args.results_file) as f:
        results = json.load(f)

    print_experiment_summary(results)


if __name__ == "__main__":
    main()
