#!/usr/bin/env python3
"""Histogram of support cardinalities from experiment results."""

import argparse
import json
import os
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-file", required=True, help="Path to results JSON")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--bin-width", type=int, default=1, help="Histogram bin width (use 25 for ethanol)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.results_file) as f:
        data = json.load(f)
    results_df = pd.DataFrame(data["replicates"])

    cardinalities = results_df["support_cardinality"]
    min_val, max_val = int(cardinalities.min()), int(cardinalities.max())
    bins = range(min_val, max_val + args.bin_width + 1, args.bin_width)

    plt.figure(figsize=(10, 7))
    plt.hist(
        cardinalities,
        bins=bins,
        color="#8B008B",
        edgecolor="white",
        align="left" if args.bin_width == 1 else "mid",
    )
    plt.xlabel(r"$|\widehat{S_{IP}}|$", fontsize=30)
    plt.ylabel("Frequency", fontsize=24)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    dataset_name = os.path.basename(args.results_file).split("_")[0]
    output_file = os.path.join(args.output_dir, f"{dataset_name}_cardinality_histogram.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
