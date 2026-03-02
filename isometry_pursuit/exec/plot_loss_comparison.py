#!/usr/bin/env python3
"""Box plot of isometry loss across replicates."""

import argparse
import json
import os
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-file", required=True, help="Path to results JSON")
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.results_file) as f:
        data = json.load(f)
    results_df = pd.DataFrame(data["replicates"])

    dataset_name = os.path.basename(args.results_file).split("_")[0]

    fig, ax = plt.subplots(figsize=(6, 7))
    ax.boxplot(results_df["isometry_loss"], labels=[dataset_name])
    ax.set_ylabel(r"Isometry loss $\ell_1(\hat{S}_{IP})$", fontsize=16)
    ax.tick_params(labelsize=13)
    plt.tight_layout()

    output_file = os.path.join(args.output_dir, f"{dataset_name}_isometry_loss.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
