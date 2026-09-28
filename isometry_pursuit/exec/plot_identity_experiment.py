#!/usr/bin/env python3
"""Plot synthetic identity-block experiment success rates."""

import argparse
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


METRIC_LABELS = {
    "strict_success_rate": "Strict success rate",
    "contains_true_block_rate": "Contains true block rate",
}


def _as_bool_series(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False})


def load_identity_summary(results_dir=None, runs_csv=None, summary_csv=None):
    if runs_csv is None and summary_csv is None:
        if results_dir is None:
            results_dir = "results/identity_experiment"
        candidate_runs = os.path.join(results_dir, "identity_experiment_runs.csv")
        candidate_summary = os.path.join(results_dir, "identity_experiment_summary.csv")
        if os.path.exists(candidate_runs):
            runs_csv = candidate_runs
        elif os.path.exists(candidate_summary):
            summary_csv = candidate_summary
        else:
            raise FileNotFoundError(
                f"Could not find identity experiment CSVs in {results_dir}"
            )

    if runs_csv is not None:
        runs = pd.read_csv(runs_csv)
        required = {
            "D",
            "corrupting_D",
            "num_corrupting_matrices",
            "strict_success",
            "contains_true_block",
        }
        missing = sorted(required - set(runs.columns))
        if missing:
            raise ValueError(f"Missing columns in {runs_csv}: {missing}")

        runs = runs.copy()
        runs["strict_success"] = _as_bool_series(runs["strict_success"])
        runs["contains_true_block"] = _as_bool_series(runs["contains_true_block"])
        return (
            runs.groupby(["D", "corrupting_D", "num_corrupting_matrices"], as_index=False)
            .agg(
                runs=("strict_success", "count"),
                strict_success_rate=("strict_success", "mean"),
                contains_true_block_rate=("contains_true_block", "mean"),
            )
            .sort_values(["D", "corrupting_D", "num_corrupting_matrices"])
        )

    summary = pd.read_csv(summary_csv)
    required = {
        "D",
        "corrupting_D",
        "num_corrupting_matrices",
        "strict_success_rate",
        "contains_true_block_rate",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"Missing columns in {summary_csv}: {missing}")
    return summary.sort_values(["D", "corrupting_D", "num_corrupting_matrices"])


def plot_identity_summary(
    summary,
    output_file,
    metric="strict_success_rate",
    title=None,
    log_x=True,
):
    if metric not in METRIC_LABELS:
        raise ValueError(f"Unknown metric: {metric}")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    dimensions = sorted(summary["D"].unique())
    corrupting_dimensions = sorted(summary["corrupting_D"].unique())
    markers = ["o", "s", "^", "D", "P", "X", "v", "*"]
    marker_by_dimension = {
        dimension: markers[index % len(markers)]
        for index, dimension in enumerate(dimensions)
    }

    cmap = plt.get_cmap("tab10")
    color_by_corrupting_dimension = {
        dimension: cmap(index % 10)
        for index, dimension in enumerate(corrupting_dimensions)
    }

    fig, ax = plt.subplots(figsize=(9, 6))

    for (dimension, corrupting_dimension), group in summary.groupby(["D", "corrupting_D"]):
        group = group.sort_values("num_corrupting_matrices")
        ax.plot(
            group["num_corrupting_matrices"],
            group[metric],
            color=color_by_corrupting_dimension[corrupting_dimension],
            marker=marker_by_dimension[dimension] if len(dimensions) > 1 else "o",
            linewidth=1.8,
            markersize=7,
            alpha=0.9,
        )

    x_values = sorted(summary["num_corrupting_matrices"].unique())
    if log_x:
        ax.set_xscale("log", base=2)
    ax.set_xticks(x_values)
    ax.set_xticklabels([str(int(value)) for value in x_values])
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Number of corrupting matrices")
    ax.set_ylabel(METRIC_LABELS[metric])
    if title is not None:
        ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)

    corrupting_handles = [
        Line2D(
            [0],
            [0],
            color=color_by_corrupting_dimension[dimension],
            linewidth=2,
            label=f"d={dimension}",
        )
        for dimension in corrupting_dimensions
    ]

    if len(dimensions) > 1:
        dimension_handles = [
            Line2D(
                [0],
                [0],
                color="black",
                marker=marker_by_dimension[dimension],
                linestyle="None",
                markersize=7,
                label=f"D={dimension}",
            )
            for dimension in dimensions
        ]
        first_legend = ax.legend(
            handles=dimension_handles,
            title="Full dimension",
            loc="lower left",
            frameon=True,
        )
        ax.add_artist(first_legend)

    legend_location = "lower right" if len(dimensions) > 1 else "best"
    ax.legend(
        handles=corrupting_handles,
        title="Corrupting dimension",
        loc=legend_location,
        frameon=True,
        ncol=2 if len(corrupting_handles) > 4 else 1,
    )

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_file


def _dimension_output_file(output_dir, metric, dimension):
    return os.path.join(output_dir, f"identity_experiment_{metric}_D{int(dimension)}.png")


def _suffixed_output_file(output_file, dimension):
    root, ext = os.path.splitext(output_file)
    return f"{root}_D{int(dimension)}{ext or '.png'}"


def main():
    parser = argparse.ArgumentParser(
        description="Plot success rates from the synthetic identity-block experiment"
    )
    parser.add_argument(
        "--results-dir",
        default="results/identity_experiment",
        help="Directory containing identity_experiment_runs.csv or identity_experiment_summary.csv",
    )
    parser.add_argument("--runs-csv", default=None, help="Run-level CSV path")
    parser.add_argument("--summary-csv", default=None, help="Summary CSV path")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--output-file", default=None)
    parser.add_argument(
        "--metric",
        choices=sorted(METRIC_LABELS),
        default="strict_success_rate",
    )
    parser.add_argument("--title", default=None)
    parser.add_argument("--linear-x", action="store_true", help="Use a linear x-axis")
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Save one combined plot instead of one plot per full dimension",
    )
    args = parser.parse_args()

    summary = load_identity_summary(
        results_dir=args.results_dir,
        runs_csv=args.runs_csv,
        summary_csv=args.summary_csv,
    )
    if args.combined:
        output_file = args.output_file
        if output_file is None:
            output_file = os.path.join(args.output_dir, f"identity_experiment_{args.metric}.png")
        plot_identity_summary(
            summary=summary,
            output_file=output_file,
            metric=args.metric,
            title=args.title,
            log_x=not args.linear_x,
        )
        print(f"Saved to {output_file}")
        return

    for dimension in sorted(summary["D"].unique()):
        dimension_summary = summary[summary["D"] == dimension]
        if args.output_file is None:
            output_file = _dimension_output_file(args.output_dir, args.metric, dimension)
        else:
            output_file = _suffixed_output_file(args.output_file, dimension)
        title = args.title
        if title is None:
            title = f"D={int(dimension)}"
        plot_identity_summary(
            summary=dimension_summary,
            output_file=output_file,
            metric=args.metric,
            title=title,
            log_x=not args.linear_x,
        )
        print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
