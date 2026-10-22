"""Plot Isometry Pursuit support size against synthetic dictionary width."""

import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/identity_swap_comparison_25x_D64/"
            "identity_experiment_runs.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/synthetic_support_vs_dictionary"),
    )
    args = parser.parse_args()

    runs = json.loads(args.input.read_text())["runs"]
    target_dimensions = np.asarray([run["D"] for run in runs], dtype=int)
    corrupting_dimensions = np.asarray(
        [run["corrupting_D"] for run in runs], dtype=int
    )
    dictionary_sizes = np.asarray(
        [
            run["D"]
            + run["num_corrupting_matrices"] * run["corrupting_D"]
            for run in runs
        ],
        dtype=int,
    )
    support_sizes = np.asarray(
        [run["support_cardinality"] for run in runs], dtype=int
    )

    # Draw large points first so smaller-D observations remain visible.
    order = np.argsort(target_dimensions)[::-1]
    target_dimensions = target_dimensions[order]
    corrupting_dimensions = corrupting_dimensions[order]
    dictionary_sizes = dictionary_sizes[order]
    support_sizes = support_sizes[order]

    fig, ax = plt.subplots(figsize=(8.2, 6.4), layout="constrained")
    unique_corrupting_dimensions = np.unique(corrupting_dimensions)
    color_indices = np.searchsorted(
        unique_corrupting_dimensions, corrupting_dimensions
    )
    color_map = plt.get_cmap("viridis", len(unique_corrupting_dimensions))
    normalization = colors.BoundaryNorm(
        np.arange(-0.5, len(unique_corrupting_dimensions) + 0.5),
        len(unique_corrupting_dimensions),
    )
    scatter = ax.scatter(
        dictionary_sizes,
        support_sizes,
        s=18 + 2.5 * target_dimensions,
        c=color_indices,
        cmap=color_map,
        norm=normalization,
        alpha=0.48,
        linewidths=0.25,
        edgecolors="white",
    )

    upper = max(dictionary_sizes.max(), support_sizes.max())
    no_trimming_line, = ax.plot(
        [2, upper], [2, upper], linestyle="--", color="0.35", linewidth=1.3,
        label=r"No trimming: $|\hat S|=P$",
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel(r"Dictionary size $P=D+b d$")
    ax.set_ylabel(r"Isometry Pursuit support size $|\hat S|$")
    ax.grid(alpha=0.22)
    line_legend = ax.legend(
        handles=[no_trimming_line], frameon=False, loc="upper left"
    )

    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label(r"Corrupting dimension $d$")
    colorbar.set_ticks(np.arange(len(unique_corrupting_dimensions)))
    colorbar.set_ticklabels([str(value) for value in unique_corrupting_dimensions])

    size_handles = [
        Line2D(
            [], [], linestyle="none", marker="o",
            markersize=np.sqrt(18 + 2.5 * dimension),
            markerfacecolor="0.55", markeredgecolor="white", alpha=0.55,
            label=str(dimension),
        )
        for dimension in np.unique(target_dimensions)
    ]
    size_legend = ax.legend(
        handles=size_handles,
        title=r"Target dimension $D$",
        frameon=False,
        loc="lower right",
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.3,
    )
    ax.add_artist(line_legend)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=250)
    fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
