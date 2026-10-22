"""Plot synthetic method loss above the known ground-truth optimum."""

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


METHODS = (
    ("Greedy → swap", "forward_backward_full_loss"),
    ("Backward → swap", "backward_swap_full_loss"),
    ("IP → greedy → swap", "forward_backward_restricted_loss"),
    ("IP → backward → swap", "backward_swap_restricted_loss"),
)


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
        default=Path("figures/synthetic_excess_loss_vs_dictionary"),
    )
    args = parser.parse_args()

    runs = json.loads(args.input.read_text())["runs"]
    target_dimensions = np.asarray([run["D"] for run in runs], dtype=int)
    corrupting_dimensions = np.asarray(
        [run["corrupting_D"] for run in runs], dtype=int
    )
    dictionary_sizes = np.asarray([run["total_columns"] for run in runs], dtype=int)

    unique_corrupting_dimensions = np.unique(corrupting_dimensions)
    color_indices = np.searchsorted(
        unique_corrupting_dimensions, corrupting_dimensions
    )
    color_map = plt.get_cmap("viridis", len(unique_corrupting_dimensions))
    normalization = colors.BoundaryNorm(
        np.arange(-0.5, len(unique_corrupting_dimensions) + 0.5),
        len(unique_corrupting_dimensions),
    )

    # Draw large points first so observations with smaller D remain visible.
    order = np.argsort(target_dimensions)[::-1]
    fig, axes = plt.subplots(
        2, 2, figsize=(11.5, 8.4), sharex=True, sharey=True,
        layout="constrained",
    )
    scatter = None
    for ax, (title, loss_key) in zip(axes.flat, METHODS):
        losses = np.asarray([run[loss_key] for run in runs], dtype=float)
        excess_losses = losses - target_dimensions
        finite = np.isfinite(excess_losses)
        plot_order = order[finite[order]]
        scatter = ax.scatter(
            dictionary_sizes[plot_order],
            excess_losses[plot_order],
            s=18 + 2.5 * target_dimensions[plot_order],
            c=color_indices[plot_order],
            cmap=color_map,
            norm=normalization,
            alpha=0.45,
            linewidths=0.25,
            edgecolors="white",
        )
        ax.axhline(0, linestyle="--", color="0.35", linewidth=1.2)
        ax.set_title(title)
        ax.set_xscale("log", base=2)
        ax.set_yscale("symlog", linthresh=1e-4, linscale=0.7)
        ax.set_ylim(bottom=-3e-5)
        ax.grid(alpha=0.22)
        omitted = len(runs) - int(finite.sum())
        if omitted:
            ax.text(
                0.98, 0.04, f"{omitted:,} runs not computed",
                transform=ax.transAxes, ha="right", va="bottom",
                color="0.4", fontsize=9,
            )

    fig.supxlabel(r"Dictionary size $P=D+b d$")
    fig.supylabel(r"Excess loss over ground truth $L-D$")

    colorbar = fig.colorbar(scatter, ax=axes, pad=0.015)
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
    fig.legend(
        handles=size_handles,
        title=r"Target dimension $D$",
        loc="outside upper center",
        ncol=len(size_handles),
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.3,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=250)
    fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
