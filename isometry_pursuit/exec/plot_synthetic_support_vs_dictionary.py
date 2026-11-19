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
            "results/cloud/full-synthetic-20261116-001/records.json"
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

    real_runs = []
    for dataset in ("iris", "wine", "ethanol"):
        source = Path("results/cloud/large-zero-smoke-001/records.json") if dataset == "ethanol" else Path("results/cloud/zero-start-smoke-001/records.json")
        payload = json.loads(source.read_text())
        saved = payload["runs"] if isinstance(payload, dict) else payload
        selected = [r for r in saved if r.get("dataset") == dataset
                    and r.get("settings", {}).get("solver") == "scs"
                    and r["settings"].get("scale") == .1]
        if len(selected) != 1:
            raise ValueError(f"Expected one saved Lambda SCS scale-0.1 result for {dataset}")
        run = selected[0]
        capped = run["status"] != "solved"
        width, support = run["shape"][1], run["support_cardinality"]
        ax.scatter(width, support, marker="*", s=180, facecolors="white" if capped else "black",
                   edgecolors="black", linewidths=1, zorder=6)
        offset = (-5, 8) if dataset == "ethanol" else (5, -12)
        ax.annotate(dataset.title(), (width, support), xytext=offset, textcoords="offset points",
                    ha="right" if dataset == "ethanol" else "left", fontsize=9, zorder=7)
        real_runs.append({"dataset": dataset, "D": run["shape"][0], "P": width,
                          "support_cardinality": support, "status": run["status"],
                          "data_sha256": run["data_sha256"], "job_id": run["job_id"],
                          "source": str(source), "settings": run["settings"]})
    Path("results/figure5_real_lambda_overlay.json").write_text(json.dumps(real_runs, indent=2) + "\n")

    upper = max(dictionary_sizes.max(), support_sizes.max())
    no_trimming_line, = ax.plot(
        [2, upper], [2, upper], linestyle="--", color="0.35", linewidth=1.3,
        label=r"No trimming: $|\hat S|=P$",
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel(r"Dictionary size $P$")
    ax.set_ylabel(r"Isometry Pursuit support size $|\hat S|$")
    ax.grid(alpha=0.22)
    line_legend = ax.legend(
        handles=[no_trimming_line, Line2D([], [], marker="*", color="black", linestyle="none",
                                         markersize=10, label="Real dataset")],
        frameon=False, loc="upper left"
    )

    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label(r"Corrupting dimension $d_c$")
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
        loc="center left",
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
