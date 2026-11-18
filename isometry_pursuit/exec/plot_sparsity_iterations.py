"""Plot the saved SCS iteration-cap study without rerunning optimization."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    frame = pd.read_csv("results/scs_primal_dual_convergence/runs.csv")
    frame = frame[frame["family"] == "real"].copy()
    frame["retained_fraction"] = frame["support_cardinality"] / frame["P"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), constrained_layout=True)
    for dataset, color in [("iris", "#54a24b"), ("wine", "#4c78a8"),
                           ("ethanol", "#e45756")]:
        group = frame[frame["dataset"] == dataset]
        summary = group.groupby("requested_cap").agg(
            retained=("retained_fraction", "median"),
            agreement=("support_exact_match", "mean"),
        )
        axes[0].plot(summary.index, summary["retained"], "o-",
                     color=color, label=dataset.title(), markersize=3)
        axes[1].plot(summary.index, summary["agreement"], "o-",
                     color=color, markersize=3)
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("SCS iteration limit")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel(r"Median retained fraction $|\widehat S|/P$")
    axes[1].set_ylabel("Exact support agreement")
    axes[0].legend(frameon=False)
    destination = Path("figures/scs_early_stop/sparsity_iterations")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix(".pdf"))
    fig.savefig(destination.with_suffix(".png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
