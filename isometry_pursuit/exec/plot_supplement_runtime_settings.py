"""Expose solver parameters on the saved supplemental runtime comparison."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
import pandas as pd


def main():
    folder = Path("results/complexity_runtime")
    points = pd.read_csv(folder / "cloud_search_figure_points.csv")
    saved = json.loads(Path("results/cloud/large-grid-20261116-001/records.json").read_text())
    solvers = saved["runs"] if isinstance(saved, dict) else saved
    lookup = {}
    for result in solvers:
        method = "isometry_pursuit" if result["settings"]["solver"] == "scs" else "isometry_pursuit_admm"
        work = result["basis_pursuit_iterations"] * result["D"] ** 2 * result["shape"][1]
        key = (method, result["D"], int(work), round(result["basis_pursuit_time_seconds"], 8))
        if key in lookup:
            raise ValueError("Ambiguous solver provenance")
        lookup[key] = result
    points["parameter"] = float("nan")
    points["optimizer_job_id"] = ""
    for index, row in points.iterrows():
        if not row["method"].startswith("isometry"):
            continue
        if row["source"] == "Lambda":
            result = lookup[(row["method"], row["D"], int(row["work"]), round(row["seconds"], 8))]
            field = "scale" if result["settings"]["solver"] == "scs" else "rho"
            points.at[index, "parameter"] = result["settings"][field]
            points.at[index, "optimizer_job_id"] = result["job_id"]
        else:
            points.at[index, "parameter"] = .1
    parameters = [.01, .1, 1., 10., 100.]
    palettes = {"isometry_pursuit": ListedColormap(plt.get_cmap("Oranges")([.35, .48, .61, .74, .9])),
                "isometry_pursuit_admm": ListedColormap(plt.get_cmap("Purples")([.35, .48, .61, .74, .9]))}
    baseline = {"forward_greedy": "#4c78a8", "backward_greedy": "#54a24b"}
    markers = {2: "o", 4: "s", 8: "^", 16: "D", 32: "P", 64: "h"}
    fig, ax = plt.subplots(figsize=(9.2, 6.7))
    fig.subplots_adjust(left=.1, right=.98, top=.96, bottom=.24)
    for row in points.itertuples():
        color = baseline[row.method] if row.method in baseline else palettes[row.method](parameters.index(row.parameter))
        style = dict(facecolors="none", edgecolors=color) if row.capped else dict(color=color)
        ax.scatter(row.work, row.seconds, marker=markers[row.D], s=27, alpha=.48, linewidths=.7, **style)
    real = pd.read_csv(folder / "complexity_runtime_real_summary.csv")
    early = pd.read_csv(folder / "complexity_runtime_early_stopped_ip_summary.csv")
    real = pd.concat([real[real.method != "isometry_pursuit"], early], ignore_index=True)
    for row in real.itertuples():
        color = baseline.get(row.method, palettes["isometry_pursuit"](1))
        ax.scatter(row.arithmetic_work_proxy, row.median_runtime_seconds, marker="*", s=110,
                   color=color, edgecolors="black", linewidths=.6, zorder=5)
        ax.annotate(row.dataset, (row.arithmetic_work_proxy, row.median_runtime_seconds),
                    xytext=(4, 5), textcoords="offset points", fontsize=7)
    cutoff = 245.36574106800015
    ax.axhline(cutoff, color=baseline["backward_greedy"], linestyle="--", alpha=.8)
    ax.annotate("backward time limit", (.98, cutoff), xycoords=ax.get_yaxis_transform(),
                xytext=(0, 7), textcoords="offset points", ha="right", va="bottom", fontsize=8)
    labels = {"forward_greedy": "Forward greedy", "backward_greedy": "Backward greedy",
              "isometry_pursuit": "IP (SCS)", "isometry_pursuit_admm": "IP (ADMM)"}
    colors = {**baseline, **{method: palette(2) for method, palette in palettes.items()}}
    first = ax.legend(handles=[Line2D([], [], color=colors[method], marker="o", linestyle="none", label=label)
                               for method, label in labels.items()], loc="upper left", fontsize=8)
    ax.add_artist(first)
    ax.legend(handles=[Line2D([], [], color=".35", marker=marker, linestyle="none", label=f"D={dimension}")
                       for dimension, marker in markers.items()] +
                      [Line2D([], [], color=".35", marker="*", linestyle="none", label="Real data")],
              loc="lower right", fontsize=8)
    norm = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5, 4.5], 5)
    for left, method, title in [(.12, "isometry_pursuit", "SCS scale"), (.58, "isometry_pursuit_admm", r"ADMM $\rho$")]:
        bar_ax = fig.add_axes([left, .09, .34, .025])
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=palettes[method]), cax=bar_ax, orientation="horizontal")
        bar.set_ticks(range(5), labels=["0.01", "0.1", "1", "10", "100"])
        bar.set_label(title)
    ax.set(xscale="log", yscale="log", xlabel="Arithmetic-work proxy", ylabel="Wall-clock time (s)")
    ax.grid(which="both", alpha=.2)
    path = Path("figures/complexity_runtime/complexity_vs_runtime_settings")
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    points.to_csv(folder / "supplement_runtime_settings_points.csv", index=False)


if __name__ == "__main__":
    main()
