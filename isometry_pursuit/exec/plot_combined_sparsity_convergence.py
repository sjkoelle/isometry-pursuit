#!/usr/bin/env python3
import argparse, os, re, sys, pickle
import numpy as np
import matplotlib.pyplot as plt

THRESH = 1e-4  # = 10e-5


def process_logs(convergence_logs, iters=None):
    I, S = [], []
    for log in convergence_logs or []:
        if not all(
            k in (log or {}) for k in ("coefficient_norms", "iterations", "sparsities")
        ):
            continue
        norms = np.asarray(log["coefficient_norms"])
        its = np.asarray(log["iterations"])
        sps = np.asarray(log["sparsities"])
        m = sps >= 0
        if not np.any(m):
            continue
        norms, its = norms[m], its[m]
        P = norms.shape[1]
        nz = (norms > THRESH).sum(1) / float(P)
        if iters:
            for t in iters:
                j = int(np.argmin(np.abs(its - t)))
                if abs(its[j] - t) <= max(1, 0.1 * t):
                    I.append(its[j])
                    S.append(nz[j])
        else:
            I += its.tolist()
            S += nz.tolist()
    if not I:
        return None, None
    o = np.argsort(I)
    return np.asarray(I)[o], np.asarray(S)[o]


import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from isometry_pursuit.plot import get_dataset_color_mapping


def plot_box(dataset_data, datasets, title, ylabel, out, overlay=True, width=0.6):
    """
    Plot boxplots per-iteration for each dataset key WITHOUT mixing values across keys.
    Logs each key as it's plotted.

    dataset_data: {key: {"iterations": np.array, "sparsities": np.array}}
    datasets: list[str]
    overlay: True -> draw all keys centered at same x; False -> side-by-side with offsets
    width: base box width (will be narrowed if side-by-side)
    """
    # Collect and sort all unique iteration values for consistent x-axis
    all_its = sorted(
        {int(i) for d in datasets for i in np.asarray(dataset_data[d]["iterations"])}
    )
    if not all_its:
        print("Nothing to plot.")
        sys.exit(1)

    K = len(datasets)
    N = len(all_its)

    # Colors per dataset - use consistent color mapping with plot_difference_boxplot
    color_mapping = get_dataset_color_mapping(datasets)
    colors = [color_mapping[dataset] for dataset in datasets]

    # Positioning
    x_base = (
        np.arange(N) + 1
    )  # boxplot positions are 1-based-ish, but any numeric is fine
    if overlay:
        offsets = [0.0 for _ in range(K)]
        box_width = width
    else:
        # spread keys around each iteration center
        spread = 0.7
        if K > 1:
            offsets = np.linspace(-spread / 2, spread / 2, K)
        else:
            offsets = [0.0]
        box_width = min(width / max(K, 1), 0.25)

    plt.figure(figsize=(12, 7))
    legends = []

    # Plot each dataset separately at its own positions, no mixing of values
    for k, key in enumerate(datasets):
        its = np.asarray(dataset_data[key]["iterations"])
        sps = np.asarray(dataset_data[key]["sparsities"])
        if its.shape[0] != sps.shape[0]:
            raise ValueError(
                f"Length mismatch for '{key}': iterations={its.shape[0]} vs sparsities={sps.shape[0]}"
            )

        # Build per-iteration lists for THIS key only
        data_k, pos_k = [], []
        for j, it in enumerate(all_its):
            m = its == it
            vals = sps[m]
            # Log as we go
            print(f"[{key}] it={it}: n={vals.size}")
            if vals.size > 0:
                data_k.append(vals.tolist())
                pos_k.append(x_base[j] + offsets[k])

        if not data_k:
            continue

        bp = plt.boxplot(
            data_k,
            positions=pos_k,
            widths=box_width,
            patch_artist=True,
            manage_ticks=False,
            whis=1.5,
            showfliers=True,
        )

        # Style - make lines match fill color
        color = colors[k % len(colors)]
        for b in bp["boxes"]:
            b.set_facecolor(color)
            b.set_alpha(0.7)  # Match the other plot's alpha
            b.set_edgecolor(color)  # Make edge color match fill color
            b.set_linewidth(1.5)
        for med in bp["medians"]:
            med.set_color(color)  # Make median color match fill color
            med.set_linewidth(2)
        for whisk in bp["whiskers"]:
            whisk.set_color(color)  # Make whisker color match fill color
            whisk.set_linewidth(1.5)
        for cap in bp["caps"]:
            cap.set_color(color)  # Make cap color match fill color
            cap.set_linewidth(1.5)

        legends.append(
            Patch(
                facecolor=color,
                edgecolor=color,  # Make legend edge color match fill color
                alpha=0.7,  # Match the plot alpha
                label=key,
            )
        )

    # Axes / labels
    plt.xticks(x_base, [str(it) for it in all_its], rotation=45, fontsize=20)
    plt.xlabel("Iterations", fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    plt.title(title, fontsize=28, fontweight="bold", pad=20)
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3)
    if legends:
        plt.legend(
            handles=legends,
            title="Dataset",
            frameon=True,
            fontsize=20,
            title_fontsize=20,
        )
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Combined sparsity convergence boxplots")
    ap.add_argument(
        "--convergence-file",
        action="append",
        required=True,
        help="Repeat for each .pkl (e.g., --convergence-file A.pkl --convergence-file B.pkl)",
    )
    ap.add_argument(
        "--dataset-names",
        nargs="*",
        help="Optional names matching the number of --convergence-file entries",
    )
    ap.add_argument(
        "--output-file", default="combined_sparsity_convergence_boxplot.png"
    )
    ap.add_argument("--title", default="Sparsity Convergence Across Datasets")
    ap.add_argument("--ylabel", default=r"$|\widehat{S}| / P$")
    ap.add_argument("--iteration-points", nargs="*", type=int)
    a = ap.parse_args()

    files = a.convergence_file  # already a list, one per flag
    if a.dataset_names and len(a.dataset_names) != len(files):
        raise ValueError(
            "dataset-names count must equal number of --convergence-file flags"
        )

    names = a.dataset_names or [
        (
            re.match(r"(.+)_convergence_all_replicates\.pkl", os.path.basename(f))
            or [None, os.path.splitext(os.path.basename(f))[0]]
        )[1]
        for f in files
    ]

    print("Files parsed:", files)
    print("Dataset names:", names)

    dataset_data, valid = {}, []
    for f, n in zip(files, names):
        try:
            with open(f, "rb") as fh:
                logs = pickle.load(fh)
            I, S = process_logs(logs, a.iteration_points)
            if I is not None:
                dataset_data[n] = {"iterations": I, "sparsities": S}
                valid.append(n)
            else:
                print(f"No valid iterations in {f}")
        except Exception as e:
            print(f"Skip {f}: {e}")

    if not valid:
        print("No valid data.")
        sys.exit(1)
    plot_box(dataset_data, valid, a.title, a.ylabel, a.output_file)
    print(f"Saved: {a.output_file} | Datasets: {', '.join(valid)}")


if __name__ == "__main__":
    main()
