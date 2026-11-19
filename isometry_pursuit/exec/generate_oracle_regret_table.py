"""Generate the supplemental summary from the real-data swap comparisons."""

import json
from pathlib import Path

import numpy as np

from isometry_pursuit.swap_plot import METHODS


def main():
    datasets = ("iris", "wine", "ethanol")
    summaries = {}
    for dataset in datasets:
        records = json.loads(
            Path(f"results/{dataset}_swap_comparison.json").read_text()
        )["replicates"]
        losses = np.asarray([
            [record["methods"][method]["loss"] for method in METHODS]
            for record in records
        ])
        if losses.shape != (25, 4) or not np.isfinite(losses).all():
            raise ValueError(f"Unexpected or incomplete results for {dataset}")
        regrets = losses - losses.min(axis=1, keepdims=True)
        regrets[np.abs(regrets) <= 1e-9] = 0
        summaries[dataset] = {
            "replicates": len(records),
            "attainment_percent": (100 * (regrets == 0).mean(axis=0)).tolist(),
            "mean_regret": regrets.mean(axis=0).tolist(),
        }
    labels = (
        r"Forward $\rightarrow$ swap",
        r"Backward $\rightarrow$ swap",
        r"IP $\rightarrow$ forward $\rightarrow$ swap",
        r"IP $\rightarrow$ backward $\rightarrow$ swap",
    )
    rows = []
    for index, label in enumerate(labels):
        cells = [label]
        for dataset in datasets:
            summary = summaries[dataset]
            percent = summary["attainment_percent"][index]
            regret = summary["mean_regret"][index]
            cells.append(rf"${percent:.0f}\%\;/\;{regret:.3g}$")
        rows.append(" & ".join(cells) + r" \\")
    destination = Path("archival/generated/oracle_regret_rows.tex")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\\begin{tabular}{lccc}\n\\toprule\n"
        + r"Method & Iris & Wine & Ethanol \\" + "\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    Path("results/oracle_regret_summary.json").write_text(
        json.dumps({"methods": METHODS, "datasets": summaries}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
