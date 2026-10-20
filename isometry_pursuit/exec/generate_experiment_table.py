#!/usr/bin/env python3
"""Generate LaTeX table rows from experiment result JSON files."""

import argparse
import json
import os

import numpy as np
from scipy.stats import ttest_rel


REQUIRED_REPLICATE_FIELDS = {
    "support_cardinality",
    "forward_backward_gl_norm",
    "forward_backward_gl_bp_norm",
}


def _parse_result_spec(spec):
    if ":" not in spec:
        path = spec
        label = os.path.basename(path).replace("_results.json", "").title()
        return path, label
    path, label = spec.rsplit(":", 1)
    if not path or not label:
        raise ValueError(f"Invalid result spec: {spec}")
    return path, label


def _format_float(value, precision=2):
    if not np.isfinite(value):
        return "--"
    if value == 0:
        return "0"
    text = f"{value:.{precision}g}"
    if "." in text:
        return text.rstrip("0").rstrip(".")
    return text


def _format_mean_sd(values):
    values = np.asarray(values, dtype=float)
    mean = np.mean(values)
    sd = np.std(values, ddof=0)
    return f"{_format_float(mean)} $\\pm$ {_format_float(sd)}"


def _format_probability(value):
    if value < 1e-3 and value > 0:
        return f"{value:.1e}"
    return _format_float(value)


def _load_result(path):
    with open(path) as handle:
        data = json.load(handle)
    if "metadata" not in data or "replicates" not in data:
        raise ValueError(f"{path} must contain 'metadata' and 'replicates'")
    return data


def _validate_result(path, data):
    shape = data["metadata"].get("data_tensor_shape")
    if shape is None or len(shape) != 3:
        raise ValueError(f"{path} metadata must include data_tensor_shape = [D, P, R]")

    replicates = data["replicates"]
    if len(replicates) != shape[2]:
        raise ValueError(
            f"{path} has {len(replicates)} replicates, but metadata R={shape[2]}"
        )
    if not replicates:
        raise ValueError(f"{path} contains no replicates")

    missing_by_index = {}
    for index, replicate in enumerate(replicates):
        missing = sorted(REQUIRED_REPLICATE_FIELDS - set(replicate))
        if missing:
            missing_by_index[index] = missing
    if missing_by_index:
        examples = ", ".join(
            f"replicate {index}: {missing}" for index, missing in list(missing_by_index.items())[:3]
        )
        raise ValueError(
            f"{path} is missing fields needed for the table ({examples}). "
            "Rerun isometry_pursuit.exec.run_experiment with the current code."
        )


def summarize_result(path, label, tie_tolerance):
    data = _load_result(path)
    _validate_result(path, data)

    D, P, R = [int(value) for value in data["metadata"]["data_tensor_shape"]]
    replicates = data["replicates"]

    support_cardinalities = np.array(
        [replicate["support_cardinality"] for replicate in replicates], dtype=float
    )
    foba_losses = np.array(
        [replicate["forward_backward_gl_norm"] for replicate in replicates], dtype=float
    )
    ip_foba_losses = np.array(
        [replicate["forward_backward_gl_bp_norm"] for replicate in replicates], dtype=float
    )

    valid = np.isfinite(foba_losses) & np.isfinite(ip_foba_losses)
    if not np.all(valid):
        raise ValueError(f"{path} contains non-finite FOBA or IP-FOBA losses")
    if len(foba_losses) < 2:
        raise ValueError(f"{path} needs at least two replicates for a paired t-test")

    greater = foba_losses > ip_foba_losses + tie_tolerance
    equal = np.isclose(foba_losses, ip_foba_losses, atol=tie_tolerance, rtol=0.0)
    p_value = ttest_rel(foba_losses, ip_foba_losses, alternative="greater").pvalue

    return {
        "label": label,
        "D": D,
        "P": P,
        "R": R,
        "foba_loss": _format_mean_sd(foba_losses),
        "support_cardinality": _format_mean_sd(support_cardinalities),
        "ip_foba_loss": _format_mean_sd(ip_foba_losses),
        "prob_greater": _format_probability(float(np.mean(greater))),
        "prob_equal": _format_probability(float(np.mean(equal))),
        "p_value": _format_probability(float(p_value)),
    }


def row_to_latex(row):
    values = [
        row["label"],
        str(row["D"]),
        str(row["P"]),
        str(row["R"]),
        row["foba_loss"],
        row["support_cardinality"],
        row["ip_foba_loss"],
        row["prob_greater"],
        row["prob_equal"],
        row["p_value"],
    ]
    return " & ".join(values) + r" \\"


def write_rows(output_path, rows):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as handle:
        handle.write("\\newcommand{\\experimenttablerows}{%\n")
        for row in rows:
            handle.write(row_to_latex(row) + "\n")
        handle.write("}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX rows for the main experiment results table"
    )
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="Result specs as path:Label, e.g. results/wine_results.json:Wine",
    )
    parser.add_argument(
        "--output",
        default="archival/generated/experiment_table_rows.tex",
        help="Output .tex rows file",
    )
    parser.add_argument(
        "--tie-tolerance",
        type=float,
        default=1e-8,
        help="Absolute tolerance for counting equal losses",
    )
    args = parser.parse_args()

    rows = [
        summarize_result(path, label, args.tie_tolerance)
        for path, label in (_parse_result_spec(spec) for spec in args.results)
    ]
    write_rows(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
