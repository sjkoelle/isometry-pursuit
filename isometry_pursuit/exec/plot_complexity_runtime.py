#!/usr/bin/env python3
"""Benchmark arithmetic-work proxies against runtime on identity simulations."""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from isometry_pursuit.algorithm import backward_elimination, greedy, isometry_pursuit
from isometry_pursuit.exec.run_identity_experiment import generate_identity_instance
from isometry_pursuit.experiment import _gl_loss


METHOD_LABELS = {
    "forward_greedy": "Forward greedy",
    "isometry_pursuit": "Isometry Pursuit (SCS)",
    "backward_greedy": "Backward greedy",
}

METHOD_COLORS = {
    "forward_greedy": "#4c78a8",
    "isometry_pursuit": "#f58518",
    "backward_greedy": "#54a24b",
}

DIMENSION_MARKERS = {2: "o", 4: "s", 8: "^", 16: "D"}

REAL_DATASETS = {
    "Iris": "data/iris_standardized_resampled_tensor.npy",
    "Wine": "data/wine_standardized_resampled_tensor.npy",
    "Ethanol": "data/ethanol_tensor.npy",
}


def forward_proxy(dimension, columns):
    return sum(
        (columns - q + 1) * dimension * q * q
        for q in range(1, dimension + 1)
    )


def backward_proxy(dimension, columns):
    return dimension * dimension * sum(
        k * (k - 1) for k in range(dimension + 1, columns + 1)
    )


def benchmark(input_csv, repeats, seed):
    conditions = (
        pd.read_csv(input_csv)[
            ["D", "corrupting_D", "num_corrupting_matrices", "total_columns"]
        ]
        .drop_duplicates()
        .sort_values(["D", "corrupting_D", "num_corrupting_matrices"])
    )
    records = []
    for condition_index, row in enumerate(conditions.itertuples(index=False)):
        d = int(row.D)
        p = int(row.total_columns)
        for replicate in range(repeats):
            rng = np.random.default_rng(seed + condition_index * repeats + replicate)
            matrix, _, _, _ = generate_identity_instance(
                dimension=d,
                corrupting_dimension=int(row.corrupting_D),
                num_corrupting=int(row.num_corrupting_matrices),
                rng=rng,
            )
            method_calls = (
                ("forward_greedy", lambda: greedy(matrix, _gl_loss, target_dimension=d)),
                ("isometry_pursuit", lambda: isometry_pursuit(matrix)),
                (
                    "backward_greedy",
                    lambda: backward_elimination(matrix, _gl_loss, target_dimension=d),
                ),
            )
            for method, call in method_calls:
                start = time.perf_counter()
                result = call()
                runtime = time.perf_counter() - start
                if method == "isometry_pursuit":
                    iterations = int(result["basis_pursuit_iterations"])
                    proxy = iterations * d * d * p
                elif method == "forward_greedy":
                    iterations = d
                    proxy = forward_proxy(d, p)
                else:
                    iterations = p - d
                    proxy = backward_proxy(d, p)
                records.append(
                    {
                        "method": method,
                        "D": d,
                        "P": p,
                        "corrupting_D": int(row.corrupting_D),
                        "num_corrupting_matrices": int(row.num_corrupting_matrices),
                        "benchmark_replicate": replicate,
                        "iterations": iterations,
                        "arithmetic_work_proxy": proxy,
                        "runtime_seconds": runtime,
                    }
                )
            print(
                f"condition {condition_index + 1}/{len(conditions)}: "
                f"D={d}, P={p}, d_corr={int(row.corrupting_D)}"
            )
    return pd.DataFrame(records)


def benchmark_real_datasets(repeats):
    records = []
    for dataset, input_path in REAL_DATASETS.items():
        tensor = np.load(input_path)
        d, p, available_replicates = tensor.shape
        for replicate in range(repeats):
            matrix = tensor[:, :, replicate % available_replicates]
            method_calls = (
                ("forward_greedy", lambda: greedy(matrix, _gl_loss, target_dimension=d)),
                ("isometry_pursuit", lambda: isometry_pursuit(matrix)),
                (
                    "backward_greedy",
                    lambda: backward_elimination(matrix, _gl_loss, target_dimension=d),
                ),
            )
            for method, call in method_calls:
                start = time.perf_counter()
                result = call()
                runtime = time.perf_counter() - start
                if method == "isometry_pursuit":
                    iterations = int(result["basis_pursuit_iterations"])
                    proxy = iterations * d * d * p
                elif method == "forward_greedy":
                    iterations = d
                    proxy = forward_proxy(d, p)
                else:
                    iterations = p - d
                    proxy = backward_proxy(d, p)
                records.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "D": d,
                        "P": p,
                        "benchmark_replicate": replicate,
                        "iterations": iterations,
                        "arithmetic_work_proxy": proxy,
                        "runtime_seconds": runtime,
                    }
                )
            print(f"real dataset: {dataset} ({replicate + 1}/{repeats})")
    return pd.DataFrame(records)


def summarize(records):
    keys = [
        "method",
        "D",
        "P",
        "corrupting_D",
        "num_corrupting_matrices",
        "iterations",
        "arithmetic_work_proxy",
    ]
    return (
        records.groupby(keys, as_index=False)
        .agg(
            repeats=("runtime_seconds", "size"),
            median_runtime_seconds=("runtime_seconds", "median"),
            min_runtime_seconds=("runtime_seconds", "min"),
            max_runtime_seconds=("runtime_seconds", "max"),
        )
    )


def plot(summary, output_path):
    methods = list(METHOD_LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5))
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.18, top=0.82, wspace=0.16)
    colors = {2: "#4c78a8", 4: "#f58518", 8: "#54a24b", 16: "#e45756"}
    for axis, method in zip(axes, methods):
        subset = summary[summary["method"] == method]
        for dimension, group in subset.groupby("D"):
            axis.scatter(
                group["arithmetic_work_proxy"],
                group["median_runtime_seconds"],
                color=colors.get(int(dimension)),
                label=f"D={int(dimension)}",
                alpha=0.8,
                s=30,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.2)
        axis.set_title(METHOD_LABELS[method])
        axis.set_xlabel("Arithmetic-work proxy")
    axes[0].set_ylabel("Median wall-clock time (s)")
    axes[-1].legend(title="Dimension", fontsize=8)
    fig.suptitle("Arithmetic-work proxies versus observed runtime")
    fig.text(
        0.5,
        0.025,
        "Pure initializers only; the common fixed-cardinality swap refinement is omitted.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_overlay(summary, output_path, real_summary=None):
    fig, axis = plt.subplots(figsize=(8.2, 5.8))
    fig.subplots_adjust(bottom=0.17)
    for method, label in METHOD_LABELS.items():
        subset = summary[summary["method"] == method]
        for dimension, group in subset.groupby("D"):
            axis.scatter(
                group["arithmetic_work_proxy"],
                group["median_runtime_seconds"],
                color=METHOD_COLORS[method],
                marker=DIMENSION_MARKERS.get(int(dimension), "o"),
                alpha=0.78,
                s=38,
            )
    if real_summary is not None:
        for row in real_summary.itertuples(index=False):
            axis.scatter(
                row.arithmetic_work_proxy,
                row.median_runtime_seconds,
                color=METHOD_COLORS[row.method],
                marker="*",
                edgecolor="black",
                linewidth=0.6,
                s=145,
                zorder=5,
            )
            axis.annotate(
                row.dataset,
                (row.arithmetic_work_proxy, row.median_runtime_seconds),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
            )

    method_handles = [
        plt.Line2D(
            [],
            [],
            color=METHOD_COLORS[method],
            marker="o",
            linestyle="none",
            label=label,
        )
        for method, label in METHOD_LABELS.items()
    ]
    dimension_handles = [
        plt.Line2D(
            [],
            [],
            color="0.35",
            marker=marker,
            linestyle="none",
            label=f"D={dimension}",
        )
        for dimension, marker in DIMENSION_MARKERS.items()
    ]
    if real_summary is not None:
        dimension_handles.append(
            plt.Line2D(
                [],
                [],
                color="0.35",
                marker="*",
                markeredgecolor="black",
                linestyle="none",
                markersize=10,
                label="Real dataset",
            )
        )
    method_legend = axis.legend(handles=method_handles, title="Method", loc="upper left")
    axis.add_artist(method_legend)
    axis.legend(handles=dimension_handles, title="Dimension", loc="lower right")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.2)
    axis.set_xlabel("Arithmetic-work proxy")
    axis.set_ylabel("Median wall-clock time (s)")
    axis.set_title("Arithmetic-work proxies versus observed runtime")
    fig.text(
        0.5,
        0.025,
        "Pure initializers only; the common fixed-cardinality swap refinement is omitted.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        default="results/identity_swap_comparison_full/identity_experiment_runs.csv",
        help="Identity experiment runs used to define the simulation sizes.",
    )
    parser.add_argument("--output-dir", default="results/complexity_runtime")
    parser.add_argument("--figure-dir", default="figures/complexity_runtime")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=77123)
    parser.add_argument(
        "--records-csv",
        help="Plot existing benchmark records instead of running benchmarks.",
    )
    parser.add_argument(
        "--real-records-csv",
        help="Add existing pure-method real-dataset benchmark records.",
    )
    parser.add_argument(
        "--benchmark-real",
        action="store_true",
        help="Benchmark pure methods on Wine, Iris, and Ethanol and add them.",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)
    records_path = os.path.join(args.output_dir, "complexity_runtime_runs.csv")
    summary_path = os.path.join(args.output_dir, "complexity_runtime_summary.csv")
    figure_path = os.path.join(args.figure_dir, "complexity_vs_runtime.png")
    overlay_path = os.path.join(
        args.figure_dir, "complexity_vs_runtime_overlay.png"
    )
    real_records_path = os.path.join(
        args.output_dir, "complexity_runtime_real_runs.csv"
    )
    real_summary_path = os.path.join(
        args.output_dir, "complexity_runtime_real_summary.csv"
    )

    if args.records_csv:
        records = pd.read_csv(args.records_csv)
    else:
        records = benchmark(args.input_csv, args.repeats, args.seed)
        records.to_csv(records_path, index=False)
    summary = summarize(records)
    summary.to_csv(summary_path, index=False)
    real_summary = None
    if args.real_records_csv or args.benchmark_real:
        if args.real_records_csv:
            real_records = pd.read_csv(args.real_records_csv)
        else:
            real_records = benchmark_real_datasets(args.repeats)
            real_records.to_csv(real_records_path, index=False)
        real_summary = (
            real_records.groupby(
                [
                    "dataset",
                    "method",
                    "D",
                    "P",
                    "iterations",
                    "arithmetic_work_proxy",
                ],
                as_index=False,
            )
            .agg(
                repeats=("runtime_seconds", "size"),
                median_runtime_seconds=("runtime_seconds", "median"),
                min_runtime_seconds=("runtime_seconds", "min"),
                max_runtime_seconds=("runtime_seconds", "max"),
            )
        )
        real_summary.to_csv(real_summary_path, index=False)
    plot(summary, figure_path)
    plot_overlay(summary, overlay_path, real_summary=real_summary)
    print(f"runs: {records_path if not args.records_csv else args.records_csv}")
    print(f"summary: {summary_path}")
    print(f"figure: {figure_path}")
    print(f"overlay figure: {overlay_path}")
    if real_summary is not None:
        print(f"real summary: {real_summary_path}")


if __name__ == "__main__":
    main()
