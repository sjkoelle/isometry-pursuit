#!/usr/bin/env python3
"""Benchmark arithmetic-work proxies against runtime on identity simulations."""

import argparse
import ast
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from isometry_pursuit.algorithm import backward_elimination, greedy, isometry_pursuit
from isometry_pursuit.exec.run_identity_experiment import generate_identity_instance
from isometry_pursuit.exec.scs_early_stop import _problem, _solve_restart
from isometry_pursuit.experiment import _gl_loss
from isometry_pursuit.transformation import exponential_transformation


STOPPING_WINDOW = 3
STOPPING_GAP_CV = 1e-2
STOPPING_NORMALIZED_GAP = 1e-6
STOPPING_NORMALIZED_FEASIBILITY = 1e-6


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


def benchmark_early_stopped_ip(stopping_diagnostics_csv, eps=1e-12):
    """Benchmark the complete IP path at each inferred per-replicate stop."""
    diagnostics = pd.read_csv(stopping_diagnostics_csv)
    stops = diagnostics[diagnostics["first_candidate_stop"].astype(bool)]
    records = []
    paths = {name.lower(): path for name, path in REAL_DATASETS.items()}
    for dataset, group in stops.groupby("dataset"):
        tensor = np.load(paths[dataset.lower()])
        d, p, available_replicates = tensor.shape
        for row in group.itertuples(index=False):
            replicate = int(row.replicate)
            cap = int(row.reported_iterations)
            matrix = tensor[:, :, replicate % available_replicates]
            start = time.perf_counter()
            transformed = exponential_transformation(matrix)
            parts = _problem(transformed)
            solved = _solve_restart(parts, transformed, cap, eps)
            runtime = time.perf_counter() - start
            iterations = int(solved["reported_iterations"])
            records.append(
                {
                    "dataset": dataset.title(),
                    "method": "isometry_pursuit",
                    "D": d,
                    "P": p,
                    "benchmark_replicate": replicate,
                    "iterations": iterations,
                    "arithmetic_work_proxy": iterations * d * d * p,
                    "runtime_seconds": runtime,
                }
            )
        print(f"early-stopped IP: {dataset.title()} ({len(group)} replicates)")
    return pd.DataFrame(records)


def summarize_early_stopped_ip(records):
    return (
        records.groupby(["dataset", "method", "D", "P"], as_index=False)
        .agg(
            repeats=("runtime_seconds", "size"),
            min_iterations=("iterations", "min"),
            median_iterations=("iterations", "median"),
            max_iterations=("iterations", "max"),
            arithmetic_work_proxy=("arithmetic_work_proxy", "median"),
            median_runtime_seconds=("runtime_seconds", "median"),
            min_runtime_seconds=("runtime_seconds", "min"),
            max_runtime_seconds=("runtime_seconds", "max"),
        )
    )


def analyze_identity_early_stops(runs_csv, window=STOPPING_WINDOW):
    """Apply the task-aware stopping rule to saved identity cap sweeps."""
    frame = pd.read_csv(runs_csv)
    analyzed = []
    for (_, _), group in frame.groupby(["trial_id", "replicate"]):
        group = group.sort_values("reported_iterations").copy().reset_index(drop=True)
        gap = group["duality_gap"].astype(float)
        rolling = gap.rolling(window)
        rolling_std = rolling.apply(
            lambda values: np.std(values, ddof=1), raw=True
        )
        group["gap_cv"] = rolling_std / rolling.mean().abs().clip(
            lower=np.finfo(float).tiny
        )
        # These older synthetic sweeps predate recording SCS's separate pobj
        # and dobj.  The recorded primal objective supplies the scale here.
        group["normalized_gap"] = gap / (
            1.0 + 2.0 * group["objective"].abs()
        )
        group["normalized_feasibility"] = group["reconstruction_fro_raw"] / (
            1.0 + np.sqrt(group["D"].astype(float))
        )
        supports = group["support"].map(
            lambda value: frozenset(ast.literal_eval(value))
        )
        downstream = group["downstream_indices"].map(
            lambda value: frozenset(ast.literal_eval(value))
        )
        group["support_stable"] = [
            i >= window - 1 and len(set(supports.iloc[i - window + 1:i + 1])) == 1
            for i in range(len(group))
        ]
        group["downstream_stable"] = [
            i >= window - 1
            and len(set(downstream.iloc[i - window + 1:i + 1])) == 1
            for i in range(len(group))
        ]
        group["candidate_stop"] = (
            (group["normalized_gap"] <= STOPPING_NORMALIZED_GAP)
            & (group["gap_cv"] <= STOPPING_GAP_CV)
            & (group["normalized_feasibility"] <= STOPPING_NORMALIZED_FEASIBILITY)
            & group["support_stable"]
            & group["downstream_stable"]
            & group["downstream_valid"].astype(bool)
        )
        positions = np.flatnonzero(group["candidate_stop"].to_numpy())
        group["first_candidate_stop"] = False
        if len(positions):
            group.loc[int(positions[0]), "first_candidate_stop"] = True
        analyzed.append(group)
    diagnostics = pd.concat(analyzed, ignore_index=True)

    per_run = []
    maximum_cap = int(diagnostics["requested_cap"].max())
    for (trial_id, replicate), group in diagnostics.groupby(
        ["trial_id", "replicate"]
    ):
        stop = group[group["first_candidate_stop"]]
        triggered = not stop.empty
        chosen = stop.iloc[0] if triggered else group.loc[group["requested_cap"].idxmax()]
        per_run.append(
            {
                "trial_id": int(trial_id),
                "replicate": int(replicate),
                "D": int(chosen.D),
                "P": int(chosen.P),
                "corrupting_D": int(chosen.corrupting_D),
                "num_corrupting_matrices": int(chosen.num_corrupting_matrices),
                "triggered": triggered,
                "stop_iterations": (
                    int(chosen.reported_iterations) if triggered else maximum_cap
                ),
                "support_exact_match": bool(chosen.support_exact_match),
                "downstream_exact_match": bool(chosen.downstream_exact_match),
            }
        )
    per_run = pd.DataFrame(per_run)
    condition_caps = (
        per_run.groupby(
            ["trial_id", "D", "P", "corrupting_D", "num_corrupting_matrices"],
            as_index=False,
        )
        .agg(
            replicates=("replicate", "size"),
            triggered_replicates=("triggered", "sum"),
            stop_iterations=("stop_iterations", "max"),
            matching_supports=("support_exact_match", "sum"),
            matching_downstream=("downstream_exact_match", "sum"),
        )
    )
    return diagnostics, per_run, condition_caps


def benchmark_early_stopped_identity(condition_caps, seed):
    """Benchmark one matched identity instance per condition at its stop cap."""
    records = []
    conditions = condition_caps.sort_values(
        ["D", "corrupting_D", "num_corrupting_matrices"]
    ).reset_index(drop=True)
    for condition_index, row in enumerate(conditions.itertuples(index=False)):
        rng = np.random.default_rng(seed + condition_index)
        matrix, _, _, _ = generate_identity_instance(
            dimension=int(row.D),
            corrupting_dimension=int(row.corrupting_D),
            num_corrupting=int(row.num_corrupting_matrices),
            rng=rng,
        )
        start = time.perf_counter()
        transformed = exponential_transformation(matrix)
        parts = _problem(transformed)
        solved = _solve_restart(
            parts, transformed, int(row.stop_iterations), 1e-12
        )
        runtime = time.perf_counter() - start
        iterations = int(solved["reported_iterations"])
        records.append(
            {
                "method": "isometry_pursuit",
                "D": int(row.D),
                "P": int(row.P),
                "corrupting_D": int(row.corrupting_D),
                "num_corrupting_matrices": int(row.num_corrupting_matrices),
                "benchmark_replicate": 0,
                "iterations": iterations,
                "arithmetic_work_proxy": iterations * int(row.D) ** 2 * int(row.P),
                "runtime_seconds": runtime,
            }
        )
        print(
            f"early identity {condition_index + 1}/{len(conditions)}: "
            f"D={int(row.D)}, P={int(row.P)}, cap={int(row.stop_iterations)}"
        )
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


def plot_overlay(
    summary,
    output_path,
    real_summary=None,
    early_stopped_summary=None,
    title="Arithmetic-work proxies versus observed runtime",
):
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
    if early_stopped_summary is not None:
        stopped_label_offsets = {
            "Wine": (7, -15),
            "Iris": (7, 7),
            "Ethanol": (-78, 10),
        }
        for row in early_stopped_summary.itertuples(index=False):
            axis.scatter(
                row.arithmetic_work_proxy,
                row.median_runtime_seconds,
                color=METHOD_COLORS["isometry_pursuit"],
                marker="X",
                edgecolor="black",
                linewidth=0.7,
                s=105,
                zorder=7,
            )
            axis.annotate(
                f"{row.dataset} stopped",
                (row.arithmetic_work_proxy, row.median_runtime_seconds),
                xytext=stopped_label_offsets.get(row.dataset, (5, -11)),
                textcoords="offset points",
                fontsize=7,
            )
            if real_summary is not None:
                original = real_summary[
                    (real_summary["dataset"] == row.dataset)
                    & (real_summary["method"] == "isometry_pursuit")
                ]
                if len(original) == 1:
                    original = original.iloc[0]
                    axis.plot(
                        [original.arithmetic_work_proxy, row.arithmetic_work_proxy],
                        [original.median_runtime_seconds, row.median_runtime_seconds],
                        color=METHOD_COLORS["isometry_pursuit"],
                        linestyle="--",
                        linewidth=1.0,
                        alpha=0.65,
                        zorder=3,
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
    if early_stopped_summary is not None:
        dimension_handles.append(
            plt.Line2D(
                [],
                [],
                color=METHOD_COLORS["isometry_pursuit"],
                marker="X",
                markeredgecolor="black",
                linestyle="none",
                markersize=8,
                label="Early-stopped IP median",
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
    axis.set_title(title)
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
    parser.add_argument(
        "--early-stop-records-csv",
        help="Add existing full-path early-stopped IP benchmark records.",
    )
    parser.add_argument(
        "--benchmark-early-stop",
        action="store_true",
        help="Benchmark IP at stops in the saved stopping-rule diagnostics.",
    )
    parser.add_argument(
        "--stopping-diagnostics-csv",
        default="results/scs_primal_dual_convergence/stopping_rule_diagnostics.csv",
    )
    parser.add_argument(
        "--identity-early-stop-runs",
        default="results/scs_early_stop/synthetic_full/runs.csv",
        help="Saved identity cap sweep used to infer per-condition stop caps.",
    )
    parser.add_argument(
        "--identity-early-stop-records-csv",
        help="Add existing early-stopped identity benchmark records.",
    )
    parser.add_argument(
        "--benchmark-identity-early-stop",
        action="store_true",
        help="Analyze the identity cap sweep and benchmark every condition.",
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
    overlay_no_early_path = os.path.join(
        args.figure_dir, "complexity_vs_runtime_overlay_no_early_stopping.png"
    )
    overlay_with_early_path = os.path.join(
        args.figure_dir, "complexity_vs_runtime_overlay_with_early_stopping.png"
    )
    real_records_path = os.path.join(
        args.output_dir, "complexity_runtime_real_runs.csv"
    )
    real_summary_path = os.path.join(
        args.output_dir, "complexity_runtime_real_summary.csv"
    )
    early_stop_records_path = os.path.join(
        args.output_dir, "complexity_runtime_early_stopped_ip_runs.csv"
    )
    early_stop_summary_path = os.path.join(
        args.output_dir, "complexity_runtime_early_stopped_ip_summary.csv"
    )
    identity_diagnostics_path = os.path.join(
        args.output_dir, "identity_early_stop_diagnostics.csv"
    )
    identity_per_run_path = os.path.join(
        args.output_dir, "identity_early_stop_per_run.csv"
    )
    identity_condition_caps_path = os.path.join(
        args.output_dir, "identity_early_stop_condition_caps.csv"
    )
    identity_early_records_path = os.path.join(
        args.output_dir, "complexity_runtime_identity_early_stopped_runs.csv"
    )
    identity_early_summary_path = os.path.join(
        args.output_dir, "complexity_runtime_identity_early_stopped_summary.csv"
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
    early_stopped_summary = None
    if args.early_stop_records_csv or args.benchmark_early_stop:
        if args.early_stop_records_csv:
            early_stop_records = pd.read_csv(args.early_stop_records_csv)
        else:
            early_stop_records = benchmark_early_stopped_ip(
                args.stopping_diagnostics_csv
            )
            early_stop_records.to_csv(early_stop_records_path, index=False)
        early_stopped_summary = summarize_early_stopped_ip(early_stop_records)
        early_stopped_summary.to_csv(early_stop_summary_path, index=False)
    identity_early_summary = None
    if args.identity_early_stop_records_csv or args.benchmark_identity_early_stop:
        diagnostics, identity_per_run, condition_caps = analyze_identity_early_stops(
            args.identity_early_stop_runs
        )
        diagnostics.to_csv(identity_diagnostics_path, index=False)
        identity_per_run.to_csv(identity_per_run_path, index=False)
        condition_caps.to_csv(identity_condition_caps_path, index=False)
        if args.identity_early_stop_records_csv:
            identity_early_records = pd.read_csv(
                args.identity_early_stop_records_csv
            )
        else:
            identity_early_records = benchmark_early_stopped_identity(
                condition_caps, args.seed
            )
            identity_early_records.to_csv(identity_early_records_path, index=False)
        identity_early_summary = summarize(identity_early_records)
        identity_early_summary.to_csv(identity_early_summary_path, index=False)
    plot(summary, figure_path)
    plot_overlay(
        summary,
        overlay_no_early_path,
        real_summary=real_summary,
        title="Runtime without early stopping (100k-iteration IP)",
    )
    if identity_early_summary is not None:
        with_early_summary = pd.concat(
            [
                summary[summary["method"] != "isometry_pursuit"],
                identity_early_summary,
            ],
            ignore_index=True,
            sort=False,
        )
        with_early_real = real_summary
        if real_summary is not None and early_stopped_summary is not None:
            with_early_real = pd.concat(
                [
                    real_summary[real_summary["method"] != "isometry_pursuit"],
                    early_stopped_summary,
                ],
                ignore_index=True,
                sort=False,
            )
        plot_overlay(
            with_early_summary,
            overlay_with_early_path,
            real_summary=with_early_real,
            title="Runtime with task-aware early stopping",
        )
        plot_overlay(
            with_early_summary,
            overlay_path,
            real_summary=with_early_real,
            title="Runtime with task-aware early stopping",
        )
    else:
        plot_overlay(
            summary,
            overlay_path,
            real_summary=real_summary,
            early_stopped_summary=early_stopped_summary,
        )
    print(f"runs: {records_path if not args.records_csv else args.records_csv}")
    print(f"summary: {summary_path}")
    print(f"figure: {figure_path}")
    print(f"overlay figure: {overlay_path}")
    if real_summary is not None:
        print(f"real summary: {real_summary_path}")
    if early_stopped_summary is not None:
        print(f"early-stop summary: {early_stop_summary_path}")
    if identity_early_summary is not None:
        print(f"identity early-stop summary: {identity_early_summary_path}")
        print(f"without early stopping: {overlay_no_early_path}")
        print(f"with early stopping: {overlay_with_early_path}")


if __name__ == "__main__":
    main()
