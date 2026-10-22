"""Paired synthetic pilot for beam look-ahead backward deletion."""

import contextlib
import io
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .algorithm import backward_elimination, isometry_pursuit
from .experiment import _gl_loss
from .lookahead_backward import lookahead_backward_deletion
from .multistart_backward import backward_work_counts, multistart_backward_deletion
from .multistart_backward_experiment import _identity_instance, _record


METHOD_LABELS = {
    "backward_greedy": "Backward greedy",
    "multistart_backward": "Multi-start",
    "lookahead_backward": "Depth-2 look-ahead",
    "isometry_pursuit": "Isometry Pursuit",
}


def run_lookahead_backward_experiment(
    conditions,
    replicates=3,
    depth=2,
    beam_width=8,
    num_starts=8,
    relative_tolerance=0.01,
    seed=20261019,
):
    """Run paired BG, multi-start, look-ahead, and IP comparisons."""
    records = []
    run_id = 0
    for condition_id, condition in enumerate(conditions):
        d = int(condition["D"])
        corrupting_d = int(condition["corrupting_D"])
        num_corrupting = int(condition["num_corrupting_matrices"])
        for replicate in range(replicates):
            run_seed = seed + run_id
            matrix, truth = _identity_instance(
                d, corrupting_d, num_corrupting, np.random.default_rng(run_seed)
            )
            p = matrix.shape[1]
            evaluations, backward_proxy = backward_work_counts(d, p, d)
            common = {
                "run_id": run_id,
                "condition_id": condition_id,
                "replicate": replicate,
                "seed": run_seed,
                "D": d,
                "P": p,
                "corrupting_D": corrupting_d,
                "num_corrupting_matrices": num_corrupting,
            }

            started = time.perf_counter()
            backward_indices = backward_elimination(
                matrix, _gl_loss, target_dimension=d, random_state=run_seed
            )
            records.append(
                {
                    **common,
                    **_record(
                        "backward_greedy",
                        backward_indices,
                        matrix,
                        truth,
                        time.perf_counter() - started,
                        evaluations,
                        backward_proxy,
                        solver_iterations=np.nan,
                        best_start=np.nan,
                    ),
                }
            )

            multi = multistart_backward_deletion(
                matrix,
                _gl_loss,
                target_dimension=d,
                num_starts=num_starts,
                relative_tolerance=relative_tolerance,
                random_state=run_seed,
            )
            records.append(
                {
                    **common,
                    **_record(
                        "multistart_backward",
                        multi["indices"],
                        matrix,
                        truth,
                        multi["runtime_seconds"],
                        multi["candidate_evaluations"],
                        multi["arithmetic_work_proxy"],
                        solver_iterations=np.nan,
                        best_start=multi["best_start"],
                    ),
                }
            )

            lookahead = lookahead_backward_deletion(
                matrix,
                _gl_loss,
                target_dimension=d,
                depth=depth,
                beam_width=beam_width,
            )
            records.append(
                {
                    **common,
                    **_record(
                        "lookahead_backward",
                        lookahead["indices"],
                        matrix,
                        truth,
                        lookahead["runtime_seconds"],
                        lookahead["candidate_evaluations"],
                        lookahead["arithmetic_work_proxy"],
                        solver_iterations=np.nan,
                        best_start=np.nan,
                    ),
                }
            )

            started = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                ip = isometry_pursuit(matrix)
            elapsed = time.perf_counter() - started
            ip_indices = [int(index) for index in ip["basis_pursuit_indices"]]
            ip_iterations = int(ip["basis_pursuit_iterations"])
            records.append(
                {
                    **common,
                    **_record(
                        "isometry_pursuit",
                        ip_indices,
                        matrix,
                        truth,
                        elapsed,
                        np.nan,
                        ip_iterations * d * d * p,
                        solver_iterations=ip_iterations,
                        best_start=np.nan,
                    ),
                }
            )
            run_id += 1
    return pd.DataFrame(records)


def summarize_lookahead_backward(records, group_conditions=True):
    keys = ["method"]
    if group_conditions:
        keys += ["D", "P", "corrupting_D", "num_corrupting_matrices"]
    return (
        records.groupby(keys, as_index=False)
        .agg(
            runs=("run_id", "size"),
            success_rate=("strict_success", "mean"),
            containment_rate=("contains_true_block", "mean"),
            mean_loss_gap_from_D=("loss_gap_from_D", "mean"),
            median_runtime_seconds=("runtime_seconds", "median"),
            mean_candidate_evaluations=("candidate_evaluations", "mean"),
            mean_arithmetic_work_proxy=("arithmetic_work_proxy", "mean"),
        )
    )


def plot_lookahead_backward_summary(records, output_path):
    overall = summarize_lookahead_backward(records, group_conditions=False).set_index(
        "method"
    )
    methods = list(METHOD_LABELS)
    labels = [METHOD_LABELS[method] for method in methods]
    colors = ["#54a24b", "#e45756", "#4c78a8", "#f58518"]
    locations = np.arange(len(methods))
    fig, axes = plt.subplots(1, 4, figsize=(15.8, 4.0))

    exact = [overall.loc[method, "success_rate"] for method in methods]
    contains = [overall.loc[method, "containment_rate"] for method in methods]
    axes[0].bar(locations - 0.18, exact, width=0.36, color=colors, label="Exact")
    axes[0].bar(
        locations + 0.18,
        contains,
        width=0.36,
        color=colors,
        alpha=0.38,
        hatch="//",
        label="Contains block",
    )
    axes[0].set_xticks(locations, labels)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Planted-support recovery")
    axes[0].legend(frameon=False, fontsize=8)

    metrics = (
        ("mean_loss_gap_from_D", "Mean loss minus D", False),
        ("median_runtime_seconds", "Median runtime (seconds)", True),
        ("mean_arithmetic_work_proxy", "Mean arithmetic-work proxy", True),
    )
    for axis, (metric, label, log_scale) in zip(axes[1:], metrics):
        axis.bar(labels, [overall.loc[method, metric] for method in methods], color=colors)
        axis.set_ylabel(label)
        if log_scale:
            axis.set_yscale("log")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Pilot: beam look-ahead backward deletion")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_lookahead_backward_experiment(records, output_dir, figure_dir, metadata):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)
    paths = {
        "runs": os.path.join(output_dir, "runs.csv"),
        "summary": os.path.join(output_dir, "summary.csv"),
        "overall": os.path.join(output_dir, "overall.csv"),
        "metadata": os.path.join(output_dir, "metadata.json"),
        "figure": os.path.join(figure_dir, "lookahead_backward_pilot.png"),
    }
    records.to_csv(paths["runs"], index=False)
    summarize_lookahead_backward(records).to_csv(paths["summary"], index=False)
    summarize_lookahead_backward(records, group_conditions=False).to_csv(
        paths["overall"], index=False
    )
    with open(paths["metadata"], "w") as handle:
        json.dump(metadata, handle, indent=2)
    plot_lookahead_backward_summary(records, paths["figure"])
    return paths
