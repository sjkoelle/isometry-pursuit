"""Pilot experiment for randomized multi-start backward deletion."""

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
from .multistart_backward import backward_work_counts, multistart_backward_deletion


METHOD_LABELS = {
    "backward_greedy": "Backward greedy",
    "multistart_backward": "Multi-start backward",
    "isometry_pursuit": "Isometry Pursuit",
}


def _identity_instance(dimension, corrupting_dimension, num_corrupting, rng):
    """Generate the identity-block model without depending on an entry point."""
    from scipy.stats import ortho_group

    blocks = [np.eye(dimension)]
    for _ in range(num_corrupting):
        if corrupting_dimension == 1:
            local = np.zeros((dimension, 1))
            local[0, 0] = rng.choice([-1.0, 1.0])
        else:
            local = np.zeros((dimension, corrupting_dimension))
            local[:corrupting_dimension, :] = ortho_group.rvs(
                corrupting_dimension, random_state=rng
            )
        blocks.append(ortho_group.rvs(dimension, random_state=rng) @ local)
    return np.column_stack(blocks), list(range(dimension))


def _record(method, indices, matrix, truth, runtime, evaluations, proxy, **extra):
    indices = [int(index) for index in indices]
    try:
        objective = float(_gl_loss(matrix[:, indices]))
    except Exception:
        objective = np.inf
    return {
        "method": method,
        "selected_indices": json.dumps(indices),
        "support_cardinality": len(indices),
        "loss": objective,
        "loss_gap_from_D": objective - matrix.shape[0],
        "strict_success": set(indices) == set(truth),
        "contains_true_block": set(truth).issubset(indices),
        "candidate_evaluations": evaluations,
        "arithmetic_work_proxy": proxy,
        "runtime_seconds": runtime,
        **extra,
    }


def run_multistart_backward_experiment(
    conditions,
    replicates=3,
    num_starts=8,
    relative_tolerance=0.01,
    seed=20261019,
):
    """Evaluate backward greedy, multi-start deletion, and IP."""
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
            elapsed = time.perf_counter() - started
            records.append(
                {
                    **common,
                    **_record(
                        "backward_greedy",
                        backward_indices,
                        matrix,
                        truth,
                        elapsed,
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


def summarize_multistart_backward(records):
    keys = ["method", "D", "P", "corrupting_D", "num_corrupting_matrices"]
    return (
        records.groupby(keys, as_index=False)
        .agg(
            replicates=("run_id", "size"),
            success_rate=("strict_success", "mean"),
            containment_rate=("contains_true_block", "mean"),
            mean_loss=("loss", "mean"),
            median_loss=("loss", "median"),
            median_runtime_seconds=("runtime_seconds", "median"),
            mean_candidate_evaluations=("candidate_evaluations", "mean"),
            mean_arithmetic_work_proxy=("arithmetic_work_proxy", "mean"),
        )
    )


def overall_multistart_backward_summary(records):
    return (
        records.groupby("method", as_index=False)
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


def plot_multistart_backward_summary(records, output_path):
    overall = overall_multistart_backward_summary(records).set_index("method")
    methods = list(METHOD_LABELS)
    labels = [METHOD_LABELS[method] for method in methods]
    colors = ["#54a24b", "#e45756", "#f58518"]
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.8))
    locations = np.arange(len(methods))
    exact = [overall.loc[method, "success_rate"] for method in methods]
    containment = [overall.loc[method, "containment_rate"] for method in methods]
    axes[0].bar(locations - 0.18, exact, width=0.36, color=colors, label="Exact")
    axes[0].bar(
        locations + 0.18,
        containment,
        width=0.36,
        color=colors,
        alpha=0.38,
        hatch="//",
        label="Contains block",
    )
    axes[0].set_xticks(locations, labels)
    axes[0].set_ylabel("Planted-support recovery")
    axes[0].set_ylim(0, 1)
    axes[0].legend(frameon=False, fontsize=8)

    values = [overall.loc[method, "mean_loss_gap_from_D"] for method in methods]
    axes[1].bar(labels, values, color=colors)
    axes[1].set_ylabel("Mean loss minus D")

    values = [overall.loc[method, "median_runtime_seconds"] for method in methods]
    axes[2].bar(labels, values, color=colors)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Median runtime (seconds)")

    values = [overall.loc[method, "mean_arithmetic_work_proxy"] for method in methods]
    axes[3].bar(labels, values, color=colors)
    axes[3].set_yscale("log")
    axes[3].set_ylabel("Mean arithmetic-work proxy")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Pilot: randomized multi-start backward deletion")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_multistart_backward_experiment(records, output_dir, figure_dir, metadata):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)
    runs_path = os.path.join(output_dir, "runs.csv")
    summary_path = os.path.join(output_dir, "summary.csv")
    overall_path = os.path.join(output_dir, "overall.csv")
    metadata_path = os.path.join(output_dir, "metadata.json")
    figure_path = os.path.join(figure_dir, "multistart_backward_pilot.png")
    records.to_csv(runs_path, index=False)
    summarize_multistart_backward(records).to_csv(summary_path, index=False)
    overall_multistart_backward_summary(records).to_csv(overall_path, index=False)
    with open(metadata_path, "w") as handle:
        json.dump(metadata, handle, indent=2)
    plot_multistart_backward_summary(records, figure_path)
    return {
        "runs": runs_path,
        "summary": summary_path,
        "overall": overall_path,
        "metadata": metadata_path,
        "figure": figure_path,
    }
