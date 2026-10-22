"""Experiment helpers for rank-safe batched backward elimination."""

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .algorithm import backward_elimination, isometry_pursuit
from .batched_backward import batched_backward_elimination
from .exec.run_identity_experiment import generate_identity_instance
from .experiment import _gl_loss


def _plain_backward_counts(dimension, columns):
    evaluations = sum(range(dimension + 1, columns + 1))
    proxy = dimension * dimension * sum(
        k * (k - 1) for k in range(dimension + 1, columns + 1)
    )
    return evaluations, proxy


def run_batched_backward_experiment(
    conditions,
    repeats=3,
    seed=95173,
    batch_sizes=(2, 4, 8, "halving"),
    ip_iterations=100000,
):
    records = []
    for condition_index, condition in enumerate(conditions):
        dimension, corrupting_dimension, num_corrupting = condition
        for replicate in range(repeats):
            run_seed = seed + 1000 * condition_index + replicate
            matrix, true_indices, _, _ = generate_identity_instance(
                dimension=dimension,
                corrupting_dimension=corrupting_dimension,
                num_corrupting=num_corrupting,
                rng=np.random.default_rng(run_seed),
            )
            columns = matrix.shape[1]
            true_set = set(true_indices)

            start = time.perf_counter()
            indices = backward_elimination(
                matrix,
                _gl_loss,
                target_dimension=dimension,
                random_state=run_seed,
            )
            runtime = time.perf_counter() - start
            evaluations, proxy = _plain_backward_counts(dimension, columns)
            records.append(
                _record(
                    "backward_greedy", dimension, columns, corrupting_dimension,
                    num_corrupting, replicate, run_seed, indices, true_set,
                    runtime, evaluations, 0, columns - dimension, proxy,
                    _gl_loss(matrix[:, indices]), None,
                )
            )

            for batch_size in batch_sizes:
                start = time.perf_counter()
                result = batched_backward_elimination(
                    matrix,
                    _gl_loss,
                    target_dimension=dimension,
                    batch_size=batch_size,
                    random_state=run_seed,
                )
                runtime = time.perf_counter() - start
                indices = result["selected_indices"]
                name = f"batch_{batch_size}"
                records.append(
                    _record(
                        name, dimension, columns, corrupting_dimension,
                        num_corrupting, replicate, run_seed, indices, true_set,
                        runtime, result["loss_evaluations"], result["rank_checks"],
                        result["num_stages"], result["arithmetic_work_proxy"],
                        _gl_loss(matrix[:, indices]), None,
                    )
                )

            start = time.perf_counter()
            ip_result = isometry_pursuit(matrix, iterations=ip_iterations)
            runtime = time.perf_counter() - start
            indices = [int(index) for index in ip_result["basis_pursuit_indices"]]
            solver_iterations = int(ip_result["basis_pursuit_iterations"])
            records.append(
                _record(
                    "isometry_pursuit", dimension, columns, corrupting_dimension,
                    num_corrupting, replicate, run_seed, indices, true_set,
                    runtime, None, None, solver_iterations,
                    solver_iterations * dimension * dimension * columns,
                    _gl_loss(matrix[:, indices]), solver_iterations,
                )
            )
    return pd.DataFrame(records)


def _record(
    method, dimension, columns, corrupting_dimension, num_corrupting,
    replicate, seed, indices, true_set, runtime, loss_evaluations,
    rank_checks, stages, proxy, final_loss, solver_iterations,
):
    selected_set = set(indices)
    return {
        "method": method,
        "D": dimension,
        "P": columns,
        "corrupting_D": corrupting_dimension,
        "num_corrupting_matrices": num_corrupting,
        "replicate": replicate,
        "seed": seed,
        "selected_indices": json.dumps([int(index) for index in indices]),
        "support_cardinality": len(indices),
        "exact_recovery": selected_set == true_set,
        "final_loss": float(final_loss),
        "runtime_seconds": runtime,
        "loss_evaluations": loss_evaluations,
        "rank_checks": rank_checks,
        "stages_or_iterations": stages,
        "solver_iterations": solver_iterations,
        "arithmetic_work_proxy": int(proxy),
    }


def summarize_batched_backward(records):
    keys = ["method", "D", "P", "corrupting_D", "num_corrupting_matrices"]
    return (
        records.groupby(keys, as_index=False)
        .agg(
            repeats=("replicate", "size"),
            exact_recovery_rate=("exact_recovery", "mean"),
            median_final_loss=("final_loss", "median"),
            median_runtime_seconds=("runtime_seconds", "median"),
            median_loss_evaluations=("loss_evaluations", "median"),
            median_rank_checks=("rank_checks", "median"),
            median_arithmetic_work_proxy=("arithmetic_work_proxy", "median"),
        )
    )


def plot_batched_backward(summary, output_path):
    method_order = [
        "backward_greedy", "batch_2", "batch_4", "batch_8", "batch_halving",
        "isometry_pursuit",
    ]
    labels = {
        "backward_greedy": "Backward greedy",
        "batch_2": "Batch 2",
        "batch_4": "Batch 4",
        "batch_8": "Batch 8",
        "batch_halving": "Halving",
        "isometry_pursuit": "IP (SCS)",
    }
    conditions = summary[["D", "P"]].drop_duplicates().sort_values(["D", "P"])
    condition_labels = [f"D={row.D}, P={row.P}" for row in conditions.itertuples()]
    x = np.arange(len(condition_labels))
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    for method in method_order:
        group = summary[summary["method"] == method].set_index(["D", "P"])
        if group.empty:
            continue
        rows = [group.loc[(row.D, row.P)] for row in conditions.itertuples()]
        axes[0, 0].plot(x, [row.exact_recovery_rate for row in rows], marker="o", label=labels[method])
        axes[0, 1].plot(x, [row.median_final_loss for row in rows], marker="o")
        axes[1, 0].plot(x, [row.median_runtime_seconds for row in rows], marker="o")
        axes[1, 1].plot(x, [row.median_arithmetic_work_proxy for row in rows], marker="o")
    axes[0, 0].set_ylabel("Exact-recovery rate")
    axes[0, 1].set_ylabel("Median final loss")
    axes[1, 0].set_ylabel("Median runtime (s)")
    axes[1, 1].set_ylabel("Median arithmetic-work proxy")
    axes[1, 0].set_yscale("log")
    axes[1, 1].set_yscale("log")
    for axis in axes.flat:
        axis.set_xticks(x, condition_labels, rotation=18, ha="right")
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle("Rank-safe batched backward deletion")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
