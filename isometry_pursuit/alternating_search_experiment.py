"""Paired synthetic evaluation of expansion-contraction local search."""

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .algorithm import backward_elimination, forward_backward, isometry_pursuit
from .alternating_search import alternating_add_delete
from .exec.run_identity_experiment import generate_identity_instance
from .experiment import _gl_loss


class CountingLoss:
    """Count subset-loss calls and their leading-order SVD work proxy."""

    def __init__(self, loss, ambient_dimension):
        self.loss = loss
        self.ambient_dimension = ambient_dimension
        self.evaluations = 0
        self.arithmetic_work_proxy = 0

    def __call__(self, sub_matrix, indices=None):
        columns = sub_matrix.shape[1]
        self.evaluations += 1
        self.arithmetic_work_proxy += (
            self.ambient_dimension
            * columns
            * min(self.ambient_dimension, columns)
        )
        return self.loss(sub_matrix, indices=indices)


def run_alternating_search_experiment(
    conditions,
    repeats=3,
    seed=71513,
    expansion_widths=(2, 4),
    max_cycles=4,
    ip_iterations=100000,
):
    records = []
    for condition_index, condition in enumerate(conditions):
        dimension, corrupting_dimension, num_corrupting = condition
        for replicate in range(repeats):
            run_seed = seed + 1000 * condition_index + replicate
            matrix, true_indices, _, _ = generate_identity_instance(
                dimension,
                corrupting_dimension,
                num_corrupting,
                np.random.default_rng(run_seed),
            )
            true_set = set(true_indices)
            columns = matrix.shape[1]

            bg_counter = CountingLoss(_gl_loss, dimension)
            start = time.perf_counter()
            bg_indices = backward_elimination(
                matrix, bg_counter, target_dimension=dimension, random_state=run_seed
            )
            bg_runtime = time.perf_counter() - start
            records.append(
                _record(
                    "backward_greedy", matrix, bg_indices, true_set, condition,
                    replicate, run_seed, bg_runtime, bg_counter.evaluations, 0,
                    bg_counter.arithmetic_work_proxy, None, None,
                )
            )

            swap_counter = CountingLoss(_gl_loss, dimension)
            start = time.perf_counter()
            swap_indices = forward_backward(
                matrix, swap_counter, target_dimension=dimension,
                selected_indices=bg_indices, random_state=run_seed,
            )
            swap_runtime = time.perf_counter() - start
            records.append(
                _record(
                    "bg_swap", matrix, swap_indices, true_set, condition,
                    replicate, run_seed, bg_runtime + swap_runtime,
                    bg_counter.evaluations + swap_counter.evaluations, 0,
                    bg_counter.arithmetic_work_proxy + swap_counter.arithmetic_work_proxy,
                    None, None,
                )
            )

            for width in expansion_widths:
                start = time.perf_counter()
                result = alternating_add_delete(
                    matrix,
                    _gl_loss,
                    bg_indices,
                    target_dimension=dimension,
                    expansion_width=width,
                    max_cycles=max_cycles,
                    random_state=run_seed,
                )
                search_runtime = time.perf_counter() - start
                records.append(
                    _record(
                        f"bg_alternating_{width}", matrix,
                        result["selected_indices"], true_set, condition, replicate,
                        run_seed, bg_runtime + search_runtime,
                        bg_counter.evaluations + result["loss_evaluations"],
                        result["rank_checks"],
                        bg_counter.arithmetic_work_proxy
                        + result["arithmetic_work_proxy"],
                        result["cycles"], result["accepted_cycles"],
                    )
                )

            start = time.perf_counter()
            ip_result = isometry_pursuit(matrix, iterations=ip_iterations)
            ip_runtime = time.perf_counter() - start
            ip_indices = [int(index) for index in ip_result["basis_pursuit_indices"]]
            ip_iterations_used = int(ip_result["basis_pursuit_iterations"])
            ip_proxy = ip_iterations_used * dimension * dimension * columns
            records.append(
                _record(
                    "isometry_pursuit", matrix, ip_indices, true_set, condition,
                    replicate, run_seed, ip_runtime, 0, 0, ip_proxy,
                    None, None, solver_iterations=ip_iterations_used,
                )
            )

            if len(ip_indices) >= dimension:
                restricted = matrix[:, ip_indices]
                restricted_counter = CountingLoss(_gl_loss, dimension)
                start = time.perf_counter()
                local_bg = backward_elimination(
                    restricted,
                    restricted_counter,
                    target_dimension=dimension,
                    random_state=run_seed,
                )
                restricted_bg_runtime = time.perf_counter() - start

                restricted_swap_counter = CountingLoss(_gl_loss, dimension)
                start = time.perf_counter()
                local_swap = forward_backward(
                    restricted,
                    restricted_swap_counter,
                    target_dimension=dimension,
                    selected_indices=local_bg,
                    random_state=run_seed,
                )
                restricted_swap_runtime = time.perf_counter() - start
                mapped_swap = [ip_indices[index] for index in local_swap]
                records.append(
                    _record(
                        "ip_bg_swap", matrix, mapped_swap, true_set, condition,
                        replicate, run_seed,
                        ip_runtime + restricted_bg_runtime + restricted_swap_runtime,
                        restricted_counter.evaluations
                        + restricted_swap_counter.evaluations,
                        0,
                        ip_proxy + restricted_counter.arithmetic_work_proxy
                        + restricted_swap_counter.arithmetic_work_proxy,
                        None, None, solver_iterations=ip_iterations_used,
                    )
                )

                start = time.perf_counter()
                restricted_alt = alternating_add_delete(
                    restricted,
                    _gl_loss,
                    local_bg,
                    target_dimension=dimension,
                    expansion_width=2,
                    max_cycles=max_cycles,
                    random_state=run_seed,
                )
                restricted_alt_runtime = time.perf_counter() - start
                mapped_alt = [
                    ip_indices[index] for index in restricted_alt["selected_indices"]
                ]
                records.append(
                    _record(
                        "ip_bg_alternating_2", matrix, mapped_alt, true_set,
                        condition, replicate, run_seed,
                        ip_runtime + restricted_bg_runtime + restricted_alt_runtime,
                        restricted_counter.evaluations
                        + restricted_alt["loss_evaluations"],
                        restricted_alt["rank_checks"],
                        ip_proxy + restricted_counter.arithmetic_work_proxy
                        + restricted_alt["arithmetic_work_proxy"],
                        restricted_alt["cycles"],
                        restricted_alt["accepted_cycles"],
                        solver_iterations=ip_iterations_used,
                    )
                )
    return pd.DataFrame(records)


def _record(
    method, matrix, indices, true_set, condition, replicate, seed, runtime,
    loss_evaluations, rank_checks, proxy, cycles, accepted_cycles,
    solver_iterations=None,
):
    dimension, corrupting_dimension, num_corrupting = condition
    selected = [int(index) for index in indices]
    final_loss = float(_gl_loss(matrix[:, selected]))
    return {
        "method": method,
        "D": dimension,
        "P": matrix.shape[1],
        "corrupting_D": corrupting_dimension,
        "num_corrupting_matrices": num_corrupting,
        "replicate": replicate,
        "seed": seed,
        "selected_indices": json.dumps(selected),
        "support_cardinality": len(selected),
        "exact_recovery": set(selected) == true_set,
        "final_loss": final_loss,
        "runtime_seconds": runtime,
        "loss_evaluations": loss_evaluations,
        "rank_checks": rank_checks,
        "arithmetic_work_proxy": int(proxy),
        "cycles": cycles,
        "accepted_cycles": accepted_cycles,
        "solver_iterations": solver_iterations,
    }


def summarize_alternating_search(records):
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
            median_accepted_cycles=("accepted_cycles", "median"),
        )
    )


def plot_alternating_search(summary, output_path):
    methods = [
        "backward_greedy", "bg_swap", "bg_alternating_2",
        "bg_alternating_4", "isometry_pursuit", "ip_bg_swap",
        "ip_bg_alternating_2",
    ]
    labels = {
        "backward_greedy": "BG",
        "bg_swap": "BG → swap",
        "bg_alternating_2": "BG → alt-2",
        "bg_alternating_4": "BG → alt-4",
        "isometry_pursuit": "IP",
        "ip_bg_swap": "IP → BG → swap",
        "ip_bg_alternating_2": "IP → BG → alt-2",
    }
    conditions = summary[["D", "P"]].drop_duplicates().sort_values(["D", "P"])
    condition_labels = [f"D={row.D}, P={row.P}" for row in conditions.itertuples()]
    x = np.arange(len(conditions))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.8))
    for method in methods:
        group = summary[summary.method == method].set_index(["D", "P"])
        if group.empty or any((row.D, row.P) not in group.index for row in conditions.itertuples()):
            continue
        rows = [group.loc[(row.D, row.P)] for row in conditions.itertuples()]
        axes[0, 0].plot(x, [row.exact_recovery_rate for row in rows], marker="o", label=labels[method])
        axes[0, 1].plot(x, [row.median_final_loss for row in rows], marker="o")
        axes[1, 0].plot(x, [row.median_runtime_seconds for row in rows], marker="o")
        axes[1, 1].plot(x, [row.median_arithmetic_work_proxy for row in rows], marker="o")
    axes[0, 0].set_ylabel("Exact-recovery rate")
    axes[0, 1].set_ylabel("Median final loss")
    axes[1, 0].set_ylabel("Median total runtime (s)")
    axes[1, 1].set_ylabel("Median arithmetic-work proxy")
    axes[1, 0].set_yscale("log")
    axes[1, 1].set_yscale("log")
    for axis in axes.flat:
        axis.set_xticks(x, condition_labels, rotation=18, ha="right")
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=7.5, ncol=2)
    fig.suptitle("Expansion-contraction search versus one-swap refinement")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
