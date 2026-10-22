#!/usr/bin/env python3
"""Study restarted SCS iteration caps for Isometry Pursuit support screening.

Every point is an independent cold solve.  The curves are therefore cap sweeps,
not samples from one continuous SCS trajectory.
"""

import argparse
import ast
import json
import time
import warnings
from pathlib import Path

import cvxpy as cp
from cvxpy.reductions.solvers.conic_solvers.scs_conif import dims_to_solver_dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scs

from isometry_pursuit.algorithm import forward_backward, greedy
from isometry_pursuit.exec.run_identity_experiment import generate_identity_instance
from isometry_pursuit.experiment import _gl_loss
from isometry_pursuit.transformation import exponential_transformation


REAL_DATA = {
    "wine": Path("data/wine_standardized_resampled_tensor.npy"),
    "iris": Path("data/iris_standardized_resampled_tensor.npy"),
    "ethanol": Path("data/ethanol_tensor.npy"),
}
SUPPORT_ENTRY_THRESHOLD = 1e-6
SUPPORT_ROW_THRESHOLD = 1e-4
STOPPING_WINDOW = 3
STOPPING_GAP_CV = 1e-2
STOPPING_NORMALIZED_GAP = 1e-6
STOPPING_NORMALIZED_FEASIBILITY = 1e-6


def _as_list(value):
    if isinstance(value, str):
        return list(ast.literal_eval(value))
    return list(value)


def _problem(matrix):
    d, p = matrix.shape
    beta = cp.Variable((p, d))
    problem = cp.Problem(
        cp.Minimize(cp.sum(cp.norm(beta, axis=1))),
        [matrix @ beta == np.eye(d)],
    )
    data, chain, inverse = problem.get_problem_data(cp.SCS)
    cones = dims_to_solver_dict(data["dims"])
    return beta, problem, data, chain, inverse, cones


def _solve_restart(problem_parts, matrix, cap, eps):
    beta, problem, data, chain, inverse, cones = problem_parts
    started = time.perf_counter()
    solver = scs.SCS(
        {key: data[key] for key in ("A", "b", "c")},
        cones,
        max_iters=int(cap),
        eps_abs=float(eps),
        eps_rel=float(eps),
        acceleration_lookback=0,
        adaptive_scale=False,
        verbose=False,
    )
    raw = solver.solve(warm_start=False)
    wall_seconds = time.perf_counter() - started
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem.unpack_results(raw, chain, inverse)
    coefficients = np.asarray(beta.value).copy()
    raw_residual = float(np.linalg.norm(matrix @ coefficients - np.eye(matrix.shape[0])))
    coefficients[np.abs(coefficients) < SUPPORT_ENTRY_THRESHOLD] = 0.0
    support = np.flatnonzero(
        np.linalg.norm(coefficients, axis=1) > SUPPORT_ROW_THRESHOLD
    ).astype(int).tolist()
    info = raw["info"]
    return {
        "requested_cap": int(cap),
        "reported_iterations": int(info["iter"]),
        "status": str(info["status"]),
        "wall_seconds": wall_seconds,
        "scs_solve_ms": float(info["solve_time"]),
        "objective": float(np.sum(np.linalg.norm(coefficients, axis=1))),
        "scs_primal_objective": float(info["pobj"]),
        "scs_dual_objective": float(info["dobj"]),
        "primal_residual": float(info["res_pri"]),
        "dual_residual": float(info["res_dual"]),
        "duality_gap": float(info["gap"]),
        "reconstruction_fro_raw": raw_residual,
        "reconstruction_fro_thresholded": float(
            np.linalg.norm(matrix @ coefficients - np.eye(matrix.shape[0]))
        ),
        "support": support,
    }


def _downstream(search_matrix, support, dimension, seed):
    if len(support) < dimension:
        return [], np.nan, False
    restricted = search_matrix[:, support]
    local = greedy(
        restricted, _gl_loss, target_dimension=dimension, random_state=seed
    )
    local = forward_backward(
        restricted,
        _gl_loss,
        target_dimension=dimension,
        selected_indices=local,
        random_state=seed,
    )
    selected = [int(support[index]) for index in local]
    return selected, float(_gl_loss(search_matrix[:, selected])), True


def _record(dataset, family, replicate, seed, raw_matrix, search_matrix,
            reference_support, reference_downstream, caps, true_support=None,
            condition=None, eps=1e-12):
    transformed = exponential_transformation(raw_matrix)
    parts = _problem(transformed)
    reference_set = set(reference_support)
    reference_downstream_set = set(reference_downstream)
    true_set = set(true_support or [])
    rows = []
    for cap in caps:
        solved = _solve_restart(parts, transformed, cap, eps)
        support = solved.pop("support")
        support_set = set(support)
        downstream, downstream_loss, downstream_valid = _downstream(
            search_matrix, support, raw_matrix.shape[0], seed
        )
        union = support_set | reference_set
        row = {
            "dataset": dataset,
            "family": family,
            "replicate": int(replicate),
            "seed": int(seed),
            "D": int(raw_matrix.shape[0]),
            "P": int(raw_matrix.shape[1]),
            **(condition or {}),
            **solved,
            "support": json.dumps(support),
            "support_cardinality": len(support),
            "reference_support": json.dumps(reference_support),
            "reference_support_cardinality": len(reference_support),
            "support_exact_match": support_set == reference_set,
            "support_jaccard": len(support_set & reference_set) / len(union) if union else 1.0,
            "downstream_valid": downstream_valid,
            "downstream_indices": json.dumps(downstream),
            "downstream_loss": downstream_loss,
            "reference_downstream_indices": json.dumps(reference_downstream),
            "downstream_exact_match": (
                downstream_valid and set(downstream) == reference_downstream_set
            ),
        }
        if true_support is not None:
            row.update({
                "support_contains_true": true_set.issubset(support_set),
                "downstream_exact_true": downstream_valid and set(downstream) == true_set,
            })
        rows.append(row)
    return rows


def run_real(caps, max_replicates, eps):
    rows = []
    for dataset, path in REAL_DATA.items():
        tensor = np.load(path)
        reference = json.loads(Path(f"results/{dataset}_results.json").read_text())
        count = min(tensor.shape[2], max_replicates or tensor.shape[2])
        for replicate in range(count):
            saved = reference["replicates"][replicate]
            raw = tensor[:, :, replicate]
            rows.extend(_record(
                dataset, "real", replicate, replicate, raw,
                exponential_transformation(raw),
                [int(x) for x in saved["basis_pursuit_indices"]],
                [int(x) for x in saved["forward_backward_gl_bp_indices"]],
                caps, eps=eps,
            ))
            print(f"{dataset} {replicate + 1}/{count}", flush=True)
    return rows


def run_synthetic(caps, runs_csv, per_condition, eps):
    frame = pd.read_csv(runs_csv).sort_values(["trial_id", "replicate"])
    frame = frame.groupby("trial_id", sort=True).head(per_condition)
    rows = []
    for number, row in enumerate(frame.itertuples(index=False), start=1):
        rng = np.random.default_rng(int(row.seed))
        matrix, true_indices, _, _ = generate_identity_instance(
            int(row.D), int(row.corrupting_D), int(row.num_corrupting_matrices), rng
        )
        rows.extend(_record(
            "synthetic", "synthetic", int(row.replicate), int(row.seed),
            matrix, matrix, _as_list(row.selected_indices),
            _as_list(row.forward_backward_restricted_indices), caps,
            true_support=true_indices,
            condition={
                "trial_id": int(row.trial_id),
                "corrupting_D": int(row.corrupting_D),
                "num_corrupting_matrices": int(row.num_corrupting_matrices),
            }, eps=eps,
        ))
        print(f"synthetic {number}/{len(frame)}", flush=True)
    return rows


def summarize(frame):
    metrics = [
        "support_exact_match", "support_jaccard", "downstream_exact_match",
        "downstream_valid", "wall_seconds", "reported_iterations",
        "support_cardinality", "reconstruction_fro_raw",
        "reconstruction_fro_thresholded", "objective",
    ]
    for optional in ("support_contains_true", "downstream_exact_true"):
        if optional in frame:
            metrics.append(optional)
    grouped = frame.groupby(["dataset", "requested_cap"], dropna=False)
    output = grouped[metrics].agg(["mean", "median", "min", "max"])
    output.columns = [f"{name}_{stat}" for name, stat in output.columns]
    output = output.reset_index()
    output["runs"] = grouped.size().to_numpy()
    return output


def support_stabilization(frame):
    """First tested cap after which support matches the reference at every later cap."""
    records = []
    real = frame[frame["family"] == "real"]
    for (dataset, replicate), group in real.groupby(["dataset", "replicate"]):
        group = group.sort_values("requested_cap").reset_index(drop=True)
        matches = group["support_exact_match"].astype(bool)
        stable_suffix = matches.iloc[::-1].cummin().iloc[::-1]
        stable_positions = np.flatnonzero(stable_suffix.to_numpy())
        first_position = int(stable_positions[0]) if len(stable_positions) else None
        records.append({
            "dataset": dataset,
            "replicate": int(replicate),
            "last_tested_unstable_cap": (
                int(group.loc[first_position - 1, "requested_cap"])
                if first_position is not None and first_position > 0 else None
            ),
            "first_tested_stable_cap": (
                int(group.loc[first_position, "requested_cap"])
                if first_position is not None else None
            ),
            "tested_through_cap": int(group["requested_cap"].max()),
        })
    return pd.DataFrame(records)


def plot_summary(summary, destination):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    colors = {"wine": "#4c78a8", "iris": "#54a24b", "ethanol": "#e45756",
              "synthetic": "#7a5195"}
    panels = [
        ("support_exact_match_mean", "Exact agreement with 100k support", "Rate"),
        ("downstream_exact_match_mean", "Same greedy→swap result as 100k", "Rate"),
        ("reconstruction_fro_thresholded_median", "Thresholded reconstruction residual", r"$\|A\hat\beta-I\|_F$"),
        ("wall_seconds_median", "Restarted solve wall time", "Seconds"),
    ]
    for axis, (column, title, ylabel) in zip(axes.flat, panels):
        for dataset, group in summary.groupby("dataset"):
            group = group.sort_values("requested_cap")
            axis.plot(group["requested_cap"], group[column], marker="o",
                      label=dataset.title(), color=colors.get(dataset))
        axis.set_xscale("log")
        if "residual" in column or "seconds" in column:
            axis.set_yscale("log")
        else:
            axis.set_ylim(-0.03, 1.03)
        axis.set_title(title)
        axis.set_xlabel("SCS iterations executed (independent cold restart)")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Isometry Pursuit early stopping (independent cold restarts)", fontsize=14)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=220)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)


def plot_real_objective_curves(frame, destination):
    real = frame[frame["family"] == "real"]
    datasets = ["ethanol", "wine", "iris"]
    colors = {"ethanol": "#e45756", "wine": "#4c78a8", "iris": "#54a24b"}
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), constrained_layout=True)
    for axis, dataset in zip(axes, datasets):
        subset = real[real["dataset"] == dataset]
        for _, replicate in subset.groupby("replicate"):
            replicate = replicate.sort_values("requested_cap")
            axis.plot(
                replicate["requested_cap"],
                replicate["objective"],
                color=colors[dataset],
                alpha=0.34,
                linewidth=1.0,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(f"{dataset.title()} ({subset['replicate'].nunique()} replicates)")
        axis.set_xlabel("SCS iterations executed (independent cold restart)")
        axis.grid(True, which="both", alpha=0.2)
    axes[0].set_ylabel(r"IP objective $\sum_p \|\beta_{p\cdot}\|_2$")
    fig.suptitle("Isometry Pursuit objective versus SCS iterations executed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=220)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)


def plot_real_primal_dual_grid(frame, destination):
    """Plot per-replicate primal/dual convergence diagnostics for real data."""
    real = frame[frame["family"] == "real"]
    datasets = ["wine", "iris", "ethanol"]
    colors = {"wine": "#4c78a8", "iris": "#54a24b", "ethanol": "#e45756"}
    metrics = [
        ("scs_primal_objective", "Primal objective", "symlog"),
        ("scs_dual_objective", "Dual objective", "symlog"),
        ("primal_residual", "Primal residual", "log"),
        ("dual_residual", "Dual residual", "log"),
        ("duality_gap", "Duality gap", "log"),
    ]
    fig, axes = plt.subplots(
        len(datasets), len(metrics), figsize=(18, 9.5), constrained_layout=True
    )
    for row, dataset in enumerate(datasets):
        subset = real[real["dataset"] == dataset]
        for column, (metric, title, yscale) in enumerate(metrics):
            axis = axes[row, column]
            for _, replicate in subset.groupby("replicate"):
                replicate = replicate.sort_values("reported_iterations")
                values = replicate[metric].astype(float)
                if yscale == "log":
                    values = values.clip(lower=np.finfo(float).tiny)
                axis.plot(
                    replicate["reported_iterations"],
                    values,
                    color=colors[dataset],
                    alpha=0.3,
                    linewidth=0.9,
                )
            axis.set_xscale("log")
            if yscale == "log":
                axis.set_yscale("log")
            elif yscale == "symlog":
                axis.set_yscale("symlog", linthresh=0.1)
            axis.grid(True, which="both", alpha=0.18)
            if row == 0:
                axis.set_title(title)
            if column == 0:
                axis.set_ylabel(f"{dataset.title()}\nvalue")
            if row == len(datasets) - 1:
                axis.set_xlabel("SCS iterations executed")
    fig.suptitle(
        "SCS primal-dual convergence by dataset (independent cold restarts)",
        fontsize=15,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=220)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)


def stopping_rule_diagnostics(frame, window=STOPPING_WINDOW):
    """Evaluate the candidate task-aware stopping rule at tested checkpoints."""
    records = []
    real = frame[frame["family"] == "real"]
    for (dataset, replicate), group in real.groupby(["dataset", "replicate"]):
        group = group.sort_values("reported_iterations").copy()
        group["normalized_gap"] = group["duality_gap"] / (
            1.0
            + group["scs_primal_objective"].abs()
            + group["scs_dual_objective"].abs()
        )
        group["normalized_feasibility"] = group["reconstruction_fro_raw"] / (
            1.0 + np.sqrt(group["D"].astype(float))
        )
        rolling = group["duality_gap"].rolling(window)
        rolling_mean = rolling.mean().abs()
        # pandas' moment-based rolling std can suffer catastrophic cancellation
        # once the gaps are nearly constant around 1e-10.  Center each short
        # window explicitly so the plotted stagnation statistic remains stable.
        rolling_std = rolling.apply(
            lambda values: np.std(values, ddof=1), raw=True
        )
        group["gap_cv"] = (
            rolling_std
            / rolling_mean.clip(lower=np.finfo(float).tiny)
        )
        supports = group["support"].map(lambda value: frozenset(_as_list(value)))
        downstream = group["downstream_indices"].map(
            lambda value: frozenset(_as_list(value))
        )
        group["support_stable"] = [
            i >= window - 1 and len(set(supports.iloc[i - window + 1:i + 1])) == 1
            for i in range(len(group))
        ]
        group["downstream_stable"] = [
            i >= window - 1 and len(set(downstream.iloc[i - window + 1:i + 1])) == 1
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
        stop_positions = np.flatnonzero(group["candidate_stop"].to_numpy())
        first_stop = int(stop_positions[0]) if len(stop_positions) else None
        group["first_candidate_stop"] = False
        if first_stop is not None:
            group.iloc[
                first_stop, group.columns.get_loc("first_candidate_stop")
            ] = True
        records.append(group)
    return pd.concat(records, ignore_index=True)


def plot_stopping_rule_grid(frame, destination):
    """Plot diagnostics with vertical bars at the inferred stopping iterations."""
    diagnostics = stopping_rule_diagnostics(frame)
    datasets = ["wine", "iris", "ethanol"]
    colors = {"wine": "#4c78a8", "iris": "#54a24b", "ethanol": "#e45756"}
    metrics = [
        ("normalized_gap", "Normalized duality gap"),
        ("gap_cv", f"Gap rolling CV (window={STOPPING_WINDOW})"),
        (
            "normalized_feasibility",
            "Normalized reconstruction error",
        ),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 9.4), constrained_layout=True)
    for row, dataset in enumerate(datasets):
        subset = diagnostics[diagnostics["dataset"] == dataset]
        stops = subset[subset["first_candidate_stop"]]
        stop_counts = stops["reported_iterations"].value_counts().sort_index()
        for column, (metric, title) in enumerate(metrics):
            axis = axes[row, column]
            for _, replicate in subset.groupby("replicate"):
                replicate = replicate.sort_values("reported_iterations")
                values = replicate[metric].astype(float).clip(
                    lower=np.finfo(float).tiny
                )
                axis.plot(
                    replicate["reported_iterations"], values,
                    color=colors[dataset], alpha=0.28, linewidth=0.9,
                )
            for stop_iteration, count in stop_counts.items():
                axis.axvline(
                    stop_iteration,
                    color="black",
                    linestyle="--",
                    linewidth=1.25,
                    alpha=0.8,
                    label=f"stop: {int(stop_iteration):,} ({int(count)}/25)",
                )
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.grid(True, which="both", alpha=0.18)
            if row == 0:
                axis.set_title(title)
            if column == 0:
                axis.set_ylabel(f"{dataset.title()}\nvalue")
            if row == len(datasets) - 1:
                axis.set_xlabel("SCS iterations executed")
            if column == 0 and len(stop_counts):
                axis.legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(
        "Candidate task-aware stopping iterations (vertical bars)",
        fontsize=15,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=220)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)
    return diagnostics


def plot_combined_stopping_grid(frame, destination):
    """Combine every early-stop and primal-dual panel with stopping bars."""
    diagnostics = stopping_rule_diagnostics(frame)
    datasets = ["wine", "iris", "ethanol"]
    colors = {"wine": "#4c78a8", "iris": "#54a24b", "ethanol": "#e45756"}
    metrics = [
        ("support_exact_match", "Exact 100k support", "mean", "linear"),
        (
            "downstream_exact_match",
            "Same greedy→swap result",
            "mean",
            "linear",
        ),
        (
            "reconstruction_fro_thresholded",
            "Thresholded reconstruction residual",
            "median",
            "log",
        ),
        ("wall_seconds", "Wall time", "median", "log"),
        ("scs_primal_objective", "Primal objective", "replicates", "symlog"),
        ("scs_dual_objective", "Dual objective", "replicates", "symlog"),
        ("primal_residual", "Primal residual", "replicates", "log"),
        ("dual_residual", "Dual residual", "replicates", "log"),
        ("duality_gap", "Duality gap", "replicates", "log"),
    ]
    fig, axes = plt.subplots(
        len(datasets), len(metrics), figsize=(31, 9.5), constrained_layout=True
    )
    for row, dataset in enumerate(datasets):
        subset = diagnostics[diagnostics["dataset"] == dataset]
        stops = subset[subset["first_candidate_stop"]]
        stop_counts = stops["reported_iterations"].value_counts().sort_index()
        for column, (metric, title, aggregation, yscale) in enumerate(metrics):
            axis = axes[row, column]
            if aggregation == "replicates":
                for _, replicate in subset.groupby("replicate"):
                    replicate = replicate.sort_values("reported_iterations")
                    values = replicate[metric].astype(float)
                    if yscale == "log":
                        values = values.clip(lower=np.finfo(float).tiny)
                    axis.plot(
                        replicate["reported_iterations"], values,
                        color=colors[dataset], alpha=0.28, linewidth=0.85,
                    )
            else:
                grouped = subset.groupby("reported_iterations")[metric]
                values = getattr(grouped, aggregation)().astype(float)
                if yscale == "log":
                    values = values.clip(lower=np.finfo(float).tiny)
                axis.plot(
                    values.index, values, color=colors[dataset],
                    marker="o", markersize=2.8, linewidth=1.35,
                )
            for stop_iteration, count in stop_counts.items():
                axis.axvline(
                    stop_iteration, color="black", linestyle="--",
                    linewidth=1.05, alpha=0.78,
                    label=f"stop: {int(stop_iteration):,} ({int(count)}/25)",
                )
            axis.set_xscale("log")
            if yscale == "log":
                axis.set_yscale("log")
            elif yscale == "symlog":
                axis.set_yscale("symlog", linthresh=0.1)
            else:
                axis.set_ylim(-0.03, 1.03)
            axis.grid(True, which="both", alpha=0.17)
            if row == 0:
                axis.set_title(title, fontsize=10)
            if column == 0:
                axis.set_ylabel(dataset.title(), fontsize=11)
                if len(stop_counts):
                    axis.legend(frameon=False, fontsize=7, loc="best")
            if row == len(datasets) - 1:
                axis.set_xlabel("SCS iterations executed", fontsize=9)
            axis.tick_params(labelsize=8)
    fig.suptitle(
        "SCS convergence and task recovery with inferred stopping iterations",
        fontsize=16,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=220)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)
    return diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caps", type=int, nargs="+",
                        default=[1, 2, 5, 10, 25, 50, 100, 250, 500, 1000,
                                 2500, 5000, 10000, 25000])
    parser.add_argument("--eps", type=float, default=1e-12,
                        help="Fixed SCS absolute and relative tolerance used by the 100k reference")
    parser.add_argument("--real-max-replicates", type=int, default=None)
    parser.add_argument("--synthetic-runs", type=Path,
                        default=Path("results/identity_swap_comparison_full/identity_experiment_runs.csv"))
    parser.add_argument("--synthetic-per-condition", type=int, default=1)
    parser.add_argument("--skip-real", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/scs_early_stop"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures/scs_early_stop"))
    args = parser.parse_args()
    if any(cap <= 0 for cap in args.caps):
        raise ValueError("caps must be positive")
    rows = []
    if not args.skip_real:
        rows.extend(run_real(args.caps, args.real_max_replicates, args.eps))
    if not args.skip_synthetic:
        rows.extend(run_synthetic(
            args.caps, args.synthetic_runs, args.synthetic_per_condition, args.eps
        ))
    frame = pd.DataFrame(rows)
    summary = summarize(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "runs.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    stabilization = support_stabilization(frame)
    stabilization.to_csv(args.output_dir / "support_stabilization.csv", index=False)
    stabilization_summary = (
        stabilization.groupby("dataset", as_index=False)
        .agg(
            replicates=("replicate", "size"),
            minimum_first_stable=("first_tested_stable_cap", "min"),
            median_first_stable=("first_tested_stable_cap", "median"),
            mean_first_stable=("first_tested_stable_cap", "mean"),
            maximum_first_stable=("first_tested_stable_cap", "max"),
        )
    )
    stabilization_summary.to_csv(
        args.output_dir / "support_stabilization_summary.csv", index=False
    )
    (args.output_dir / "runs.json").write_text(
        json.dumps({"runs": json.loads(frame.to_json(orient="records"))}, indent=2)
        + "\n"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {"summary": json.loads(summary.to_json(orient="records"))},
            indent=2,
        )
        + "\n"
    )
    metadata = {
        "study_design": "independent cold SCS restart at every requested cap",
        "not_a_continuous_trace": True,
        "production_defaults_changed": False,
        "scs_version": scs.__version__,
        "cvxpy_version": cp.__version__,
        "caps": args.caps,
        "eps_abs_and_rel": args.eps,
        "support_entry_threshold": SUPPORT_ENTRY_THRESHOLD,
        "support_row_threshold": SUPPORT_ROW_THRESHOLD,
        "real_max_replicates": args.real_max_replicates,
        "synthetic_per_condition": args.synthetic_per_condition,
        "rows": len(frame),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot_summary(summary, args.figure_dir / "scs_early_stop.png")
    plot_real_objective_curves(
        frame, args.figure_dir / "scs_objective_by_replicate.png"
    )
    if {"scs_primal_objective", "scs_dual_objective"}.issubset(frame.columns):
        plot_real_primal_dual_grid(
            frame, args.figure_dir / "scs_primal_dual_grid.png"
        )
        stopping_diagnostics = plot_stopping_rule_grid(
            frame, args.figure_dir / "scs_stopping_rule_grid.png"
        )
        plot_combined_stopping_grid(
            frame, args.figure_dir / "scs_combined_stopping_grid.png"
        )
        stopping_diagnostics.to_csv(
            args.output_dir / "stopping_rule_diagnostics.csv", index=False
        )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
