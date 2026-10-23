"""Benchmark new synthetic (D, P) sizes and merge them with saved timings."""

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd

from isometry_pursuit.algorithm import backward_elimination, greedy, isometry_pursuit
from isometry_pursuit.exec.plot_complexity_runtime import (
    BACKWARD_WORK_CUTOFF,
    backward_proxy,
    forward_proxy,
)
from isometry_pursuit.exec.run_identity_experiment import generate_identity_instance
from isometry_pursuit.experiment import _gl_loss


METHODS = ("forward_greedy", "isometry_pursuit", "backward_greedy")


def _write(records, path):
    frame = pd.DataFrame(records)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditions-csv",
        type=Path,
        default=Path(
            "results/identity_swap_comparison_25x_D64/"
            "identity_experiment_runs.csv"
        ),
    )
    parser.add_argument(
        "--base-records",
        type=Path,
        default=Path("results/complexity_runtime/complexity_runtime_runs.csv"),
    )
    parser.add_argument(
        "--extension-records",
        type=Path,
        default=Path(
            "results/complexity_runtime/complexity_runtime_D32_D64_runs.csv"
        ),
    )
    parser.add_argument(
        "--merged-records",
        type=Path,
        default=Path(
            "results/complexity_runtime/complexity_runtime_runs_expanded.csv"
        ),
    )
    parser.add_argument("--seed", type=int, default=91427)
    parser.add_argument(
        "--max-backward-work", type=float, default=BACKWARD_WORK_CUTOFF
    )
    args = parser.parse_args()

    base = pd.read_csv(args.base_records)
    source = pd.read_csv(args.conditions_csv).rename(columns={"total_columns": "P"})
    conditions = (
        source[["D", "P", "corrupting_D", "num_corrupting_matrices"]]
        .drop_duplicates(["D", "P"])
        .sort_values(["D", "P"])
        .reset_index(drop=True)
    )
    existing_sizes = set(map(tuple, base[["D", "P"]].drop_duplicates().to_numpy()))
    conditions = conditions[
        ~conditions.apply(lambda row: (row["D"], row["P"]) in existing_sizes, axis=1)
    ].reset_index(drop=True)

    if args.extension_records.exists():
        records = pd.read_csv(args.extension_records).to_dict("records")
    else:
        records = []
    completed = {
        (int(record["D"]), int(record["P"]), record["method"])
        for record in records
    }

    for condition_index, row in enumerate(conditions.itertuples(index=False)):
        d, p = int(row.D), int(row.P)
        rng = np.random.default_rng(args.seed + condition_index)
        matrix, _, _, _ = generate_identity_instance(
            dimension=d,
            corrupting_dimension=int(row.corrupting_D),
            num_corrupting=int(row.num_corrupting_matrices),
            rng=rng,
        )
        work = {
            "forward_greedy": forward_proxy(d, p),
            "backward_greedy": backward_proxy(d, p),
        }
        calls = {
            "forward_greedy": lambda: greedy(
                matrix, _gl_loss, target_dimension=d
            ),
            "isometry_pursuit": lambda: isometry_pursuit(matrix),
            "backward_greedy": lambda: backward_elimination(
                matrix, _gl_loss, target_dimension=d
            ),
        }
        for method in METHODS:
            if (d, p, method) in completed:
                continue
            if (
                method == "backward_greedy"
                and work[method] > args.max_backward_work
            ):
                continue
            start = time.perf_counter()
            result = calls[method]()
            runtime = time.perf_counter() - start
            if method == "isometry_pursuit":
                iterations = int(result["basis_pursuit_iterations"])
                arithmetic_work = iterations * d * d * p
            elif method == "forward_greedy":
                iterations = d
                arithmetic_work = work[method]
            else:
                iterations = p - d
                arithmetic_work = work[method]
            records.append(
                {
                    "method": method,
                    "D": d,
                    "P": p,
                    "corrupting_D": int(row.corrupting_D),
                    "num_corrupting_matrices": int(row.num_corrupting_matrices),
                    "benchmark_replicate": 0,
                    "iterations": iterations,
                    "arithmetic_work_proxy": arithmetic_work,
                    "runtime_seconds": runtime,
                }
            )
            completed.add((d, p, method))
            _write(records, args.extension_records)
            print(
                f"D={d}, P={p}, {method}: {runtime:.3f}s "
                f"({len(records)} saved)",
                flush=True,
            )

    extension = pd.DataFrame(records)
    merged = pd.concat([base, extension], ignore_index=True)
    _write(merged.to_dict("records"), args.merged_records)
    print(f"Merged timings: {args.merged_records}", flush=True)


if __name__ == "__main__":
    main()
