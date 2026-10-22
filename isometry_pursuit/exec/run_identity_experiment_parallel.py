#!/usr/bin/env python3
"""Run deterministic identity-experiment condition shards in parallel."""

import argparse
import ast
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Avoid multiplying native BLAS threads inside each process worker.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import pandas as pd

from isometry_pursuit.exec.run_identity_experiment import (
    run_identity_experiment,
    write_simple_yaml,
)


DEFAULT_D_VALUES = [2, 4, 8, 16, 32, 64]
DEFAULT_CORRUPTING_DIMS = {
    2: [1],
    4: [1, 2, 3],
    8: [1, 2, 4, 7],
    16: [1, 4, 8, 15],
    32: [1, 8, 16, 31],
    64: [1, 16, 32, 63],
}
DEFAULT_NUM_CORRUPTING = [1, 2, 4, 8, 16]


def _literal(value, name):
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Could not parse {name}: {value}") from exc


def _corrupting_dims(raw_value, dimensions):
    if raw_value is None:
        return {
            dimension: DEFAULT_CORRUPTING_DIMS.get(
                dimension,
                sorted({1, dimension // 4, dimension // 2, dimension - 1}),
            )
            for dimension in dimensions
        }
    parsed = _literal(raw_value, "--corrupting-d-values")
    if isinstance(parsed, dict):
        mapping = {int(key): [int(value) for value in values]
                   for key, values in parsed.items()}
        missing = sorted(set(dimensions) - set(mapping))
        if missing:
            raise ValueError(f"Missing corrupting dimensions for D={missing}")
        return {dimension: mapping[dimension] for dimension in dimensions}
    shared = [int(value) for value in parsed]
    return {
        dimension: [value for value in shared if 0 < value < dimension]
        for dimension in dimensions
    }


def _build_trials(dimensions, corrupting_dims, num_corrupting_values):
    return [
        {
            "D": dimension,
            "corrupting_D": corrupting_dimension,
            "num_corrupting_matrices": num_corrupting,
        }
        for dimension in dimensions
        for corrupting_dimension in corrupting_dims[dimension]
        for num_corrupting in num_corrupting_values
    ]


def _shard_dir(shards_dir, trial_index, trial):
    return shards_dir / (
        f"trial_{trial_index:03d}_D{trial['D']}_dc{trial['corrupting_D']}_"
        f"b{trial['num_corrupting_matrices']}"
    )


def _same_optional(left, right):
    if left is None or right is None:
        return left is right
    return float(left) == float(right)


def _valid_shard(
    shard_dir, trial_index, trial, replicates, seed, settings
):
    runs_path = shard_dir / "identity_experiment_runs.json"
    csv_path = shard_dir / "identity_experiment_runs.csv"
    summary_path = shard_dir / "identity_experiment_summary.csv"
    if not all(path.exists() for path in (runs_path, csv_path, summary_path)):
        return False
    try:
        records = json.loads(runs_path.read_text())["runs"]
    except (OSError, ValueError, KeyError):
        return False
    return (
        len(records) == replicates
        and sorted(int(record["replicate"]) for record in records)
        == list(range(replicates))
        and all(
            all(int(record[key]) == int(trial[key]) for key in (
                "D", "corrupting_D", "num_corrupting_matrices"
            ))
            and int(record["seed"])
            == seed + trial_index * replicates + int(record["replicate"])
            and _same_optional(
                record.get("constraint_violation"),
                settings["constraint_violation"],
            )
            and _same_optional(record.get("iterations"), settings["iterations"])
            and _same_optional(record.get("eps"), settings["eps"])
            for record in records
        )
    )


def _run_shard(payload):
    trial_index, trial, replicates, shards_dir, seed, settings = payload
    shard_dir = _shard_dir(Path(shards_dir), trial_index, trial)
    if _valid_shard(
        shard_dir, trial_index, trial, replicates, seed, settings
    ):
        return trial_index, str(shard_dir), True

    # The serial runner uses seed + local run_id. Offsetting each condition by
    # its global run range preserves the exact serial-sweep random streams.
    shard_seed = seed + trial_index * replicates
    run_identity_experiment(
        trials=[trial],
        replicates=replicates,
        output_dir=str(shard_dir),
        seed=shard_seed,
        constraint_violation=settings["constraint_violation"],
        iterations=settings["iterations"],
        eps=settings["eps"],
        config_file=None,
        max_backward_columns=settings["max_backward_columns"],
        max_backward_work=settings["max_backward_work"],
    )
    if not _valid_shard(
        shard_dir, trial_index, trial, replicates, seed, settings
    ):
        raise RuntimeError(f"Shard validation failed: {shard_dir}")
    return trial_index, str(shard_dir), False


def _merge_shards(output_dir, shards_dir, trials, replicates, parameters):
    merged_records = []
    merged_frames = []
    summary_frames = []

    for trial_index, trial in enumerate(trials):
        shard_dir = _shard_dir(shards_dir, trial_index, trial)
        if not _valid_shard(
            shard_dir,
            trial_index,
            trial,
            replicates,
            parameters["seed"],
            parameters,
        ):
            raise RuntimeError(f"Cannot merge incomplete shard: {shard_dir}")

        records = json.loads(
            (shard_dir / "identity_experiment_runs.json").read_text()
        )["runs"]
        frame = pd.read_csv(shard_dir / "identity_experiment_runs.csv")
        summary_frames.append(
            pd.read_csv(shard_dir / "identity_experiment_summary.csv")
        )

        for record in records:
            replicate = int(record["replicate"])
            record["trial_id"] = trial_index
            record["run_id"] = trial_index * replicates + replicate
            merged_records.append(record)
        frame["trial_id"] = trial_index
        frame["run_id"] = trial_index * replicates + frame["replicate"].astype(int)
        merged_frames.append(frame)

    merged_records.sort(key=lambda row: int(row["run_id"]))
    merged_frame = pd.concat(merged_frames, ignore_index=True).sort_values("run_id")
    summary_frame = pd.concat(summary_frames, ignore_index=True).sort_values(
        ["D", "corrupting_D", "num_corrupting_matrices"]
    )

    runs_json = output_dir / "identity_experiment_runs.json"
    runs_csv = output_dir / "identity_experiment_runs.csv"
    summary_csv = output_dir / "identity_experiment_summary.csv"
    metadata_yaml = output_dir / "identity_experiment_metadata.yaml"

    temporary_json = runs_json.with_suffix(".json.tmp")
    temporary_csv = runs_csv.with_suffix(".csv.tmp")
    temporary_summary = summary_csv.with_suffix(".csv.tmp")
    temporary_json.write_text(json.dumps({"runs": merged_records}, indent=2))
    merged_frame.to_csv(temporary_csv, index=False)
    summary_frame.to_csv(temporary_summary, index=False)
    temporary_json.replace(runs_json)
    temporary_csv.replace(runs_csv)
    temporary_summary.replace(summary_csv)

    metadata = {
        "experiment": "identity_matrices_parallel",
        "description": (
            "Deterministically merged condition shards for the synthetic "
            "identity-block experiment."
        ),
        "parameters": {**parameters, "trials": trials},
        "outputs": {
            "runs_json": str(runs_json),
            "runs_csv": str(runs_csv),
            "summary_csv": str(summary_csv),
            "shards_dir": str(shards_dir),
        },
    }
    write_simple_yaml(metadata_yaml, metadata)
    return runs_json, runs_csv, summary_csv, metadata_yaml


def main():
    parser = argparse.ArgumentParser(
        description="Run identity-experiment condition shards in parallel"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/identity_swap_comparison_25x_D64"),
    )
    parser.add_argument("--D-values", default=str(DEFAULT_D_VALUES))
    parser.add_argument(
        "--corrupting-d-values",
        help="Dict keyed by D or a shared list; defaults scale with each D.",
    )
    parser.add_argument(
        "--num-corrupting-values", default=str(DEFAULT_NUM_CORRUPTING)
    )
    parser.add_argument("--replicates", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--constraint-violation", type=float, default=0.0)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--eps", type=float)
    parser.add_argument(
        "--max-backward-columns", type=int,
        help="Skip backward-initialized methods above this dictionary size",
    )
    parser.add_argument(
        "--max-backward-work", type=float, default=1e9,
        help="Skip backward methods above this arithmetic-work proxy",
    )
    args = parser.parse_args()

    dimensions = [int(value) for value in _literal(args.D_values, "--D-values")]
    num_corrupting = [
        int(value)
        for value in _literal(args.num_corrupting_values,
                              "--num-corrupting-values")
    ]
    corrupting_dims = _corrupting_dims(args.corrupting_d_values, dimensions)
    trials = _build_trials(dimensions, corrupting_dims, num_corrupting)
    if args.replicates < 1 or args.workers < 1:
        parser.error("--replicates and --workers must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = args.output_dir / "shards"
    shards_dir.mkdir(exist_ok=True)
    settings = {
        "constraint_violation": args.constraint_violation,
        "iterations": args.iterations,
        "eps": args.eps,
        "max_backward_columns": args.max_backward_columns,
        "max_backward_work": args.max_backward_work,
    }
    payloads = [
        (index, trial, args.replicates, str(shards_dir), args.seed, settings)
        for index, trial in enumerate(trials)
    ]

    print(
        f"Running {len(trials)} conditions x {args.replicates} replicates "
        f"= {len(trials) * args.replicates} runs with {args.workers} workers",
        flush=True,
    )
    completed = 0
    resumed = 0
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=context
    ) as executor:
        futures = [executor.submit(_run_shard, payload) for payload in payloads]
        for future in as_completed(futures):
            trial_index, shard_dir, was_resumed = future.result()
            completed += 1
            resumed += int(was_resumed)
            print(
                f"[{completed}/{len(trials)}] trial {trial_index}: {shard_dir}"
                f"{' (resumed)' if was_resumed else ''}",
                flush=True,
            )

    outputs = _merge_shards(
        args.output_dir,
        shards_dir,
        trials,
        args.replicates,
        {
            "dimensions": dimensions,
            "corrupting_dimensions": corrupting_dims,
            "num_corrupting_matrices": num_corrupting,
            "replicates": args.replicates,
            "seed": args.seed,
            "workers": args.workers,
            **settings,
        },
    )
    print(f"Merged {len(trials)} shards ({resumed} resumed)", flush=True)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
