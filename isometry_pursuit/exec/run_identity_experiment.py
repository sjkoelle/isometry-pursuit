#!/usr/bin/env python3
"""Run synthetic identity-block Isometry Pursuit experiments."""

import argparse
import ast
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import ortho_group
from tqdm import tqdm

from isometry_pursuit.algorithm import forward_backward, greedy, greedy_swap, isometry_pursuit
from isometry_pursuit.experiment import _gl_loss


DEFAULT_CORRUPTING_DIMS = {
    2: [1],
    4: [1, 2, 3],
    8: [1, 2, 4, 7],
    16: [1, 4, 8, 15],
}

DEFAULT_CONFIG = {
    "output_dir": "results/identity_experiment",
    "trials": [
        {"D": 2, "corrupting_D": 1, "num_corrupting_matrices": 1},
        {"D": 2, "corrupting_D": 1, "num_corrupting_matrices": 2},
        {"D": 2, "corrupting_D": 1, "num_corrupting_matrices": 4},
        {"D": 4, "corrupting_D": 1, "num_corrupting_matrices": 1},
        {"D": 4, "corrupting_D": 1, "num_corrupting_matrices": 2},
        {"D": 4, "corrupting_D": 1, "num_corrupting_matrices": 4},
        {"D": 4, "corrupting_D": 2, "num_corrupting_matrices": 1},
        {"D": 4, "corrupting_D": 2, "num_corrupting_matrices": 2},
        {"D": 4, "corrupting_D": 2, "num_corrupting_matrices": 4},
        {"D": 4, "corrupting_D": 3, "num_corrupting_matrices": 1},
        {"D": 4, "corrupting_D": 3, "num_corrupting_matrices": 2},
        {"D": 4, "corrupting_D": 3, "num_corrupting_matrices": 4},
    ],
    "replicates": 5,
    "seed": 1234,
    "constraint_violation": 0.0,
    "iterations": None,
    "eps": None,
}


def _parse_literal(value, name):
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Could not parse {name}: {value}") from exc


def _parse_yaml_scalar(value):
    value = value.strip()
    if value == "":
        return ""
    lowered = value.lower()
    if lowered in {"null", "none"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def load_simple_yaml(path):
    """Load the small YAML subset used by the identity experiment config."""
    data = {}
    current_key = None
    current_list_item = None
    with open(path) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.split("#", 1)[0].rstrip()
            if not line:
                continue

            indent = len(line) - len(line.lstrip(" "))
            stripped = line.strip()

            if stripped.startswith("- "):
                if current_key is None:
                    raise ValueError(f"List item without parent on line {line_number}: {raw_line.rstrip()}")
                item_body = stripped[2:].strip()
                if not isinstance(data[current_key], list):
                    data[current_key] = []
                if item_body == "":
                    current_list_item = {}
                    data[current_key].append(current_list_item)
                elif ":" in item_body:
                    key, value = item_body.split(":", 1)
                    current_list_item = {_parse_yaml_scalar(key): _parse_yaml_scalar(value)}
                    data[current_key].append(current_list_item)
                else:
                    data[current_key].append(_parse_yaml_scalar(item_body))
                    current_list_item = None
                continue

            if ":" not in stripped:
                raise ValueError(f"Invalid YAML line {line_number}: {raw_line.rstrip()}")

            key, value = stripped.split(":", 1)
            key = _parse_yaml_scalar(key)
            value = value.strip()

            if indent == 0:
                if value == "":
                    data[key] = []
                    current_key = key
                    current_list_item = None
                else:
                    data[key] = _parse_yaml_scalar(value)
                    current_key = None
                    current_list_item = None
            elif indent in {2, 4} and current_key is not None and current_list_item is not None:
                current_list_item[key] = _parse_yaml_scalar(value)
            else:
                raise ValueError(
                    f"Unsupported YAML indentation on line {line_number}: {raw_line.rstrip()}"
                )
    return data


def load_config(path):
    config = DEFAULT_CONFIG.copy()
    if path is None:
        return config
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file does not exist: {path}")
    config.update(load_simple_yaml(path))
    return config


def _default_corrupting_dims(dimension):
    if dimension in DEFAULT_CORRUPTING_DIMS:
        return DEFAULT_CORRUPTING_DIMS[dimension]
    candidates = [1, dimension // 2, dimension - 1]
    return sorted({value for value in candidates if 0 < value < dimension})


def _resolve_corrupting_dims(raw_value, dimensions):
    if raw_value is None:
        return {dimension: _default_corrupting_dims(dimension) for dimension in dimensions}

    parsed = _parse_literal(raw_value, "--corrupting-d-values")
    if isinstance(parsed, dict):
        return {
            int(dimension): [int(value) for value in values]
            for dimension, values in parsed.items()
        }

    values = [int(value) for value in parsed]
    return {
        dimension: [value for value in values if 0 < value < dimension]
        for dimension in dimensions
    }


def build_trials_from_grid(dimensions, corrupting_dims_by_dimension, num_corrupting_values):
    trials = []
    for dimension in dimensions:
        for corrupting_dimension in corrupting_dims_by_dimension[dimension]:
            for num_corrupting in num_corrupting_values:
                trials.append({
                    "D": dimension,
                    "corrupting_D": corrupting_dimension,
                    "num_corrupting_matrices": num_corrupting,
                })
    return trials


def _normalize_trials(raw_trials):
    trials = _parse_literal(raw_trials, "trials")
    normalized = []
    for trial in trials:
        normalized.append({
            "D": int(trial["D"]),
            "corrupting_D": int(trial["corrupting_D"]),
            "num_corrupting_matrices": int(trial["num_corrupting_matrices"]),
        })
    return normalized


def _random_orthonormal_block(ambient_dimension, block_dimension, rng):
    if block_dimension == ambient_dimension:
        return ortho_group.rvs(ambient_dimension, random_state=rng)

    basis = np.zeros((ambient_dimension, block_dimension))
    if block_dimension == 1:
        basis[0, 0] = rng.choice([-1.0, 1.0])
    else:
        basis[:block_dimension, :] = ortho_group.rvs(block_dimension, random_state=rng)
    rotation = ortho_group.rvs(ambient_dimension, random_state=rng)
    return rotation @ basis


def generate_identity_instance(
    dimension,
    corrupting_dimension,
    num_corrupting,
    rng,
):
    """Generate one matrix with one true full-D block and lower-rank distractors."""
    true_block = np.eye(dimension)
    blocks = [true_block]
    block_ranges = []
    start = 0
    block_ranges.append({
        "kind": "true",
        "dimension": dimension,
        "start": start,
        "stop": start + dimension,
    })
    start += dimension

    for block_index in range(num_corrupting):
        block = _random_orthonormal_block(dimension, corrupting_dimension, rng)
        blocks.append(block)
        block_ranges.append({
            "kind": "corrupting",
            "dimension": corrupting_dimension,
            "block_index": block_index,
            "start": start,
            "stop": start + corrupting_dimension,
        })
        start += corrupting_dimension

    matrix = np.column_stack(blocks)
    true_indices = list(range(dimension))
    corrupting_indices = list(range(dimension, matrix.shape[1]))
    return matrix, true_indices, corrupting_indices, block_ranges


def _json_list(values):
    return json.dumps([int(value) for value in values])


def _selection_metrics(indices, true_set, corrupting_set):
    selected_set = set(indices)
    true_selected = sorted(selected_set & true_set)
    corrupting_selected = sorted(selected_set & corrupting_set)
    missing_true = sorted(true_set - selected_set)
    return {
        "indices": indices,
        "true_selected": true_selected,
        "corrupting_selected": corrupting_selected,
        "missing_true": missing_true,
        "num_true_selected": len(true_selected),
        "num_corrupting_selected": len(corrupting_selected),
        "strict_success": selected_set == true_set,
        "contains_true_block": true_set.issubset(selected_set),
    }


def _write_yaml_value(handle, key, value, indent=0):
    prefix = " " * indent
    if isinstance(value, dict):
        handle.write(f"{prefix}{key}:\n")
        for child_key, child_value in value.items():
            _write_yaml_value(handle, child_key, child_value, indent + 2)
    elif isinstance(value, list):
        handle.write(f"{prefix}{key}:\n")
        for item in value:
            item_prefix = " " * (indent + 2)
            if isinstance(item, dict):
                handle.write(f"{item_prefix}-\n")
                for child_key, child_value in item.items():
                    _write_yaml_value(handle, child_key, child_value, indent + 4)
            else:
                handle.write(f"{item_prefix}- {item}\n")
    elif isinstance(value, str):
        handle.write(f"{prefix}{key}: {json.dumps(value)}\n")
    elif value is None:
        handle.write(f"{prefix}{key}: null\n")
    else:
        handle.write(f"{prefix}{key}: {value}\n")


def write_simple_yaml(path, data):
    with open(path, "w") as handle:
        for key, value in data.items():
            _write_yaml_value(handle, key, value)


def run_identity_experiment(
    trials,
    replicates,
    output_dir,
    seed=1234,
    constraint_violation=0.0,
    iterations=None,
    eps=None,
    config_file=None,
    max_backward_columns=None,
    max_backward_work=None,
):
    os.makedirs(output_dir, exist_ok=True)

    records = []
    run_id = 0
    total = len(trials) * replicates

    with tqdm(total=total, desc="Identity experiment") as pbar:
        for trial_index, trial in enumerate(trials):
            dimension = trial["D"]
            corrupting_dimension = trial["corrupting_D"]
            num_corrupting = trial["num_corrupting_matrices"]
            if not 0 < corrupting_dimension < dimension:
                raise ValueError(
                    f"corrupting dimension must satisfy 0 < d_corr < D; "
                    f"got d_corr={corrupting_dimension}, D={dimension}"
                )
            for replicate in range(replicates):
                run_seed = seed + run_id
                rng = np.random.default_rng(run_seed)
                X, true_indices, corrupting_indices, block_ranges = generate_identity_instance(
                    dimension=dimension,
                    corrupting_dimension=corrupting_dimension,
                    num_corrupting=num_corrupting,
                    rng=rng,
                )

                try:
                    result = isometry_pursuit(
                        X,
                        constraint_violation=constraint_violation,
                        iterations=iterations,
                        eps=eps,
                    )
                    selected_indices = [int(value) for value in result["basis_pursuit_indices"]]
                    objective_value = float(result["objective_value"])
                    solver_iterations = result["basis_pursuit_iterations"]
                    error = None
                except Exception as exc:
                    selected_indices = []
                    objective_value = np.nan
                    solver_iterations = None
                    error = repr(exc)

                selected_set = set(selected_indices)
                true_set = set(true_indices)
                corrupting_set = set(corrupting_indices)
                true_selected = sorted(selected_set & true_set)
                corrupting_selected = sorted(selected_set & corrupting_set)
                missing_true = sorted(true_set - selected_set)

                try:
                    greedy_full_indices = [
                        int(value)
                        for value in greedy(X, _gl_loss, target_dimension=dimension,
                                            random_state=run_seed)
                    ]
                    greedy_full_loss = float(_gl_loss(X[:, greedy_full_indices]))
                    greedy_full_error = None
                except Exception as exc:
                    greedy_full_indices = []
                    greedy_full_loss = np.nan
                    greedy_full_error = repr(exc)

                try:
                    forward_backward_full_indices = [
                        int(value)
                        for value in forward_backward(
                            X,
                            _gl_loss,
                            target_dimension=dimension,
                            selected_indices=greedy_full_indices,
                            random_state=run_seed,
                        )
                    ]
                    forward_backward_full_loss = float(
                        _gl_loss(X[:, forward_backward_full_indices])
                    )
                    forward_backward_full_error = None
                except Exception as exc:
                    forward_backward_full_indices = []
                    forward_backward_full_loss = np.nan
                    forward_backward_full_error = repr(exc)

                if len(selected_indices) >= dimension:
                    try:
                        selected_X = X[:, selected_indices]
                        local_indices = greedy(
                            selected_X,
                            _gl_loss,
                            target_dimension=dimension,
                            random_state=run_seed,
                        )
                        greedy_restricted_indices = [
                            int(selected_indices[value]) for value in local_indices
                        ]
                        greedy_restricted_loss = float(
                            _gl_loss(X[:, greedy_restricted_indices])
                        )
                        greedy_restricted_error = None
                    except Exception as exc:
                        greedy_restricted_indices = []
                        greedy_restricted_loss = np.nan
                        greedy_restricted_error = repr(exc)

                    try:
                        selected_X = X[:, selected_indices]
                        local_indices = forward_backward(
                            selected_X,
                            _gl_loss,
                            target_dimension=dimension,
                            selected_indices=local_indices,
                            random_state=run_seed,
                        )
                        forward_backward_restricted_indices = [
                            int(selected_indices[value]) for value in local_indices
                        ]
                        forward_backward_restricted_loss = float(
                            _gl_loss(X[:, forward_backward_restricted_indices])
                        )
                        forward_backward_restricted_error = None
                    except Exception as exc:
                        forward_backward_restricted_indices = []
                        forward_backward_restricted_loss = np.nan
                        forward_backward_restricted_error = repr(exc)
                else:
                    greedy_restricted_indices = []
                    greedy_restricted_loss = np.nan
                    greedy_restricted_error = "IP support has fewer than D columns"
                    forward_backward_restricted_indices = []
                    forward_backward_restricted_loss = np.nan
                    forward_backward_restricted_error = "IP support has fewer than D columns"

                greedy_full = _selection_metrics(
                    greedy_full_indices,
                    true_set,
                    corrupting_set,
                )
                forward_backward_full = _selection_metrics(
                    forward_backward_full_indices,
                    true_set,
                    corrupting_set,
                )
                greedy_restricted = _selection_metrics(
                    greedy_restricted_indices,
                    true_set,
                    corrupting_set,
                )
                forward_backward_restricted = _selection_metrics(
                    forward_backward_restricted_indices,
                    true_set,
                    corrupting_set,
                )

                backward_methods = {}
                for prefix, dictionary in (
                    ("backward_swap_full", list(range(X.shape[1]))),
                    ("backward_swap_restricted", selected_indices),
                ):
                    backward_work = dimension ** 2 * sum(
                        k * (k - 1)
                        for k in range(dimension + 1, len(dictionary) + 1)
                    )
                    if (
                        max_backward_columns is not None
                        and len(dictionary) > max_backward_columns
                    ):
                        chosen, method_loss = [], np.nan
                        method_error = (
                            "Skipped: backward initialization has "
                            f"{len(dictionary)} columns, exceeding the "
                            f"configured limit of {max_backward_columns}"
                        )
                    elif (
                        max_backward_work is not None
                        and backward_work > max_backward_work
                    ):
                        chosen, method_loss = [], np.nan
                        method_error = (
                            "Skipped: backward arithmetic-work proxy is "
                            f"{backward_work}, exceeding the configured limit "
                            f"of {max_backward_work}"
                        )
                    else:
                        try:
                            local_indices = greedy_swap(
                                X[:, dictionary], _gl_loss,
                                target_dimension=dimension,
                                initialization="backward",
                                random_state=run_seed,
                            )
                            chosen = [int(dictionary[j]) for j in local_indices]
                            method_loss = float(_gl_loss(X[:, chosen]))
                            method_error = None
                        except Exception as exc:
                            chosen, method_loss, method_error = [], np.nan, repr(exc)
                    metrics = _selection_metrics(chosen, true_set, corrupting_set)
                    backward_methods.update({
                        f"{prefix}_indices": chosen,
                        f"{prefix}_loss": method_loss,
                        f"{prefix}_strict_success": metrics["strict_success"],
                        f"{prefix}_contains_true_block": metrics["contains_true_block"],
                        f"{prefix}_error": method_error,
                        f"{prefix}_work_proxy": backward_work,
                    })

                records.append({
                    **backward_methods,
                    "run_id": run_id,
                    "trial_id": trial_index,
                    "seed": run_seed,
                    "D": dimension,
                    "corrupting_D": corrupting_dimension,
                    "num_corrupting_matrices": num_corrupting,
                    "replicate": replicate,
                    "total_columns": int(X.shape[1]),
                    "constraint_violation": constraint_violation,
                    "iterations": iterations,
                    "eps": eps,
                    "max_backward_columns": max_backward_columns,
                    "max_backward_work": max_backward_work,
                    "true_indices": true_indices,
                    "corrupting_indices": corrupting_indices,
                    "selected_indices": selected_indices,
                    "true_selected": true_selected,
                    "corrupting_selected": corrupting_selected,
                    "missing_true_indices": missing_true,
                    "support_cardinality": len(selected_indices),
                    "num_true_selected": len(true_selected),
                    "num_corrupting_selected": len(corrupting_selected),
                    "strict_success": selected_set == true_set,
                    "contains_true_block": true_set.issubset(selected_set),
                    "objective_value": objective_value,
                    "basis_pursuit_iterations": solver_iterations,
                    "greedy_full_indices": greedy_full["indices"],
                    "greedy_full_loss": greedy_full_loss,
                    "greedy_full_true_selected": greedy_full["true_selected"],
                    "greedy_full_corrupting_selected": greedy_full["corrupting_selected"],
                    "greedy_full_missing_true_indices": greedy_full["missing_true"],
                    "greedy_full_num_true_selected": greedy_full["num_true_selected"],
                    "greedy_full_num_corrupting_selected": greedy_full["num_corrupting_selected"],
                    "greedy_full_strict_success": greedy_full["strict_success"],
                    "greedy_full_contains_true_block": greedy_full["contains_true_block"],
                    "greedy_full_contained_in_ip": set(greedy_full_indices).issubset(selected_set),
                    "greedy_full_error": greedy_full_error,
                    "forward_backward_full_indices": forward_backward_full["indices"],
                    "forward_backward_full_loss": forward_backward_full_loss,
                    "forward_backward_full_true_selected": forward_backward_full["true_selected"],
                    "forward_backward_full_corrupting_selected": forward_backward_full["corrupting_selected"],
                    "forward_backward_full_missing_true_indices": forward_backward_full["missing_true"],
                    "forward_backward_full_num_true_selected": forward_backward_full["num_true_selected"],
                    "forward_backward_full_num_corrupting_selected": forward_backward_full["num_corrupting_selected"],
                    "forward_backward_full_strict_success": forward_backward_full["strict_success"],
                    "forward_backward_full_contains_true_block": forward_backward_full["contains_true_block"],
                    "forward_backward_full_contained_in_ip": set(forward_backward_full_indices).issubset(selected_set),
                    "forward_backward_full_error": forward_backward_full_error,
                    "greedy_restricted_indices": greedy_restricted["indices"],
                    "greedy_restricted_loss": greedy_restricted_loss,
                    "greedy_restricted_true_selected": greedy_restricted["true_selected"],
                    "greedy_restricted_corrupting_selected": greedy_restricted["corrupting_selected"],
                    "greedy_restricted_missing_true_indices": greedy_restricted["missing_true"],
                    "greedy_restricted_num_true_selected": greedy_restricted["num_true_selected"],
                    "greedy_restricted_num_corrupting_selected": greedy_restricted["num_corrupting_selected"],
                    "greedy_restricted_strict_success": greedy_restricted["strict_success"],
                    "greedy_restricted_contains_true_block": greedy_restricted["contains_true_block"],
                    "greedy_restricted_error": greedy_restricted_error,
                    "forward_backward_restricted_indices": forward_backward_restricted["indices"],
                    "forward_backward_restricted_loss": forward_backward_restricted_loss,
                    "forward_backward_restricted_true_selected": forward_backward_restricted["true_selected"],
                    "forward_backward_restricted_corrupting_selected": forward_backward_restricted["corrupting_selected"],
                    "forward_backward_restricted_missing_true_indices": forward_backward_restricted["missing_true"],
                    "forward_backward_restricted_num_true_selected": forward_backward_restricted["num_true_selected"],
                    "forward_backward_restricted_num_corrupting_selected": forward_backward_restricted["num_corrupting_selected"],
                    "forward_backward_restricted_strict_success": forward_backward_restricted["strict_success"],
                    "forward_backward_restricted_contains_true_block": forward_backward_restricted["contains_true_block"],
                    "forward_backward_restricted_error": forward_backward_restricted_error,
                    "block_ranges": block_ranges,
                    "error": error,
                })
                run_id += 1
                pbar.update(1)

    runs_json = os.path.join(output_dir, "identity_experiment_runs.json")
    with open(runs_json, "w") as handle:
        json.dump({"runs": records}, handle, indent=2)

    csv_records = []
    for record in records:
        csv_record = {
            key: value
            for key, value in record.items()
            if key not in {
                "true_indices",
                "corrupting_indices",
                "selected_indices",
                "true_selected",
                "corrupting_selected",
                "missing_true_indices",
                "greedy_full_indices",
                "greedy_full_true_selected",
                "greedy_full_corrupting_selected",
                "greedy_full_missing_true_indices",
                "forward_backward_full_indices",
                "forward_backward_full_true_selected",
                "forward_backward_full_corrupting_selected",
                "forward_backward_full_missing_true_indices",
                "greedy_restricted_indices",
                "greedy_restricted_true_selected",
                "greedy_restricted_corrupting_selected",
                "greedy_restricted_missing_true_indices",
                "forward_backward_restricted_indices",
                "forward_backward_restricted_true_selected",
                "forward_backward_restricted_corrupting_selected",
                "forward_backward_restricted_missing_true_indices",
                "block_ranges",
            }
        }
        for key in [
            "true_indices",
            "corrupting_indices",
            "selected_indices",
            "true_selected",
            "corrupting_selected",
            "missing_true_indices",
            "greedy_full_indices",
            "greedy_full_true_selected",
            "greedy_full_corrupting_selected",
            "greedy_full_missing_true_indices",
            "forward_backward_full_indices",
            "forward_backward_full_true_selected",
            "forward_backward_full_corrupting_selected",
            "forward_backward_full_missing_true_indices",
            "greedy_restricted_indices",
            "greedy_restricted_true_selected",
            "greedy_restricted_corrupting_selected",
            "greedy_restricted_missing_true_indices",
            "forward_backward_restricted_indices",
            "forward_backward_restricted_true_selected",
            "forward_backward_restricted_corrupting_selected",
            "forward_backward_restricted_missing_true_indices",
        ]:
            csv_record[key] = _json_list(record[key])
        csv_record["block_ranges_json"] = json.dumps(record["block_ranges"])
        csv_records.append(csv_record)

    runs_csv = os.path.join(output_dir, "identity_experiment_runs.csv")
    pd.DataFrame(csv_records).to_csv(runs_csv, index=False)

    summary_df = (
        pd.DataFrame(csv_records)
        .groupby(["D", "corrupting_D", "num_corrupting_matrices"], as_index=False)
        .agg(
            runs=("run_id", "count"),
            strict_success_rate=("strict_success", "mean"),
            contains_true_block_rate=("contains_true_block", "mean"),
            mean_support_cardinality=("support_cardinality", "mean"),
            mean_num_true_selected=("num_true_selected", "mean"),
            mean_num_corrupting_selected=("num_corrupting_selected", "mean"),
            greedy_full_strict_success_rate=("greedy_full_strict_success", "mean"),
            greedy_full_contains_true_block_rate=("greedy_full_contains_true_block", "mean"),
            greedy_full_contained_in_ip_rate=("greedy_full_contained_in_ip", "mean"),
            greedy_restricted_strict_success_rate=("greedy_restricted_strict_success", "mean"),
            greedy_restricted_contains_true_block_rate=("greedy_restricted_contains_true_block", "mean"),
            forward_backward_full_strict_success_rate=("forward_backward_full_strict_success", "mean"),
            forward_backward_full_contains_true_block_rate=("forward_backward_full_contains_true_block", "mean"),
            forward_backward_full_contained_in_ip_rate=("forward_backward_full_contained_in_ip", "mean"),
            forward_backward_restricted_strict_success_rate=("forward_backward_restricted_strict_success", "mean"),
            forward_backward_restricted_contains_true_block_rate=("forward_backward_restricted_contains_true_block", "mean"),
            mean_greedy_full_loss=("greedy_full_loss", "mean"),
            mean_greedy_restricted_loss=("greedy_restricted_loss", "mean"),
            mean_forward_backward_full_loss=("forward_backward_full_loss", "mean"),
            mean_forward_backward_restricted_loss=("forward_backward_restricted_loss", "mean"),
            backward_swap_full_strict_success_rate=("backward_swap_full_strict_success", "mean"),
            backward_swap_full_contains_true_block_rate=("backward_swap_full_contains_true_block", "mean"),
            backward_swap_restricted_strict_success_rate=("backward_swap_restricted_strict_success", "mean"),
            backward_swap_restricted_contains_true_block_rate=("backward_swap_restricted_contains_true_block", "mean"),
            mean_backward_swap_full_loss=("backward_swap_full_loss", "mean"),
            mean_backward_swap_restricted_loss=("backward_swap_restricted_loss", "mean"),
        )
    )
    summary_csv = os.path.join(output_dir, "identity_experiment_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    metadata = {
        "experiment": "identity_matrices",
        "description": (
            "Synthetic matrices with one true full-rank orthonormal D-block and "
            "lower-dimensional corrupting orthonormal blocks."
        ),
        "parameters": {
            "trials": trials,
            "replicates": replicates,
            "seed": seed,
            "constraint_violation": constraint_violation,
            "iterations": iterations,
            "eps": eps,
            "max_backward_columns": max_backward_columns,
            "max_backward_work": max_backward_work,
            "config_file": config_file,
        },
        "outputs": {
            "runs_json": runs_json,
            "runs_csv": runs_csv,
            "summary_csv": summary_csv,
        },
        "record_fields": {
            "strict_success": "selected_indices equals true_indices exactly",
            "contains_true_block": "all true indices were selected, possibly with extras",
            "greedy_full_indices": "greedy-selected D columns from the full dictionary",
            "greedy_restricted_indices": "greedy-selected D columns after restricting to selected_indices",
            "forward_backward_full_indices": "forward-backward-selected D columns from the full dictionary",
            "forward_backward_restricted_indices": "forward-backward-selected D columns after restricting to selected_indices",
            "greedy_full_contained_in_ip": "all full-greedy indices are contained in selected_indices",
            "forward_backward_full_contained_in_ip": "all full-forward-backward indices are contained in selected_indices",
            "true_indices": "columns of the full D-dimensional block",
            "corrupting_indices": "columns from lower-dimensional distractor blocks",
        },
    }
    metadata_yaml = os.path.join(output_dir, "identity_experiment_metadata.yaml")
    write_simple_yaml(metadata_yaml, metadata)

    return {
        "runs_json": runs_json,
        "runs_csv": runs_csv,
        "summary_csv": summary_csv,
        "metadata_yaml": metadata_yaml,
    }


def main():
    parser = argparse.ArgumentParser(description="Run synthetic identity-block Isometry Pursuit experiments")
    parser.add_argument(
        "--config",
        default="configs/identity_experiment.yaml",
        help="YAML config file with experiment defaults",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--trials", default=None, help="List of trial triples")
    parser.add_argument("--D-values", default=None, help="List of ambient dimensions")
    parser.add_argument(
        "--corrupting-d-values",
        default=None,
        help="Dict by D or shared list, e.g. '{4:[1,2,3],8:[1,4,7]}' or '[1,2,3]'",
    )
    parser.add_argument(
        "--num-corrupting-values",
        default=None,
        help="List of numbers of corrupting matrices",
    )
    parser.add_argument("--replicates", type=int, default=None, help="Replicates per configuration")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--constraint-violation", type=float, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument(
        "--max-backward-columns",
        type=int,
        default=None,
        help="Skip backward-initialized methods when their dictionary exceeds this size",
    )
    parser.add_argument(
        "--max-backward-work",
        type=float,
        default=None,
        help="Skip backward methods above this arithmetic-work proxy",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    cli_overrides = {
        "output_dir": args.output_dir,
        "trials": args.trials,
        "D_values": args.D_values,
        "corrupting_D_values": args.corrupting_d_values,
        "num_corrupting_values": args.num_corrupting_values,
        "replicates": args.replicates,
        "seed": args.seed,
        "constraint_violation": args.constraint_violation,
        "iterations": args.iterations,
        "eps": args.eps,
        "max_backward_columns": args.max_backward_columns,
        "max_backward_work": args.max_backward_work,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            config[key] = value

    if config.get("trials") is not None:
        trials = _normalize_trials(config["trials"])
    else:
        dimensions = [int(value) for value in _parse_literal(config["D_values"], "D_values")]
        num_corrupting_values = [
            int(value) for value in _parse_literal(config["num_corrupting_values"], "num_corrupting_values")
        ]
        corrupting_dims_by_dimension = _resolve_corrupting_dims(config["corrupting_D_values"], dimensions)
        trials = build_trials_from_grid(dimensions, corrupting_dims_by_dimension, num_corrupting_values)

    outputs = run_identity_experiment(
        trials=trials,
        replicates=int(config["replicates"]),
        output_dir=config["output_dir"],
        seed=int(config["seed"]),
        constraint_violation=float(config["constraint_violation"]),
        iterations=config["iterations"],
        eps=config["eps"],
        config_file=args.config,
        max_backward_columns=config.get("max_backward_columns"),
        max_backward_work=config.get("max_backward_work"),
    )

    print("Identity experiment complete.")
    for label, path in outputs.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
