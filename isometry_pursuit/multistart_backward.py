"""Randomized multi-start backward deletion for fixed-cardinality selection."""

import time

import numpy as np


def backward_work_counts(ambient_dimension, initial_size, target_dimension):
    """Return candidate evaluations and the shape-aware arithmetic-work proxy."""
    sizes = range(target_dimension + 1, initial_size + 1)
    evaluations = sum(sizes)
    proxy = ambient_dimension**2 * sum(k * (k - 1) for k in sizes)
    return int(evaluations), int(proxy)


def randomized_backward_deletion(
    matrix,
    loss,
    target_dimension=None,
    selected_indices=None,
    relative_tolerance=0.0,
    random_state=None,
):
    """Run one randomized near-greedy backward-deletion path.

    At each stage all removals are evaluated.  A removal is sampled uniformly
    from those with loss at most ``best * (1 + relative_tolerance)``.  More
    generally, the additive allowance is ``relative_tolerance * abs(best)``.
    Only finite-loss candidates are eligible, so a loss function that returns
    infinity for rank-deficient subsets preserves rank validity.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if target_dimension is None:
        target_dimension = matrix.shape[0]
    if int(target_dimension) != target_dimension:
        raise ValueError("target_dimension must be an integer")
    target_dimension = int(target_dimension)
    if relative_tolerance < 0:
        raise ValueError("relative_tolerance must be nonnegative")

    if selected_indices is None:
        selected = list(range(matrix.shape[1]))
    else:
        selected = [int(index) for index in selected_indices]
    if len(set(selected)) != len(selected):
        raise ValueError("selected_indices must not contain duplicates")
    if any(index < 0 or index >= matrix.shape[1] for index in selected):
        raise ValueError("selected_indices contains an out-of-range index")
    if not 1 <= target_dimension <= len(selected):
        raise ValueError("target_dimension must be between 1 and the initial support size")

    rng = np.random.default_rng(random_state)
    trace = []
    candidate_evaluations = 0
    while len(selected) > target_dimension:
        candidates = []
        candidate_losses = []
        for position in range(len(selected)):
            candidate = selected[:position] + selected[position + 1 :]
            value = float(
                loss(matrix[:, candidate], indices=np.asarray(candidate, dtype=int))
            )
            candidate_evaluations += 1
            if np.isfinite(value):
                candidates.append(candidate)
                candidate_losses.append(value)

        if not candidates:
            raise ValueError("no finite-loss removal preserves feasibility")
        values = np.asarray(candidate_losses)
        best = float(values.min())
        allowance = relative_tolerance * abs(best)
        eligible = np.flatnonzero(values <= best + allowance)
        choice = int(rng.choice(eligible))
        previous = selected
        selected = candidates[choice]
        trace.append(
            {
                "size": len(selected),
                "removed_index": int(next(iter(set(previous) - set(selected)))),
                "chosen_loss": float(values[choice]),
                "best_loss": best,
                "eligible_removals": int(eligible.size),
            }
        )

    final_loss = float(loss(matrix[:, selected], indices=np.asarray(selected, dtype=int)))
    _, arithmetic_proxy = backward_work_counts(
        matrix.shape[0],
        len(selected_indices) if selected_indices is not None else matrix.shape[1],
        target_dimension,
    )
    return {
        "indices": [int(index) for index in selected],
        "loss": final_loss,
        "candidate_evaluations": candidate_evaluations,
        "arithmetic_work_proxy": arithmetic_proxy,
        "trace": trace,
    }


def multistart_backward_deletion(
    matrix,
    loss,
    target_dimension=None,
    selected_indices=None,
    num_starts=8,
    relative_tolerance=0.01,
    random_state=None,
):
    """Return the lowest-loss result from one greedy and randomized starts.

    Start zero uses zero tolerance and therefore follows an ordinary greedy
    deletion path (with randomized exact-tie breaking).  Remaining starts use
    ``relative_tolerance``.  Final ties retain the earlier result, so extra
    starts cannot make the returned objective worse than start zero.
    """
    if int(num_starts) != num_starts or num_starts < 1:
        raise ValueError("num_starts must be a positive integer")
    if relative_tolerance < 0:
        raise ValueError("relative_tolerance must be nonnegative")

    seed_sequence = np.random.SeedSequence(random_state)
    child_seeds = seed_sequence.spawn(int(num_starts))
    start_results = []
    start_time = time.perf_counter()
    for start_index, child_seed in enumerate(child_seeds):
        result = randomized_backward_deletion(
            matrix,
            loss,
            target_dimension=target_dimension,
            selected_indices=selected_indices,
            relative_tolerance=0.0 if start_index == 0 else relative_tolerance,
            random_state=child_seed,
        )
        result["start"] = start_index
        result["relative_tolerance"] = (
            0.0 if start_index == 0 else float(relative_tolerance)
        )
        start_results.append(result)

    best_index = min(range(len(start_results)), key=lambda i: start_results[i]["loss"])
    best = start_results[best_index]
    return {
        "indices": best["indices"],
        "loss": best["loss"],
        "best_start": best_index,
        "num_starts": int(num_starts),
        "relative_tolerance": float(relative_tolerance),
        "candidate_evaluations": int(
            sum(result["candidate_evaluations"] for result in start_results)
        ),
        "arithmetic_work_proxy": int(
            sum(result["arithmetic_work_proxy"] for result in start_results)
        ),
        "runtime_seconds": time.perf_counter() - start_time,
        "starts": start_results,
    }
