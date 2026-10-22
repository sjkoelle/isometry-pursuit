"""Rank-safe batched backward elimination for subset selection."""

import math

import numpy as np


def _batch_width(batch_size, excess):
    if batch_size == "halving":
        return max(1, math.ceil(excess / 2))
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)):
        raise ValueError("batch_size must be a positive integer or 'halving'")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return min(int(batch_size), excess)


def _kernel_proxy(ambient_dimension, num_columns):
    """Leading-order SVD proxy for a D by q matrix."""
    return ambient_dimension * num_columns * min(ambient_dimension, num_columns)


def batched_backward_elimination(
    matrix,
    loss,
    target_dimension=None,
    selected_indices=None,
    batch_size=2,
    random_state=None,
):
    """Delete batches using stale one-deletion scores and rank-safe acceptance.

    At each stage, all one-column removals are scored. Candidates are ordered by
    their resulting loss, then accepted into the batch in that order provided
    the jointly retained matrix has maximal possible rank. Scores are not
    recomputed within a batch. ``batch_size='halving'`` removes half of the
    current excess above the target at each stage.

    Returns a dictionary containing the selected indices, stage history, loss
    evaluation and rank-check counts, and a leading-order SVD work proxy. The
    proxy counts both loss evaluations and the extra rank checks.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    ambient_dimension, dictionary_dimension = matrix.shape
    if target_dimension is None:
        target_dimension = ambient_dimension
    if (
        isinstance(target_dimension, bool)
        or int(target_dimension) != target_dimension
        or not 1 <= target_dimension <= dictionary_dimension
    ):
        raise ValueError("target_dimension must be between 1 and the number of columns")
    target_dimension = int(target_dimension)
    _batch_width(batch_size, max(1, dictionary_dimension - target_dimension))

    if selected_indices is None:
        selected = list(range(dictionary_dimension))
    else:
        selected = [int(index) for index in selected_indices]
    if len(selected) != len(set(selected)):
        raise ValueError("selected_indices must not contain duplicates")
    if any(index < 0 or index >= dictionary_dimension for index in selected):
        raise ValueError("selected_indices contains an out-of-range column")
    if len(selected) < target_dimension:
        raise ValueError("selected_indices must contain at least target_dimension columns")

    rng = np.random.default_rng(random_state)
    loss_evaluations = 0
    rank_checks = 0
    arithmetic_work_proxy = 0
    history = []

    while len(selected) > target_dimension:
        current_size = len(selected)
        desired = _batch_width(batch_size, current_size - target_dimension)
        scored = []
        score_order = rng.permutation(current_size)
        for position in score_order:
            candidate = selected[:position] + selected[position + 1 :]
            candidate_loss = float(
                loss(matrix[:, candidate], indices=np.asarray(candidate, dtype=int))
            )
            loss_evaluations += 1
            arithmetic_work_proxy += _kernel_proxy(ambient_dimension, len(candidate))
            scored.append((candidate_loss, int(position), selected[position]))
        scored.sort(key=lambda item: item[0])

        removed = []
        for candidate_loss, _, index in scored:
            if not np.isfinite(candidate_loss):
                continue
            trial = [value for value in selected if value not in set(removed + [index])]
            required_rank = min(ambient_dimension, len(trial))
            rank = int(np.linalg.matrix_rank(matrix[:, trial]))
            rank_checks += 1
            arithmetic_work_proxy += _kernel_proxy(ambient_dimension, len(trial))
            if rank == required_rank:
                removed.append(index)
                if len(removed) == desired:
                    break

        if not removed:
            raise RuntimeError("no rank-safe deletion was available")
        selected = [index for index in selected if index not in set(removed)]
        history.append(
            {
                "size_before": current_size,
                "size_after": len(selected),
                "desired_batch_size": desired,
                "actual_batch_size": len(removed),
                "removed_indices": removed,
            }
        )

    return {
        "selected_indices": selected,
        "history": history,
        "num_stages": len(history),
        "loss_evaluations": loss_evaluations,
        "rank_checks": rank_checks,
        "arithmetic_work_proxy": int(arithmetic_work_proxy),
    }
