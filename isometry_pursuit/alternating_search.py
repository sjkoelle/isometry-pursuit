"""Expansion-contraction local search for fixed-cardinality subsets."""

import numpy as np


def _work_proxy(ambient_dimension, num_columns):
    return ambient_dimension * num_columns * min(ambient_dimension, num_columns)


def alternating_add_delete(
    matrix,
    loss,
    selected_indices,
    target_dimension=None,
    expansion_width=2,
    max_cycles=None,
    improvement_tol=0.0,
    random_state=None,
):
    """Expand a target-size support and greedily contract it again.

    Each cycle greedily adds ``expansion_width`` distinct columns, temporarily
    allowing support sizes above the target. It then greedily removes columns
    until the target size is restored. The complete cycle is accepted only when
    its final loss strictly improves by more than ``improvement_tol``.

    Unlike exhaustive one-swap search, additions are chosen using losses at the
    expanded cardinalities and contractions are chosen sequentially. For width
    greater than one, a cycle may coordinate several replacements.
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
    if (
        isinstance(expansion_width, bool)
        or not isinstance(expansion_width, (int, np.integer))
        or expansion_width < 1
    ):
        raise ValueError("expansion_width must be a positive integer")
    if max_cycles is None:
        max_cycles = dictionary_dimension
    if isinstance(max_cycles, bool) or int(max_cycles) != max_cycles or max_cycles < 1:
        raise ValueError("max_cycles must be a positive integer")
    if improvement_tol < 0:
        raise ValueError("improvement_tol must be nonnegative")

    selected = [int(index) for index in selected_indices]
    if len(selected) != target_dimension or len(selected) != len(set(selected)):
        raise ValueError("selected_indices must contain target_dimension distinct columns")
    if any(index < 0 or index >= dictionary_dimension for index in selected):
        raise ValueError("selected_indices contains an out-of-range column")
    required_rank = min(ambient_dimension, target_dimension)
    if np.linalg.matrix_rank(matrix[:, selected]) != required_rank:
        raise ValueError("selected_indices must have maximal possible rank")

    rng = np.random.default_rng(random_state)
    loss_evaluations = 1
    candidate_evaluations = 0
    rank_checks = 1
    arithmetic_work_proxy = 2 * _work_proxy(ambient_dimension, target_dimension)
    current_loss = float(
        loss(matrix[:, selected], indices=np.asarray(selected, dtype=int))
    )
    if not np.isfinite(current_loss):
        raise ValueError("selected_indices must have finite loss")
    history = []

    available_expansion = min(int(expansion_width), dictionary_dimension - target_dimension)
    if available_expansion == 0:
        return _result(
            selected, current_loss, history, loss_evaluations,
            candidate_evaluations, rank_checks, arithmetic_work_proxy,
        )

    for cycle in range(int(max_cycles)):
        expanded = selected.copy()
        additions = []
        for _ in range(available_expansion):
            candidates = [index for index in range(dictionary_dimension) if index not in expanded]
            rng.shuffle(candidates)
            scored = []
            for index in candidates:
                trial = expanded + [index]
                value = float(loss(matrix[:, trial], indices=np.asarray(trial, dtype=int)))
                loss_evaluations += 1
                candidate_evaluations += 1
                arithmetic_work_proxy += _work_proxy(ambient_dimension, len(trial))
                if np.isfinite(value):
                    scored.append((value, index))
            if not scored:
                raise RuntimeError("no finite expansion candidate was available")
            best_value = min(value for value, _ in scored)
            best = [index for value, index in scored if value == best_value]
            added = int(rng.choice(best))
            expanded.append(added)
            additions.append(added)

        contracted = expanded.copy()
        contractions = []
        contraction_loss = np.nan
        while len(contracted) > target_dimension:
            positions = list(rng.permutation(len(contracted)))
            scored = []
            for position in positions:
                trial = contracted[:position] + contracted[position + 1 :]
                rank = int(np.linalg.matrix_rank(matrix[:, trial]))
                rank_checks += 1
                arithmetic_work_proxy += _work_proxy(ambient_dimension, len(trial))
                if rank != min(ambient_dimension, len(trial)):
                    continue
                value = float(loss(matrix[:, trial], indices=np.asarray(trial, dtype=int)))
                loss_evaluations += 1
                candidate_evaluations += 1
                arithmetic_work_proxy += _work_proxy(ambient_dimension, len(trial))
                if np.isfinite(value):
                    scored.append((value, int(position)))
            if not scored:
                raise RuntimeError("no rank-safe contraction candidate was available")
            best_value = min(value for value, _ in scored)
            best_positions = [position for value, position in scored if value == best_value]
            removed_position = int(rng.choice(best_positions))
            contractions.append(contracted[removed_position])
            contracted.pop(removed_position)
            contraction_loss = best_value

        accepted = contraction_loss < current_loss - improvement_tol
        history.append(
            {
                "cycle": cycle + 1,
                "additions": additions,
                "contractions": contractions,
                "proposed_indices": contracted.copy(),
                "proposed_loss": float(contraction_loss),
                "accepted": bool(accepted),
            }
        )
        if not accepted:
            break
        selected = contracted
        current_loss = float(contraction_loss)

    return _result(
        selected, current_loss, history, loss_evaluations,
        candidate_evaluations, rank_checks, arithmetic_work_proxy,
    )


def _result(
    selected, loss_value, history, loss_evaluations, candidate_evaluations,
    rank_checks, arithmetic_work_proxy,
):
    return {
        "selected_indices": selected,
        "loss": float(loss_value),
        "history": history,
        "cycles": len(history),
        "accepted_cycles": sum(step["accepted"] for step in history),
        "loss_evaluations": loss_evaluations,
        "candidate_evaluations": candidate_evaluations,
        "rank_checks": rank_checks,
        "arithmetic_work_proxy": int(arithmetic_work_proxy),
    }
