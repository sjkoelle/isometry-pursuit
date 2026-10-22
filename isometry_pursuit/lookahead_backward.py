"""Rank-safe beam look-ahead for backward subset deletion."""

import time

import numpy as np


def lookahead_backward_deletion(
    matrix,
    loss,
    target_dimension=None,
    selected_indices=None,
    depth=2,
    beam_width=8,
):
    """Delete columns using a replanned beam search over future removals.

    At each accepted deletion, the first search layer evaluates every possible
    removal.  The best ``beam_width`` finite-loss nodes are retained, expanded
    until ``depth`` deletions have been considered, and the first deletion on
    the lowest-loss terminal path is accepted.  The search is then restarted
    from the new support.  Returning infinity from ``loss`` excludes a node,
    which preserves rank validity when the loss enforces it.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if target_dimension is None:
        target_dimension = matrix.shape[0]
    if int(target_dimension) != target_dimension:
        raise ValueError("target_dimension must be an integer")
    if int(depth) != depth or depth < 1:
        raise ValueError("depth must be a positive integer")
    if int(beam_width) != beam_width or beam_width < 1:
        raise ValueError("beam_width must be a positive integer")
    target_dimension = int(target_dimension)
    depth = int(depth)
    beam_width = int(beam_width)

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

    candidate_evaluations = 0
    arithmetic_work_proxy = 0
    trace = []
    started = time.perf_counter()

    while len(selected) > target_dimension:
        horizon = min(depth, len(selected) - target_dimension)
        nodes = [
            {
                "support": selected,
                "first_removed": None,
                "first_loss": None,
                "path": [],
            }
        ]
        layer_counts = []
        for layer in range(horizon):
            expanded = []
            for node in nodes:
                support = node["support"]
                for position, removed in enumerate(support):
                    candidate = support[:position] + support[position + 1 :]
                    value = float(
                        loss(
                            matrix[:, candidate],
                            indices=np.asarray(candidate, dtype=int),
                        )
                    )
                    candidate_evaluations += 1
                    arithmetic_work_proxy += matrix.shape[0] ** 2 * len(candidate)
                    if np.isfinite(value):
                        expanded.append(
                            {
                                "support": candidate,
                                "loss": value,
                                "first_removed": (
                                    int(removed)
                                    if node["first_removed"] is None
                                    else node["first_removed"]
                                ),
                                "first_loss": (
                                    value
                                    if node["first_loss"] is None
                                    else node["first_loss"]
                                ),
                                "path": node["path"] + [int(removed)],
                            }
                        )
            if not expanded:
                raise ValueError("no finite-loss removal preserves feasibility")
            expanded.sort(
                key=lambda node: (
                    node["loss"],
                    tuple(node["support"]),
                    tuple(node["path"]),
                )
            )
            layer_counts.append(len(expanded))
            nodes = expanded[:beam_width]

        best = nodes[0]
        removed = best["first_removed"]
        selected = [index for index in selected if index != removed]
        immediate_loss = float(best["first_loss"])
        trace.append(
            {
                "size": len(selected),
                "removed_index": removed,
                "immediate_loss": immediate_loss,
                "planned_terminal_loss": float(best["loss"]),
                "planned_path": best["path"],
                "horizon": horizon,
                "expanded_nodes_by_layer": layer_counts,
            }
        )

    final_loss = (
        trace[-1]["immediate_loss"]
        if trace
        else float(loss(matrix[:, selected], indices=np.asarray(selected, dtype=int)))
    )
    return {
        "indices": [int(index) for index in selected],
        "loss": final_loss,
        "depth": depth,
        "beam_width": beam_width,
        "candidate_evaluations": int(candidate_evaluations),
        "arithmetic_work_proxy": int(arithmetic_work_proxy),
        "runtime_seconds": time.perf_counter() - started,
        "trace": trace,
    }
