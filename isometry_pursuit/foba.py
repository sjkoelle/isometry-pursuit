"""Coordinate FoBa from Liu, Fujimaki and Ye (ICML 2014), Algorithm 1.

This is distinct from algorithm.forward_backward's fixed-size subset swaps.
"""

import numpy as np


def foba(objective, restricted_minimizer, dimension, *, variant="obj",
         coordinate_minimizer=None, gradient=None, delta=1e-8, epsilon=1e-8,
         max_iterations=1000, target_dimension=None):
    """Minimize a smooth convex coefficient objective using FoBa-obj or -gdt.

    ``restricted_minimizer(support)`` returns a full length coefficient vector,
    zero outside support. ``coordinate_minimizer(beta, index)`` returns the
    minimizing *increment* along that coordinate, keeping all others fixed.
    These callbacks must solve their subproblems accurately. ``gradient(beta)``
    is required for gdt; coordinate_minimizer is required for obj.

    delta thresholds one-coordinate objective reduction; epsilon thresholds
    the gradient infinity norm when target_dimension is None (paper stopping).
    Otherwise these stopping thresholds are ignored: continue until the selected
    set has target_dimension coordinates after backward pruning. This counts
    selected coordinates, including any whose fitted coefficient is zero.
    This target-size stopping rule is an adaptation of the paper. Pruning can
    prevent reaching the target; the iteration cap then reports non-convergence.
    The safety iteration cap is reported separately from convergence. History
    records each accepted addition/deletion, including pre-refit deletion cost.
    """
    if variant not in ("obj", "gdt"):
        raise ValueError("variant must be 'obj' or 'gdt'")
    if dimension < 1 or int(dimension) != dimension:
        raise ValueError("dimension must be a positive integer")
    if target_dimension is not None:
        if (not np.isfinite(target_dimension) or int(target_dimension) != target_dimension
                or not 1 <= target_dimension <= dimension):
            raise ValueError("target_dimension must be an integer between 1 and dimension")
    if max_iterations < 1 or int(max_iterations) != max_iterations:
        raise ValueError("max_iterations must be a positive integer")
    if not np.isfinite(delta) or delta <= 0 or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("delta and epsilon must be finite and positive")
    if variant == "obj" and coordinate_minimizer is None:
        raise ValueError("FoBa-obj requires coordinate_minimizer")
    if variant == "gdt" and gradient is None:
        raise ValueError("FoBa-gdt requires gradient")

    def value(coefficients):
        result = float(objective(coefficients))
        if not np.isfinite(result):
            raise ValueError("objective returned a non-finite value")
        return result

    def refit(support):
        result = np.asarray(restricted_minimizer(tuple(support)), dtype=float)
        if result.shape != (dimension,) or not np.all(np.isfinite(result)):
            raise ValueError("restricted_minimizer must return a finite full length vector")
        outside = np.ones(dimension, dtype=bool)
        outside[support] = False
        if np.any(result[outside] != 0):
            raise ValueError("restricted_minimizer returned coefficients outside support")
        return result.copy()

    beta = np.zeros(dimension)
    support, gains, history = [], [], []
    current = value(beta)
    status = "max_iterations"
    for _ in range(max_iterations):
        candidates = [j for j in range(dimension) if j not in support]
        if variant == "gdt":
            grad = np.asarray(gradient(beta), dtype=float)
            if grad.shape != beta.shape or not np.all(np.isfinite(grad)):
                raise ValueError("gradient must return a finite full length vector")
            if target_dimension is None and np.max(np.abs(grad)) < epsilon:
                status = "threshold"
                break
        if not candidates:
            status = "full_support"
            break
        if variant == "obj":
            scores = []
            for j in candidates:
                trial = beta.copy()
                trial[j] += coordinate_minimizer(beta.copy(), j)
                scores.append(current - value(trial))
            position = int(np.argmax(scores))
            if target_dimension is None and scores[position] < delta:
                status = "threshold"
                break
        else:
            position = int(np.argmax(np.abs(grad[candidates])))
        added = candidates[position]
        support.append(added)
        beta = refit(support)
        updated = value(beta)
        gains.append(current - updated)
        current = updated
        history.append(dict(action="add", index=added, support=support.copy(),
                            objective=current, forward_gain=gains[-1]))

        # k is the current support size. Decrementing k restores delta^(k)
        # from the earlier forward step, as in Algorithm 1 (not a frozen gain).
        while support:
            costs = []
            for j in support:
                trial = beta.copy()
                trial[j] = 0.0
                costs.append(value(trial) - current)
            position = int(np.argmin(costs))
            threshold = gains[-1] / 2.0
            if costs[position] >= threshold:
                break
            removed = support.pop(position)
            gains.pop()
            beta = refit(support)
            current = value(beta)
            history.append(dict(action="remove", index=removed, support=support.copy(),
                                objective=current, deletion_cost=costs[position],
                                backward_threshold=threshold))

        if target_dimension is not None and len(support) == target_dimension:
            status = "target_dimension"
            break

    return dict(beta=beta, support=support.copy(), objective_value=current,
                history=history, status=status, converged=status != "max_iterations",
                forward_steps=sum(event["action"] == "add" for event in history))


def foba_least_squares(X, y, *, variant="obj", delta=1e-8, epsilon=1e-8,
                       max_iterations=1000, target_dimension=None):
    """FoBa for Q(beta) = ||X beta - y||^2 / 2 (no normalization/intercept).

    X has shape (samples, features), y has shape (samples,). Subproblems
    use exact coordinate updates and NumPy least-squares support refits.
    target_dimension selects target-size stopping instead of delta/epsilon;
    see foba for pruning and selected-coordinate counting semantics.
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if X.ndim != 2 or y.shape != (X.shape[0],):
        raise ValueError("expected X (samples, features) and y (samples,)")
    if X.shape[0] == 0 or not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must be finite with at least one sample")
    squared_norms = np.sum(X * X, axis=0)

    def objective(beta):
        residual = X @ beta - y
        return 0.5 * float(residual @ residual)

    def restricted_minimizer(support):
        beta = np.zeros(X.shape[1])
        if support:
            beta[list(support)] = np.linalg.lstsq(X[:, list(support)], y, rcond=None)[0]
        return beta

    def coordinate_minimizer(beta, j):
        if squared_norms[j] == 0:
            return 0.0
        return -float(X[:, j] @ (X @ beta - y)) / squared_norms[j]

    return foba(objective, restricted_minimizer, X.shape[1], variant=variant,
                coordinate_minimizer=coordinate_minimizer,
                gradient=lambda beta: X.T @ (X @ beta - y), delta=delta,
                epsilon=epsilon, max_iterations=max_iterations,
                target_dimension=target_dimension)
