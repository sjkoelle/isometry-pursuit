from itertools import combinations

import numpy as np
import cvxpy as cp
from sklearn.linear_model import MultiTaskLasso


def greedy(matrix, loss, target_dimension=None, selected_indices=[]):

    if target_dimension is None:
        dictionary_dimension, target_dimension = (
            matrix.shape
        )  # NOTE (Sam): this won't hold necessarily for regression in ambient space instead of tangent space.
    else:
        dictionary_dimension = matrix.shape[0]

    if target_dimension > 0:
        for d in range(dictionary_dimension):
            if d not in selected_indices:
                loss_ = loss(matrix[[selected_indices, d]])
        most_isometric_index = np.nanargmin(loss_)
        selected_indices.append(most_isometric_index)
        selected_indices = greedy(
            matrix, loss, target_dimension - d, ignore_elements=selected_indices
        )
        return selected_indices
    else:
        return None


def brute(matrix, target_dimension, loss):

    dictionary_dimension = matrix.shape[0]
    parametrizations = combinations(dictionary_dimension, target_dimension)
    losses = np.asarray([])
    for parametrization in parametrizations:
        losses.append(loss(matrix[parametrization]))
    return parametrization[losses.argmin()]


def group_basis_pursuit(
    matrix,
    eps=1e-12,
    threshold=1e-6,
):
    p, d = matrix.shape
    beta = cp.Variable((p, d))  # could initialize with lasso?
    objective = cp.Minimize(cp.sum(cp.norm(beta, axis=1)))
    constraints = [matrix @ beta == np.identity(d)]
    problem = cp.Problem(objective, constraints)
    scs_opts = {"eps": eps}
    output = problem.solve(solver=cp.SCS, **scs_opts)
    if output is np.inf:
        raise ValueError("No solution found")
    beta_optimized = beta.value
    beta_sparse = beta_optimized.copy()
    beta_sparse[np.abs(beta_sparse) < threshold] = 0
    # assert entire rows are 0 or raise warning
    return beta_sparse


def group_lasso(matrix, lambda_):

    p, d = matrix.shape
    y = np.identity(d)
    mtl = MultiTaskLasso(
        alpha=lambda_, tol=1e-16, max_iter=1000000, fit_intercept=False
    )
    mtl.fit(matrix, y)
    return mtl.coef_
