from itertools import combinations

import numpy as np
import cvxpy as cp
from sklearn.linear_model import MultiTaskLasso
from tqdm import tqdm


def greedy(matrix, loss, target_dimension=None, selected_indices=[]):
    """Matrix is d \times p dimensional"""
    if target_dimension is None:
        target_dimension, dictionary_dimension = (
            matrix.shape
        )  # NOTE (Sam): this won't hold necessarily for regression in ambient space instead of tangent space.
    else:
        dictionary_dimension = matrix.shape[1]

    while len(selected_indices) < target_dimension:
        # if target_dimension > 0:
        candidate_losses = np.repeat(np.nan, dictionary_dimension)

        for p in range(dictionary_dimension):
            if p not in selected_indices:
                candidate_parametrization = np.concatenate(
                    [selected_indices, [p]]
                ).astype(int)
                candidate_losses[p] = loss(matrix[:, candidate_parametrization])
        most_isometric_index = np.nanargmin(candidate_losses)
        selected_indices.append(most_isometric_index)
        selected_indices = greedy(
            matrix, loss, target_dimension, selected_indices=selected_indices
        )

    return selected_indices


# else:
#     return None


def brute(matrix, loss, target_dimension):

    dictionary_dimension = matrix.shape[1]
    print(
        f"Computing brute force solution for dictionary dimension {dictionary_dimension} and target_dimension {target_dimension}"
    )
    parametrizations = combinations(range(dictionary_dimension), target_dimension)

    losses = []
    for parametrization in tqdm(parametrizations):
        putative_X_S = matrix[:, parametrization]
        losses.append(loss(putative_X_S))

    selected_indices = np.asarray(losses).argmin()
    parametrizations = combinations(range(dictionary_dimension), target_dimension)
    return list(parametrizations)[selected_indices]


def group_basis_pursuit(
    matrix,
    eps=1e-12,
    threshold=1e-6,
):
    D, P = matrix.shape
    beta = cp.Variable((P, D))  # could initialize with lasso?
    objective = cp.Minimize(cp.sum(cp.norm(beta, axis=1)))
    constraints = [matrix @ beta == np.identity(D)]
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

    D, P = matrix.shape
    y = np.identity(D)
    mtl = MultiTaskLasso(
        alpha=lambda_, tol=1e-16, max_iter=1000000, fit_intercept=False
    )
    mtl.fit(matrix, y)
    return mtl.coef_
