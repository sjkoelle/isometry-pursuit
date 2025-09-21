from itertools import combinations

import numpy as np
import cvxpy as cp
from sklearn.linear_model import MultiTaskLasso
from tqdm import tqdm

import random


def greedy(
    matrix, loss, target_dimension=None, selected_indices=None, random_state=None, verbose=False
):
    """Matrix is d \times p dimensional"""
    if selected_indices is None:
        selected_indices = []
    if random_state is not None:
        random.seed(random_state)
    if target_dimension is None:
        target_dimension, dictionary_dimension = (
            matrix.shape
        )  # NOTE (Sam): this won't hold necessarily for regression in ambient space instead of tangent space.
    else:
        dictionary_dimension = matrix.shape[1]

    while len(selected_indices) < target_dimension:
        # if target_dimension > 0:
        candidate_losses = np.repeat(np.nan, dictionary_dimension)
        candidate_indices = list(range(dictionary_dimension))
        random.shuffle(candidate_indices)
        # print(candidate_indices)
        for p in candidate_indices:
            if p not in selected_indices:
                candidate_parametrization = np.concatenate(
                    [selected_indices, [p]]
                ).astype(int)
                candidate_losses[p] = loss(matrix[:, candidate_parametrization])
        # print(candidate_losses)
        minimum_loss = np.nanmin(candidate_losses)
        most_isometric_indices = np.where(candidate_losses == minimum_loss)[0]
        most_isometric_index = random.choice(most_isometric_indices)
        # most_isometric_index = np.nanargmin(candidate_losses)
        # print(most_isometric_index)
        selected_indices.append(most_isometric_index)
        selected_indices = greedy(
            matrix, loss, target_dimension, selected_indices=selected_indices, verbose=verbose
        )
    
    # Only print the final result when we have completed the greedy search and verbose is True
    if len(selected_indices) == target_dimension and verbose:
        print("greedy", selected_indices)
    
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
    steps=None,
):
    assert not (steps and eps)

    if steps is None:
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

        # Get iteration count from solver info
        iterations = (
            problem.solver_stats.num_iters
            if hasattr(problem.solver_stats, "num_iters")
            else None
        )

    # assert entire rows are 0 or raise warning
    if steps is not None:
        D, P = matrix.shape
        beta = cp.Variable((P, D))
        objective = cp.Minimize(cp.sum(cp.norm(beta, axis=1)))
        constraints = [matrix @ beta == np.identity(D)]
        problem = cp.Problem(objective, constraints)
        scs_opts = {"max_iters": steps}
        output = problem.solve(solver=cp.SCS, **scs_opts)
        if output is np.inf:
            raise ValueError("No solution found")
        beta_optimized = beta.value
        beta_sparse = beta_optimized.copy()
        beta_sparse[np.abs(beta_sparse) < threshold] = 0
        # Get iteration count from solver info
        iterations = (
            problem.solver_stats.num_iters
            if hasattr(problem.solver_stats, "num_iters")
            else None
        )

    return beta_sparse, iterations


def create_convergence_log(matrix, threshold=1e-6, iteration_points=None):
    """
    Create convergence log by calling group_basis_pursuit multiple times with different iteration limits.

    Parameters:
    -----------
    matrix : np.ndarray
        The design matrix
    threshold : float
        Threshold for sparsity calculation
    iteration_points : list, optional
        List of iteration counts to test. Default: [1, 10, 100, 1000, 10000]

    Returns:
    --------
    convergence_log : dict
        Dictionary with 'iterations', 'losses', 'sparsities', 'residuals' lists
    """
    if iteration_points is None:
        iteration_points = [1, 10, 100, 1000, 10000]

    D, P = matrix.shape
    convergence_log = {
        "iterations": [],
        "losses": [],
        "sparsities": [],
        "residuals": [],
    }

    print(
        f"Creating convergence log at {len(iteration_points)} iteration points: {iteration_points}"
    )

    for iter_count in iteration_points:
        try:
            # Call group_basis_pursuit with specific iteration limit
            beta, _ = group_basis_pursuit(matrix, threshold=threshold, steps=iter_count)

            if beta is not None:
                # Calculate metrics
                current_loss = np.sum(np.linalg.norm(beta, axis=1))
                sparsity = np.sum(np.linalg.norm(beta, axis=1) > threshold)
                residual = np.linalg.norm(matrix @ beta - np.identity(D))

                convergence_log["iterations"].append(iter_count)
                convergence_log["losses"].append(current_loss)
                convergence_log["sparsities"].append(sparsity)
                convergence_log["residuals"].append(residual)

                print(
                    f"  Iter {iter_count}: loss={current_loss:.4f}, sparsity={sparsity}, residual={residual:.2e}"
                )

        except Exception as e:
            print(f"  Iter {iter_count}: Failed - {e}")
            continue

    return convergence_log


def group_lasso(matrix, lambda_):

    D, P = matrix.shape
    y = np.identity(D)
    mtl = MultiTaskLasso(
        alpha=lambda_, tol=1e-16, max_iter=1000000, fit_intercept=False
    )
    mtl.fit(matrix, y)
    return mtl.coef_
