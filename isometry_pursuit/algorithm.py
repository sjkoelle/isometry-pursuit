from itertools import combinations

import numpy as np
import cvxpy as cp
from sklearn.linear_model import MultiTaskLasso
from tqdm import tqdm

import random


def greedy(
    matrix,
    loss,
    target_dimension=None,
    selected_indices=None,
    random_state=None,
    verbose=False,
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
                candidate_losses[p] = loss(
                    matrix[:, candidate_parametrization],
                    indices=candidate_parametrization,
                )
        # print(candidate_losses)
        minimum_loss = np.nanmin(candidate_losses)
        most_isometric_indices = np.where(candidate_losses == minimum_loss)[0]
        most_isometric_index = random.choice(most_isometric_indices)
        # most_isometric_index = np.nanargmin(candidate_losses)
        # print(most_isometric_index)
        selected_indices.append(most_isometric_index)
        selected_indices = greedy(
            matrix,
            loss,
            target_dimension,
            selected_indices=selected_indices,
            verbose=verbose,
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
        losses.append(loss(putative_X_S, indices=parametrization))

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


def create_convergence_log_with_coefficients(
    matrix, threshold=1e-6, iteration_points=None
):
    """
    Create convergence log with full coefficient matrices saved as structured numpy arrays.

    This version saves the actual coefficient matrices at each iteration point,
    making the data much more usable for analysis.

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
    convergence_data : dict
        Dictionary containing:
        - 'iterations': list of iteration counts
        - 'coefficient_norms': np.array of shape (n_iterations, n_features)
          containing ||beta[i,:]|| for each feature at each iteration
        - 'losses': np.array of total loss at each iteration
        - 'sparsities': np.array of sparsity count at each iteration
        - 'residuals': np.array of constraint residual at each iteration
        - 'coefficient_matrices': list of full beta matrices at each iteration
    """
    if iteration_points is None:
        iteration_points = [1, 10, 100, 1000, 10000]

    D, P = matrix.shape
    n_iterations = len(iteration_points)

    # Initialize arrays
    coefficient_norms = np.zeros((n_iterations, P))
    losses = np.zeros(n_iterations)
    sparsities = np.zeros(n_iterations, dtype=int)
    residuals = np.zeros(n_iterations)
    coefficient_matrices = []

    valid_iterations = []

    print(
        f"Creating convergence log with coefficients at {len(iteration_points)} iteration points: {iteration_points}"
    )

    for idx, iter_count in enumerate(iteration_points):
        try:
            # Call group_basis_pursuit with specific iteration limit
            # Use threshold instead of eps when steps is specified
            beta, _ = group_basis_pursuit(
                matrix, threshold=threshold, steps=iter_count, eps=None
            )

            if beta is not None:
                # Calculate metrics
                current_loss = np.sum(np.linalg.norm(beta, axis=1))
                sparsity = np.sum(np.linalg.norm(beta, axis=1) > threshold)
                residual = np.linalg.norm(matrix @ beta - np.identity(D))

                # Store coefficient norms for each feature
                coefficient_norms[idx, :] = np.linalg.norm(beta, axis=1)
                losses[idx] = current_loss
                sparsities[idx] = sparsity
                residuals[idx] = residual
                coefficient_matrices.append(beta.copy())
                valid_iterations.append(iter_count)

                print(
                    f"  Iter {iter_count}: loss={current_loss:.4f}, sparsity={sparsity}, residual={residual:.2e}"
                )

        except Exception as e:
            print(f"  Iter {iter_count}: Failed - {e}")
            # Fill with NaN for failed iterations
            coefficient_norms[idx, :] = np.nan
            losses[idx] = np.nan
            sparsities[idx] = -1  # Use -1 instead of NaN for integer array
            residuals[idx] = np.nan
            coefficient_matrices.append(None)
            continue

    convergence_data = {
        "iterations": valid_iterations,
        "coefficient_norms": coefficient_norms,
        "losses": losses,
        "sparsities": sparsities,
        "residuals": residuals,
        "coefficient_matrices": coefficient_matrices,
        "matrix_shape": (D, P),
        "iteration_points": iteration_points,
    }

    return convergence_data


def save_convergence_data(convergence_data, filepath):
    """
    Save convergence data in a structured numpy format (.npz file).

    This creates a much more usable file format than pickle, with:
    - coefficient_norms: (n_iterations, n_features) array of ||beta[i,:]||
    - losses: (n_iterations,) array of total losses
    - sparsities: (n_iterations,) array of sparsity counts
    - residuals: (n_iterations,) array of constraint residuals
    - iterations: (n_iterations,) array of iteration counts
    - matrix_shape: (2,) array of (D, P)

    Parameters:
    -----------
    convergence_data : dict
        Output from create_convergence_log_with_coefficients()
    filepath : str
        Path to save the .npz file
    """
    np.savez(
        filepath,
        coefficient_norms=convergence_data["coefficient_norms"],
        losses=convergence_data["losses"],
        sparsities=convergence_data["sparsities"],
        residuals=convergence_data["residuals"],
        iterations=np.array(convergence_data["iterations"]),
        matrix_shape=np.array(convergence_data["matrix_shape"]),
        iteration_points=np.array(convergence_data["iteration_points"]),
    )
    print(f"Convergence data saved to {filepath}")


def load_convergence_data(filepath):
    """
    Load convergence data from .npz file.

    Parameters:
    -----------
    filepath : str
        Path to the .npz file

    Returns:
    --------
    convergence_data : dict
        Dictionary with the same structure as create_convergence_log_with_coefficients()
    """
    data = np.load(filepath)
    return {
        "coefficient_norms": data["coefficient_norms"],
        "losses": data["losses"],
        "sparsities": data["sparsities"],
        "residuals": data["residuals"],
        "iterations": data["iterations"].tolist(),
        "matrix_shape": tuple(data["matrix_shape"]),
        "iteration_points": data["iteration_points"].tolist(),
    }


def group_lasso(matrix, lambda_):

    D, P = matrix.shape
    y = np.identity(D)
    mtl = MultiTaskLasso(
        alpha=lambda_, tol=1e-16, max_iter=1000000, fit_intercept=False
    )
    mtl.fit(matrix, y)
    return mtl.coef_
