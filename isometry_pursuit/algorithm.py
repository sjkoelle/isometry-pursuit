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


def backward_elimination(
    matrix,
    loss,
    target_dimension=None,
    selected_indices=None,
    random_state=None,
    verbose=False,
):
    """Greedily remove columns until ``target_dimension`` columns remain."""
    if target_dimension is None:
        target_dimension = matrix.shape[0]
    if selected_indices is None:
        selected_indices = list(range(matrix.shape[1]))
    else:
        selected_indices = [int(i) for i in selected_indices]
    if random_state is not None:
        random.seed(random_state)

    if len(selected_indices) < target_dimension:
        raise ValueError("selected_indices must contain at least target_dimension columns")

    while len(selected_indices) > target_dimension:
        candidate_losses = []
        candidate_subsets = []
        removal_order = list(range(len(selected_indices)))
        random.shuffle(removal_order)

        for position in removal_order:
            candidate_subset = selected_indices[:position] + selected_indices[position + 1 :]
            candidate_losses.append(
                loss(
                    matrix[:, candidate_subset],
                    indices=np.asarray(candidate_subset, dtype=int),
                )
            )
            candidate_subsets.append(candidate_subset)

        minimum_loss = np.nanmin(candidate_losses)
        best_positions = np.where(np.asarray(candidate_losses) == minimum_loss)[0]
        best_position = random.choice(best_positions)
        previous_indices = selected_indices
        selected_indices = candidate_subsets[best_position]

        if verbose:
            removed_indices = sorted(set(previous_indices) - set(selected_indices))
            print(
                "backward_elimination",
                f"{len(previous_indices)} -> {len(selected_indices)}",
                f"removed={removed_indices}",
                f"loss={minimum_loss}",
            )

    if verbose:
        print("backward_elimination", selected_indices)

    return selected_indices


def greedy_swap(
    matrix, loss, target_dimension=None, initialization="forward",
    random_state=None, max_rounds=None, improvement_tol=0.0, verbose=False,
):
    """Forward or backward greedy initialization followed by best one-swap search.

    Pass an IP-restricted matrix to keep both stages inside the IP dictionary.
    Returned indices are local to the matrix passed in. Backward initialization
    starts with all columns; forward initialization starts empty.
    """
    if initialization not in ("forward", "backward"):
        raise ValueError("initialization must be 'forward' or 'backward'")
    if target_dimension is None:
        target_dimension = matrix.shape[0]
    if not 1 <= target_dimension <= matrix.shape[1] or int(target_dimension) != target_dimension:
        raise ValueError("target_dimension must be between 1 and the number of columns")
    return forward_backward(
        matrix, loss, target_dimension=target_dimension,
        selected_indices=(list(range(matrix.shape[1])) if initialization == "backward" else []),
        random_state=random_state, max_rounds=max_rounds,
        improvement_tol=improvement_tol, verbose=verbose,
    )


def forward_backward(
    matrix,
    loss,
    target_dimension=None,
    selected_indices=None,
    random_state=None,
    max_rounds=None,
    improvement_tol=0.0,
    verbose=False,
):
    """
    Fixed-cardinality forward-backward selection.

    First build a target-size support by forward greedy selection. Then repeatedly
    try one add/delete move and accept it if the target-size loss improves.
    """
    if random_state is not None:
        random.seed(random_state)
    if target_dimension is None:
        target_dimension = matrix.shape[0]
    dictionary_dimension = matrix.shape[1]

    if selected_indices is None:
        selected_indices = []
    selected_indices = [int(i) for i in selected_indices]

    if len(selected_indices) > target_dimension:
        selected_indices = backward_elimination(
            matrix,
            loss,
            target_dimension=target_dimension,
            selected_indices=selected_indices,
            random_state=random_state,
            verbose=verbose,
        )
    elif len(selected_indices) < target_dimension:
        selected_indices = greedy(
            matrix,
            loss,
            target_dimension=target_dimension,
            selected_indices=selected_indices,
            verbose=verbose,
        )

    current_loss = loss(
        matrix[:, selected_indices],
        indices=np.asarray(selected_indices, dtype=int),
    )
    if max_rounds is None:
        max_rounds = dictionary_dimension

    for round_index in range(max_rounds):
        best_loss = current_loss
        best_indices = selected_indices
        candidate_additions = [
            index for index in range(dictionary_dimension) if index not in selected_indices
        ]
        random.shuffle(candidate_additions)

        for added_index in candidate_additions:
            expanded_indices = selected_indices + [added_index]
            removal_order = list(range(len(expanded_indices)))
            random.shuffle(removal_order)

            for removal_position in removal_order:
                candidate_indices = (
                    expanded_indices[:removal_position]
                    + expanded_indices[removal_position + 1 :]
                )
                if set(candidate_indices) == set(selected_indices):
                    continue
                candidate_loss = loss(
                    matrix[:, candidate_indices],
                    indices=np.asarray(candidate_indices, dtype=int),
                )
                if candidate_loss < best_loss - improvement_tol:
                    best_loss = candidate_loss
                    best_indices = candidate_indices

        if best_indices == selected_indices:
            break

        if verbose:
            removed_indices = sorted(set(selected_indices) - set(best_indices))
            added_indices = sorted(set(best_indices) - set(selected_indices))
            print(
                "forward_backward",
                f"round={round_index + 1}",
                f"added={added_indices}",
                f"removed={removed_indices}",
                f"loss={best_loss}",
            )

        selected_indices = best_indices
        current_loss = best_loss

    if verbose:
        print("forward_backward", selected_indices)

    return selected_indices


from .transformation import exponential_transformation

THRESHOLD = 1e-4


def line_search_for_cardinality(
    X,
    target_cardinality,
    power=1.0,
    max_iterations=100,
    initial_violation=None,
):
    """
    Perform line search to find constraint violation that achieves target support cardinality.

    Parameters:
    -----------
    X : numpy.ndarray
        Data matrix for the experiment
    target_cardinality : int
        Target number of features to select
    power : float
        Power parameter for transformation
    max_iterations : int
        Maximum number of line search iterations
    initial_bounds : tuple
        Initial bounds for constraint violation search
    Returns:
    --------
    best_violation : float
        Best constraint violation found
    best_cardinality : int
        Actual cardinality achieved
    best_result : dict
        Results from the best experiment
    """
    # Binary search bounds
    best_violation = None
    best_cardinality = None
    best_result = None
    constraint_violation = (
        initial_violation if initial_violation is not None else X.shape[0]
    )
    lower_bound = 0.0
    upper_bound = None
    for iteration in range(max_iterations):
        # Test midpoint

        # Run isometry pursuit with current constraint violation
        try:
            result = isometry_pursuit(
                X,
                power=power,
                constraint_violation=constraint_violation,
            )  # Calculate actual cardinality
            actual_cardinality = len(result["basis_pursuit_indices"])
            cardinality_violation = actual_cardinality - target_cardinality
        except Exception as e:
            cardinality_violation = np.inf
            actual_cardinality = None
        print(
            f"Iteration {iteration + 1}, Cardinality {actual_cardinality}, Cardinality violation {cardinality_violation}, Constraint violation {constraint_violation}"
        )
        # Check if we've found exact target cardinality

        if cardinality_violation == 0:
            print(result["basis_pursuit_indices"])
            return actual_cardinality, result

        # Update search bounds based on cardinality direction
        if cardinality_violation > 0:
            lower_bound = constraint_violation
            if upper_bound is None:
                constraint_violation = constraint_violation * 2
            else:
                constraint_violation = (constraint_violation + upper_bound) / 2
        else:
            upper_bound = constraint_violation
            constraint_violation = (constraint_violation + lower_bound) / 2

    raise ValueError("No solution found")


def isometry_pursuit(
    X,
    power=1.0,
    limit=1e6,
    max_steps=None,
    log_convergence=False,
    constraint_violation=0.0,
    iterations=None,
    eps=None,
):
    D, P = X.shape
    data_transformed = exponential_transformation(X, power=power)
    # Use CVXPY implementation
    beta, basis_pursuit_iterations, objective_value = group_basis_pursuit(
        data_transformed, constraint_violation=constraint_violation, steps=iterations, **({'eps': eps} if eps is not None else {})
    )
    print('obv',objective_value)
    # Note: Convergence logging is not implemented for the main experiment flow
    # since it would require calling group_basis_pursuit multiple times with different parameters
    # For convergence analysis, use the standalone convergence logging functions
    basis_pursuit_indices = np.where(np.linalg.norm(beta, axis=1) > THRESHOLD)[0]

    return {
        "beta": beta,
        "basis_pursuit_indices": basis_pursuit_indices,
        "basis_pursuit_iterations": basis_pursuit_iterations,
        "objective_value": objective_value,
    }


def isometry_pursuit_leave_one_out(X, power=1.0, constraint_violation=0.0):
    """
    Perform greedy leave-one-out analysis on isometry pursuit results.

    Parameters:
    -----------
    X : numpy.ndarray
        Data matrix of shape (D, P)
    power : float
        Power parameter for transformation (default: 1.0)
    constraint_violation : float
        Constraint violation tolerance (default: 0.0)

    Returns:
    --------
    dict
        Dictionary containing:
        - "losses": list of objective values at each step
        - "selected_indices": final subset of features after greedy removal
    """
    D, P = X.shape  # Extract dimensions from input matrix

    # Run initial isometry pursuit on full matrix
    initial_result = isometry_pursuit(
        X, power=power, constraint_violation=constraint_violation
    )

    # Handle edge case: if no features selected initially
    if len(initial_result["basis_pursuit_indices"]) == 0:
        return {
            "losses": [initial_result["objective_value"]],
            "selected_indices": np.array([]),
        }

    # Initialize tracking
    current_indices = initial_result["basis_pursuit_indices"].copy()
    losses = [initial_result["objective_value"]]

    results = {}
    # Greedy removal loop - continue until len(current_indices) = D
    while len(current_indices) > D:
        print(current_indices)
        trial_losses = []
        trial_features = []

        for i, feature in enumerate(current_indices):
            print(f"Removing feature {feature}")

            # Create subset by removing current feature
            trial_indices = np.delete(current_indices, i)
            # Subset X to trial indices

            X_trial = X[:, trial_indices]
            trial_result = isometry_pursuit(
                X_trial, power=power, constraint_violation=constraint_violation
            )
            # Record the objective value for this trial
            print(trial_result["objective_value"])
            print(len(trial_result["basis_pursuit_indices"]))

            results[frozenset(trial_result["basis_pursuit_indices"])] = trial_result[
                "objective_value"
            ]
            trial_losses.append(trial_result["objective_value"])
            trial_features.append(feature)

        # # Check if any trials succeeded
        # if len(trial_losses) == 0:
        #     # No valid trials, stop greedy removal
        #     break

        # Find the feature whose removal gives the lowest loss
        best_trial_idx = np.argmin(trial_losses)
        best_loss = trial_losses[best_trial_idx]
        best_feature = trial_features[best_trial_idx]

        # Remove the feature and continue
        current_indices = np.delete(
            current_indices, np.where(current_indices == best_feature)[0][0]
        )
        losses.append(best_loss)

    return {"losses": losses, "selected_indices": current_indices}


def brute(matrix, loss, target_dimension):

    dictionary_dimension = matrix.shape[1]
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
    constraint_violation=0.0,
):
    if steps is not None:
        eps = None
    assert not (steps and eps)

    D, P = matrix.shape
    beta = cp.Variable((P, D))  # could initialize with lasso?
    objective = cp.Minimize(cp.sum(cp.norm(beta, axis=1)))
    if constraint_violation == 0.0:
        constraints = [matrix @ beta == np.identity(D)]
    else:
        constraints = [
            # cp.norm(matrix @ beta - np.identity(D), "fro")
            cp.sum_squares(matrix @ beta - np.identity(D))
            <= constraint_violation
        ]
    problem = cp.Problem(objective, constraints)

    # Set SCS options based on whether steps is specified
    if steps is None:
        scs_opts = {"eps_abs": eps, "eps_rel": eps}
    else:
        scs_opts = {"max_iters": steps}
    scs_opts["acceleration_lookback"] = 0
    scs_opts["adaptive_scale"] = False

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

    # Calculate objective function value (sum of L2 norms of rows)
    objective_value = np.sum(np.linalg.norm(beta_sparse, axis=1))

    return beta_sparse, iterations, objective_value


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
            beta, _, _ = group_basis_pursuit(
                matrix, threshold=threshold, steps=iter_count, constraint_violation=0.0
            )

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
            beta, _, _ = group_basis_pursuit(
                matrix,
                threshold=threshold,
                steps=iter_count,
                eps=None,
                constraint_violation=0.0,
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
