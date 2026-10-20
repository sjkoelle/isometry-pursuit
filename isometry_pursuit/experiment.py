import time

import numpy as np

from .algorithm import (
    isometry_pursuit,
    greedy,
    forward_backward,
    backward_elimination,
    group_basis_pursuit,
)
from .loss import group_lasso_norm
from .transformation import exponential_transformation


PINV_RCOND = 0.0
RANK_TOL = 0.0
RECONSTRUCTION_TOL = 1e-6


def _gl_loss(sub_matrix, indices=None):
    ambient_dimension, num_columns = sub_matrix.shape
    singular_values = np.linalg.svd(sub_matrix, compute_uv=False)
    rank = int(np.sum(singular_values > RANK_TOL))

    if num_columns <= ambient_dimension and rank < num_columns:
        return np.inf
    if num_columns >= ambient_dimension and rank < ambient_dimension:
        return np.inf

    beta = np.linalg.pinv(sub_matrix, rcond=PINV_RCOND)
    if num_columns >= ambient_dimension:
        residual = np.linalg.norm(sub_matrix @ beta - np.eye(ambient_dimension), ord="fro")
        if residual > RECONSTRUCTION_TOL:
            return np.inf
    return group_lasso_norm(beta)


def _bp_objective_loss(
    sub_matrix,
    indices=None,
    iterations=None,
    eps=None,
    reconstruction_tol=RECONSTRUCTION_TOL,
):
    ambient_dimension, num_columns = sub_matrix.shape
    singular_values = np.linalg.svd(sub_matrix, compute_uv=False)
    rank = int(np.sum(singular_values > RANK_TOL))

    if num_columns < ambient_dimension or rank < ambient_dimension:
        return np.inf

    try:
        beta, _, objective_value = group_basis_pursuit(
            sub_matrix,
            steps=iterations,
            **({"eps": eps} if eps is not None else {}),
        )
    except Exception:
        return np.inf

    residual = np.linalg.norm(sub_matrix @ beta - np.eye(ambient_dimension), ord="fro")
    if residual > reconstruction_tol:
        return np.inf
    return float(objective_value)


def run_tensor_experiment(
    data_tensor,
    constraint_violation=0.0,
    iterations=None,
    eps=None,
    run_restricted_bp_backward=False,
    restricted_bp_backward_iterations=250,
    restricted_bp_backward_eps=None,
    restricted_bp_backward_reconstruction_tol=1e-3,
    restricted_bp_backward_max_replicates=None,
    verbose_restricted_bp_backward=False,
):
    """
    Run isometry pursuit on each replicate in a data tensor.

    Parameters:
    -----------
    data_tensor : numpy.ndarray
        Shape (D, P, R) — D features, P samples per replicate, R replicates
    constraint_violation : float
        Constraint violation tolerance for group basis pursuit

    Returns:
    --------
    replicates : list of dict
        Per-replicate results including basis_pursuit_indices
    coefficient_matrices : list of np.ndarray
    """
    R = data_tensor.shape[2]

    replicates = []
    coefficient_matrices = []

    for i in range(R):
        X = data_tensor[:, :, i]
        start_time = time.perf_counter()
        result = isometry_pursuit(X, constraint_violation=constraint_violation, iterations=iterations, eps=eps)
        basis_pursuit_time = time.perf_counter() - start_time
        indices = result["basis_pursuit_indices"]

        exp_X = exponential_transformation(X)

        # Greedy on full exp_X using group lasso norm
        start_time = time.perf_counter()
        greedy_gl_indices = [int(j) for j in greedy(exp_X, _gl_loss)]
        greedy_gl_time = time.perf_counter() - start_time
        greedy_gl_norm = float(_gl_loss(exp_X[:, greedy_gl_indices]))

        start_time = time.perf_counter()
        forward_backward_gl_indices = [
            int(j) for j in forward_backward(exp_X, _gl_loss, selected_indices=greedy_gl_indices)
        ]
        forward_backward_gl_time = greedy_gl_time + time.perf_counter() - start_time
        forward_backward_gl_norm = float(
            _gl_loss(exp_X[:, forward_backward_gl_indices])
        )

        start_time = time.perf_counter()
        backward_gl_indices = backward_elimination(exp_X, _gl_loss)
        backward_gl_norm = float(_gl_loss(exp_X[:, backward_gl_indices]))
        backward_swap_gl_indices = [int(j) for j in forward_backward(
            exp_X, _gl_loss, selected_indices=backward_gl_indices,
        )]
        backward_swap_gl_time = time.perf_counter() - start_time
        backward_swap_gl_norm = float(_gl_loss(exp_X[:, backward_swap_gl_indices]))

        # Greedy on BP-selected subset of exp_X using group lasso norm
        if len(indices) >= X.shape[0]:
            exp_X_bp = exp_X[:, indices]
            start_time = time.perf_counter()
            local_indices = greedy(exp_X_bp, _gl_loss)
            greedy_gl_bp_time = time.perf_counter() - start_time
            greedy_gl_bp_indices = [int(indices[j]) for j in local_indices]
            greedy_gl_bp_norm = float(_gl_loss(exp_X[:, greedy_gl_bp_indices]))

            start_time = time.perf_counter()
            local_backward_indices = backward_elimination(exp_X_bp, _gl_loss)
            backward_gl_bp_time = time.perf_counter() - start_time
            backward_gl_bp_indices = [int(indices[j]) for j in local_backward_indices]
            backward_gl_bp_norm = float(_gl_loss(exp_X[:, backward_gl_bp_indices]))

            start_time = time.perf_counter()
            local_backward_swap_indices = forward_backward(
                exp_X_bp, _gl_loss, selected_indices=local_backward_indices,
            )
            backward_swap_gl_bp_time = backward_gl_bp_time + time.perf_counter() - start_time
            backward_swap_gl_bp_indices = [int(indices[j]) for j in local_backward_swap_indices]
            backward_swap_gl_bp_norm = float(_gl_loss(exp_X[:, backward_swap_gl_bp_indices]))

            start_time = time.perf_counter()
            local_forward_backward_indices = forward_backward(
                exp_X_bp, _gl_loss, selected_indices=local_indices,
            )
            forward_backward_gl_bp_time = greedy_gl_bp_time + time.perf_counter() - start_time
            forward_backward_gl_bp_indices = [
                int(indices[j]) for j in local_forward_backward_indices
            ]
            forward_backward_gl_bp_norm = float(
                _gl_loss(exp_X[:, forward_backward_gl_bp_indices])
            )

            should_run_restricted_bp_backward = (
                run_restricted_bp_backward
                and (
                    restricted_bp_backward_max_replicates is None
                    or i < restricted_bp_backward_max_replicates
                )
            )

            if should_run_restricted_bp_backward:
                if verbose_restricted_bp_backward:
                    print(
                        "restricted BP backward",
                        f"replicate={i}",
                        f"support={len(indices)}",
                        f"target={X.shape[0]}",
                    )

                def restricted_bp_loss(sub_matrix, indices=None):
                    return _bp_objective_loss(
                        sub_matrix,
                        indices=indices,
                        iterations=restricted_bp_backward_iterations,
                        eps=restricted_bp_backward_eps,
                        reconstruction_tol=restricted_bp_backward_reconstruction_tol,
                    )

                start_time = time.perf_counter()
                local_backward_bp_indices = backward_elimination(
                    exp_X_bp,
                    restricted_bp_loss,
                    verbose=verbose_restricted_bp_backward,
                )
                backward_bp_time = time.perf_counter() - start_time
                backward_bp_gl_indices = [
                    int(indices[j]) for j in local_backward_bp_indices
                ]
                backward_bp_gl_norm = float(_gl_loss(exp_X[:, backward_bp_gl_indices]))
            else:
                backward_bp_gl_indices = None
                backward_bp_gl_norm = None
                backward_bp_time = None
        else:
            greedy_gl_bp_indices = None
            greedy_gl_bp_norm = None
            greedy_gl_bp_time = None
            backward_gl_bp_indices = None
            backward_gl_bp_norm = None
            backward_gl_bp_time = None
            backward_swap_gl_bp_indices = None
            backward_swap_gl_bp_norm = None
            backward_swap_gl_bp_time = None
            forward_backward_gl_bp_indices = None
            forward_backward_gl_bp_norm = None
            forward_backward_gl_bp_time = None
            backward_bp_gl_indices = None
            backward_bp_gl_norm = None
            backward_bp_time = None

        # Compute group lasso norm of pseudoinverse for convex solution
        bp_gl_norm = float(_gl_loss(exp_X[:, indices]))

        replicates.append({
            "replicate": i,
            "support_cardinality": len(indices),
            "group_lasso_norm": bp_gl_norm,
            "basis_pursuit_iterations": result["basis_pursuit_iterations"],
            "basis_pursuit_time_seconds": basis_pursuit_time,
            "basis_pursuit_indices": indices.tolist(),
            "greedy_gl_indices": greedy_gl_indices,
            "greedy_gl_norm": greedy_gl_norm,
            "greedy_gl_time_seconds": greedy_gl_time,
            "forward_backward_gl_indices": forward_backward_gl_indices,
            "forward_backward_gl_norm": forward_backward_gl_norm,
            "forward_backward_gl_time_seconds": forward_backward_gl_time,
            "backward_gl_indices": [int(j) for j in backward_gl_indices],
            "backward_gl_norm": backward_gl_norm,
            "backward_swap_gl_indices": backward_swap_gl_indices,
            "backward_swap_gl_norm": backward_swap_gl_norm,
            "backward_swap_gl_time_seconds": backward_swap_gl_time,
            "greedy_gl_bp_indices": greedy_gl_bp_indices,
            "greedy_gl_bp_norm": greedy_gl_bp_norm,
            "greedy_gl_bp_time_seconds": greedy_gl_bp_time,
            "backward_gl_bp_indices": backward_gl_bp_indices,
            "backward_gl_bp_norm": backward_gl_bp_norm,
            "backward_gl_bp_time_seconds": backward_gl_bp_time,
            "backward_swap_gl_bp_indices": backward_swap_gl_bp_indices,
            "backward_swap_gl_bp_norm": backward_swap_gl_bp_norm,
            "backward_swap_gl_bp_time_seconds": backward_swap_gl_bp_time,
            "forward_backward_gl_bp_indices": forward_backward_gl_bp_indices,
            "forward_backward_gl_bp_norm": forward_backward_gl_bp_norm,
            "forward_backward_gl_bp_time_seconds": forward_backward_gl_bp_time,
            "backward_bp_gl_indices": backward_bp_gl_indices,
            "backward_bp_gl_norm": backward_bp_gl_norm,
            "backward_bp_time_seconds": backward_bp_time,
        })
        coefficient_matrices.append(result["beta"])

    return replicates, coefficient_matrices


def print_experiment_summary(results):
    """
    Print summary statistics comparing convex (basis pursuit) vs greedy approach.

    Parameters:
    -----------
    results : list of dict
        List of replicate results from run_tensor_experiment
    """
    import json

    # Handle both list of dicts and loaded JSON with 'replicates' key
    if isinstance(results, dict) and 'replicates' in results:
        replicates = results['replicates']
    else:
        replicates = results

    convex_norms = []
    greedy_norms = []
    convex_cards = []
    greedy_cards = []

    for rep in replicates:
        convex_norms.append(rep['group_lasso_norm'])
        greedy_norms.append(rep['greedy_gl_norm'])
        convex_cards.append(rep['support_cardinality'])
        greedy_cards.append(len(rep['greedy_gl_indices']))

    convex_norms = np.array(convex_norms)
    greedy_norms = np.array(greedy_norms)
    convex_cards = np.array(convex_cards)
    greedy_cards = np.array(greedy_cards)

    # Compute wins
    greedy_wins = np.sum(greedy_norms < convex_norms)
    convex_wins = np.sum(convex_norms < greedy_norms)
    ties = np.sum(convex_norms == greedy_norms)

    print("\n" + "="*70)
    print("CONVEX vs GREEDY COMPARISON")
    print("="*70)
    print(f"\nGroup Lasso Norm (lower is better):")
    print(f"  Convex:  mean={convex_norms.mean():.4f}, std={convex_norms.std():.4f}, "
          f"min={convex_norms.min():.4f}, max={convex_norms.max():.4f}")
    print(f"  Greedy:  mean={greedy_norms.mean():.4f}, std={greedy_norms.std():.4f}, "
          f"min={greedy_norms.min():.4f}, max={greedy_norms.max():.4f}")
    print(f"\nSupport Cardinality:")
    print(f"  Convex:  mean={convex_cards.mean():.2f}, std={convex_cards.std():.2f}")
    print(f"  Greedy:  mean={greedy_cards.mean():.2f}, std={greedy_cards.std():.2f}")
    print(f"\nHead-to-head (out of {len(replicates)} replicates):")
    print(f"  Greedy wins: {greedy_wins}")
    print(f"  Convex wins: {convex_wins}")
    print(f"  Ties:        {ties}")
    print("="*70 + "\n")


def compare_swap_initializations(matrix, ip_indices, target_dimension=None, random_state=0):
    """Compare four greedy-initialized exchange searches on the same dictionary.

    ``matrix`` is already transformed, as in run_tensor_experiment. IP methods
    restrict both initialization and exchanges to ip_indices. Timings include
    initialization and swaps, but exclude the externally computed IP support.
    """
    if target_dimension is None:
        target_dimension = matrix.shape[0]
    ip_indices = [int(j) for j in ip_indices]
    if len(set(ip_indices)) != len(ip_indices) or any(
        j < 0 or j >= matrix.shape[1] for j in ip_indices
    ):
        raise ValueError("ip_indices must be distinct valid column indices")
    methods = {}
    for restricted in (False, True):
        dictionary = ip_indices if restricted else list(range(matrix.shape[1]))
        X = matrix[:, dictionary]
        for initialization in ("greedy", "backward"):
            name = ("ip_" if restricted else "") + initialization + "_swap"
            if len(dictionary) < target_dimension:
                methods[name] = dict(status="insufficient_ip_support", indices=None,
                                     loss=None, initial_loss=None, time_seconds=None)
                continue
            start_time = time.perf_counter()
            initializer = greedy if initialization == "greedy" else backward_elimination
            initial = initializer(X, _gl_loss, target_dimension=target_dimension,
                                  random_state=random_state)
            initial_loss = float(_gl_loss(X[:, initial]))
            selected = forward_backward(X, _gl_loss, target_dimension=target_dimension,
                                        selected_indices=initial, random_state=random_state)
            loss = float(_gl_loss(X[:, selected]))
            if not np.isfinite(initial_loss) or not np.isfinite(loss):
                raise ValueError(f"{name} did not produce a finite full-rank basis")
            methods[name] = dict(
                status="ok", indices=[dictionary[j] for j in selected], loss=loss,
                initial_indices=[dictionary[j] for j in initial], initial_loss=initial_loss,
                time_seconds=time.perf_counter() - start_time,
            )
    return methods
