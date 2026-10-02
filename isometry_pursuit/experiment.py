import numpy as np

from .algorithm import isometry_pursuit, greedy, backward_elimination
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


def run_tensor_experiment(data_tensor, constraint_violation=0.0, iterations=None, eps=None):
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
        result = isometry_pursuit(X, constraint_violation=constraint_violation, iterations=iterations, eps=eps)
        indices = result["basis_pursuit_indices"]

        exp_X = exponential_transformation(X)

        # Greedy on full exp_X using group lasso norm
        greedy_gl_indices = [int(j) for j in greedy(exp_X, _gl_loss)]
        greedy_gl_norm = float(_gl_loss(exp_X[:, greedy_gl_indices]))

        # Greedy on BP-selected subset of exp_X using group lasso norm
        if len(indices) >= X.shape[0]:
            exp_X_bp = exp_X[:, indices]
            local_indices = greedy(exp_X_bp, _gl_loss)
            greedy_gl_bp_indices = [int(indices[j]) for j in local_indices]
            greedy_gl_bp_norm = float(_gl_loss(exp_X[:, greedy_gl_bp_indices]))

            local_backward_indices = backward_elimination(exp_X_bp, _gl_loss)
            backward_gl_bp_indices = [int(indices[j]) for j in local_backward_indices]
            backward_gl_bp_norm = float(_gl_loss(exp_X[:, backward_gl_bp_indices]))
        else:
            greedy_gl_bp_indices = None
            greedy_gl_bp_norm = None
            backward_gl_bp_indices = None
            backward_gl_bp_norm = None

        # Compute group lasso norm of pseudoinverse for convex solution
        bp_gl_norm = float(_gl_loss(exp_X[:, indices]))

        replicates.append({
            "replicate": i,
            "support_cardinality": len(indices),
            "group_lasso_norm": bp_gl_norm,
            "basis_pursuit_iterations": result["basis_pursuit_iterations"],
            "basis_pursuit_indices": indices.tolist(),
            "greedy_gl_indices": greedy_gl_indices,
            "greedy_gl_norm": greedy_gl_norm,
            "greedy_gl_bp_indices": greedy_gl_bp_indices,
            "greedy_gl_bp_norm": greedy_gl_bp_norm,
            "backward_gl_bp_indices": backward_gl_bp_indices,
            "backward_gl_bp_norm": backward_gl_bp_norm,
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
