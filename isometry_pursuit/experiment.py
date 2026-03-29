import numpy as np

from .algorithm import isometry_pursuit, greedy
from .loss import group_lasso_norm
from .transformation import exponential_transformation


def _gl_loss(sub_matrix, indices=None):
    return group_lasso_norm(np.linalg.pinv(sub_matrix))


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
        else:
            greedy_gl_bp_indices = None
            greedy_gl_bp_norm = None

        replicates.append({
            "replicate": i,
            "support_cardinality": len(indices),
            "group_lasso_norm": float(group_lasso_norm(result["beta"])),
            "basis_pursuit_iterations": result["basis_pursuit_iterations"],
            "basis_pursuit_indices": indices.tolist(),
            "greedy_gl_indices": greedy_gl_indices,
            "greedy_gl_norm": greedy_gl_norm,
            "greedy_gl_bp_indices": greedy_gl_bp_indices,
            "greedy_gl_bp_norm": greedy_gl_bp_norm,
        })
        coefficient_matrices.append(result["beta"])

    return replicates, coefficient_matrices
