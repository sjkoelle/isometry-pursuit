import numpy as np

from .algorithm import isometry_pursuit
from .loss import isometry_loss, group_lasso_norm


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
        loss = isometry_loss(X[:, indices]) if len(indices) > 0 else None
        replicates.append({
            "replicate": i,
            "support_cardinality": len(indices),
            "isometry_loss": loss,
            "group_lasso_norm": float(group_lasso_norm(result["beta"])),
            "basis_pursuit_iterations": result["basis_pursuit_iterations"],
            "basis_pursuit_indices": indices.tolist(),
        })
        coefficient_matrices.append(result["beta"])

    return replicates, coefficient_matrices
