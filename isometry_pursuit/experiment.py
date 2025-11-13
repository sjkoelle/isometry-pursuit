from math import comb

import pandas as pd
import numpy as np

from .algorithm import (
    greedy,
    brute,
    group_basis_pursuit,
    isometry_pursuit,
    line_search_for_cardinality,
)
from .loss import isometry_loss, group_lasso_norm


def run_tensor_experiment(
    data_tensor,
    steps=None,
    constraint_violation=0.0,
    target_cardinality=None,
    use_line_search=True,
):
    """
    Run experiment using pre-resampled data tensor with optional line search for target cardinality.

    Parameters:
    -----------
    data_tensor : numpy.ndarray
        Pre-resampled data tensor of shape (D, n_samples_per_replicate, n_replicates)
    steps : int, optional
        Maximum number of steps for optimization
    constraint_violation : float
        Constraint violation tolerance (used if not using line search)
    target_cardinality : int, optional
        Target support cardinality (used with line search)
    use_line_search : bool
        Whether to use line search to find optimal constraint violation

    Returns:
    --------
    results_df : pandas.DataFrame
        Results from the experiment
    """
    import numpy as np
    import pandas as pd

    D = data_tensor.shape[0]  # Extract D from tensor shape
    R = data_tensor.shape[2]  # Use all available replicates
    P = data_tensor.shape[1]  # Number of samples per replicate

    # Initialize result lists
    constraint_violations = []
    cardinalities = []
    basis_pursuit_iterations_list = []
    coefficient_matrices = []

    for i in range(R):
        # Extract data for this replicate
        X = data_tensor[:, :, i]

        if use_line_search and target_cardinality is not None:
            # Use line search to find optimal constraint violation
            best_violation, best_cardinality, result = line_search_for_cardinality(
                X,
                target_cardinality=target_cardinality,
            )
            constraint_violations.append(best_violation)
            cardinalities.append(best_cardinality)
        else:
            # Use fixed constraint violation
            result = isometry_pursuit(
                X, steps=steps, constraint_violation=constraint_violation
            )
            constraint_violations.append(constraint_violation)
            cardinalities.append(len(result["basis_pursuit_indices"]))

        basis_pursuit_iterations_list.append(result["basis_pursuit_iterations"])
        coefficient_matrices.append(result["beta"])

    # Create results DataFrame
    results_data = {
        "constraint_violation": constraint_violations,
        "support_cardinality": cardinalities,
        "basis_pursuit_iterations": basis_pursuit_iterations_list,
    }

    results_df = pd.DataFrame(results_data)

    return results_df, coefficient_matrices
