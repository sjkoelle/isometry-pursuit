import numpy as np

from .simulations import simulate_unitary_matrix
from .loss import isometry_loss
from .algorithms import greedy, brute


def run_unitary_experiment(
    ambient_dimension=10, unitary_dimension=5, noise_dimension=50
):

    unitary_matrix = simulate_unitary_matrix(
        ambient_dimension=ambient_dimension,
        unitary_dimension=unitary_dimension,
        noise_dimension=noise_dimension,
    )
    # NOTE (Sam): target dimension is assumed known
    greedy_selection = greedy(
        matrix=unitary_matrix, target_dimension=unitary_dimension, loss=isometry_loss
    )
    brute_selection = brute(
        matrix=unitary_matrix, target_dimension=unitary_dimension, loss=isometry_loss
    )

    #
    greedy_basis_pursuit_selection = greedy(
        matrix=unitary_matrix,
        target_dimension=unitary_dimension,
        loss=basis_pursuit_loss,
    )

    #
    brute_basis_pursuit_selection = brute(
        matrix=unitary_matrix,
        target_dimension=unitary_dimension,
        loss=basis_pursuit_loss,
    )

    # convex basis pursuit (this should give the same answer as brute_basis_pursuit_selection) since the problem is convex
    brute_basis_pursuit_selection = isometric_basis_pursuit(
        matrix=unitary_matrix
    )  # look at rank since more than D functions selected

    # Two stage basis_pursuit_selection
    brute_two_stage_selection = brute(
        matrix=unitary_matrix[brute_basis_pursuit_selection],
        target_dimension=unitary_dimension,
        loss=isometry_loss,
    )  # could also use the basis_pursuit_loss here

    lambdas = np.logspace(1e-6, 1e-3)
    for lambda_ in lambdas:
        lasso_selection = isometric_lasso(matrix=unitary_matrix, lambda_=lambda_)

    # compute angle of the two spaces (its not the angle but kinda... frob norm of projection)
