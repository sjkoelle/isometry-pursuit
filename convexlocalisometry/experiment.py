import numpy as np

from .simulation import simulate_unitary_matrix
from .loss import isometry_loss
from .algorithm import greedy, brute


def run_experiment(matrix, target_dimension, lambdas=np.logspace(1e-6, 1e-3)):
    # NOTE (Sam): target dimension is assumed known
    greedy_selection = greedy(
        matrix=matrix, target_dimension=target_dimension, loss=isometry_loss
    )
    brute_selection = brute(
        matrix=matrix, target_dimension=target_dimension, loss=isometry_loss
    )

    #
    greedy_basis_pursuit_selection = greedy(
        matrix=matrix,
        target_dimension=target_dimension,
        loss=basis_pursuit_loss,
        # We can't do a greedy selection in the same way since we need to invert first.
        # Thus, we need to first invert at d=1, then select the best candidate w.r.t. the inversion, then select at d=2, invert, select, etc.
    )

    #
    brute_basis_pursuit_selection = brute(
        matrix=matrix,
        target_dimension=target_dimension,
        loss=basis_pursuit_loss,  # This isn't quite right.  I was thinking to do this on the dual.  Does it make sense here? % Direct minimization of this norm has nothing to do with correlation.
    )

    # convex basis pursuit (this should give the same answer as brute_basis_pursuit_selection) since the problem is convex
    brute_basis_pursuit_selection = isometric_basis_pursuit(
        matrix=matrix
    )  # look at rank since more than D functions selected

    # Two stage basis_pursuit_selection
    brute_two_stage_selection = brute(
        matrix=matrix[brute_basis_pursuit_selection],
        target_dimension=target_dimension,
        loss=isometry_loss,
    )  # could also use the basis_pursuit_loss here

    for lambda_ in lambdas:
        lasso_selection = isometric_lasso(matrix=matrix, lambda_=lambda_)

    # How many functions are retained?
    # compute angle of the two spaces (its not the angle but kinda... frob norm of projection)


def run_unitary_experiment(
    ambient_dimension=10, unitary_dimension=5, noise_dimension=50
):

    unitary_matrix = simulate_unitary_matrix(
        ambient_dimension=ambient_dimension,
        unitary_dimension=unitary_dimension,
        noise_dimension=noise_dimension,
    )
    run_experiment(matrix=unitary_matrix, target_dimension=unitary_dimension)
