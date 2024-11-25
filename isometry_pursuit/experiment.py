import pandas as pd
import numpy as np

from .transformation import exponential_transformation
from .algorithm import greedy, brute, group_basis_pursuit
from .loss import isometry_loss, group_lasso_norm


def run_experiment(data, D, frac=0.5, R=25, compute_brute=False, power=1.0):

    isometry_loss_power = lambda x: isometry_loss(x, power)
    group_brute_loss = lambda x: group_lasso_norm(np.linalg.inv(x))
    losses = []
    support_cardinalities_basis_pursuit = []
    two_stage_losses = []
    random_two_stage_losses = []
    greedy_multitask_norms_two_stage = []
    brute_isometry_losses = []
    brute_losses = []

    for i in range(R):
        np.random.seed(i)
        X = data.sample(frac=frac).to_numpy().transpose()  # .5
        print("Data subsampled dimension", X.shape)
        output = greedy(X, isometry_loss_power, D, [])
        loss = isometry_loss_power(X[:, output])
        losses.append(loss)

        data_transformed = exponential_transformation(X, power=power)
        beta = group_basis_pursuit(data_transformed)
        basis_pursuit_indices = np.where(np.linalg.norm(beta, axis=1))[0]

        support_cardinalities_basis_pursuit.append(len(basis_pursuit_indices))

        two_stage_output = basis_pursuit_indices[
            np.asarray(brute(X[:, basis_pursuit_indices], isometry_loss_power, D))
        ]  # plainly this is too hard 178**13 combinations
        two_stage_loss = isometry_loss_power(X[:, two_stage_output])
        two_stage_losses.append(two_stage_loss)

        two_stage_multitask = basis_pursuit_indices[
            np.asarray(
                brute(data_transformed[:, basis_pursuit_indices], group_brute_loss, D)
            )
        ]  # plainly this is too hard 178**13 combinations
        greedy_multitask_norms_two_stage.append(
            group_brute_loss(data_transformed[:, two_stage_multitask])
        )

        random_indices = np.random.choice(
            range(X.shape[1]), len(basis_pursuit_indices), replace=False
        )
        random_two_stage_losses.append(isometry_loss_power(X[:, random_indices]))

        if compute_brute:
            brute_solution = brute(data_transformed[:, :], group_brute_loss, D)
            brute_losses.append(group_brute_loss(data_transformed[:, brute_solution]))

            brute_isometry_solution = brute(X, isometry_loss_power, D)
            brute_isometry_losses.append(
                isometry_loss_power(X[:, brute_isometry_solution])
            )
        else:
            brute_losses.append(np.nan)
            brute_isometry_losses.append(np.nan)

    # Creating the dataframe
    results_df = pd.DataFrame(
        {
            "Losses": losses,
            "Support Cardinalities (Basis Pursuit)": support_cardinalities_basis_pursuit,
            "Two-Stage Losses": two_stage_losses,
            "Random Two-Stage Losses": random_two_stage_losses,
            "Greedy Multitask Norms (Two-Stage)": greedy_multitask_norms_two_stage,
            "Brute Isometry Losses": brute_isometry_losses,
            "Brute Losses": brute_losses,
        }
    )
    return results_df
