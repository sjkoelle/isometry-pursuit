import pandas as pd
import numpy as np

from .transformation import exponential_transformation
from .algorithm import greedy, brute, group_basis_pursuit
from .loss import isometry_loss, group_lasso_norm


def analyze_data(X, D, compute_brute=False, power=1.0):
    isometry_loss_power = lambda x: isometry_loss(x, power)
    group_brute_loss = lambda x: group_lasso_norm(np.linalg.inv(x))
    output = greedy(X, isometry_loss_power, D, [])
    loss = isometry_loss_power(X[:, output])

    data_transformed = exponential_transformation(X, power=power)
    beta = group_basis_pursuit(data_transformed)
    basis_pursuit_indices = np.where(np.linalg.norm(beta, axis=1))[0]

    two_stage_output = basis_pursuit_indices[
        np.asarray(brute(X[:, basis_pursuit_indices], isometry_loss_power, D))
    ]  # plainly this is too hard 178**13 combinations
    two_stage_loss = isometry_loss_power(X[:, two_stage_output])

    two_stage_multitask = basis_pursuit_indices[
        np.asarray(
            brute(data_transformed[:, basis_pursuit_indices], group_brute_loss, D)
        )
    ]  # plainly this is too hard 178**13 combinations
    greedy_multitask_norm_two_stage = group_brute_loss(
        data_transformed[:, two_stage_multitask]
    )

    random_indices = np.random.choice(
        range(X.shape[1]), len(basis_pursuit_indices), replace=False
    )
    random_two_stage_loss = isometry_loss_power(X[:, random_indices])

    if compute_brute:
        brute_solution = brute(data_transformed[:, :], group_brute_loss, D)
        brute_loss = group_brute_loss(data_transformed[:, brute_solution])

        brute_isometry_solution = brute(X, isometry_loss_power, D)
        brute_isometry_loss = isometry_loss_power(X[:, brute_isometry_solution])
    else:
        brute_loss = np.nan
        brute_isometry_loss = np.nan

    return (
        loss,
        len(basis_pursuit_indices),
        two_stage_loss,
        random_two_stage_loss,
        greedy_multitask_norm_two_stage,
        brute_loss,
        brute_isometry_loss,
    )


def run_resampling_experiment(data, D, frac=0.5, R=25, compute_brute=False, power=1.0):

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
        (
            loss,
            support_cardinality_basis_pursuit,
            two_stage_loss,
            random_two_stage_loss,
            greedy_multitask_norm_two_stage,
            brute_loss,
            brute_isometry_loss,
        ) = analyze_data(X, D, compute_brute=compute_brute, power=1.0)
        losses.append(loss)
        support_cardinalities_basis_pursuit.append(support_cardinality_basis_pursuit)
        two_stage_losses.append(two_stage_loss)
        greedy_multitask_norms_two_stage.append(greedy_multitask_norm_two_stage)
        random_two_stage_losses.append(random_two_stage_loss)
        brute_losses.append(brute_loss)
        brute_isometry_losses.append(brute_isometry_loss)

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
