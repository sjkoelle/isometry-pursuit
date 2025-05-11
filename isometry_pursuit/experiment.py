from math import comb
import time

import pandas as pd
import numpy as np

from .transformation import exponential_transformation
from .algorithm import greedy, brute, group_basis_pursuit
from .loss import isometry_loss, group_lasso_norm


def analyze_data(X, compute_brute=False, power=1.0, limit=1e9):
    D, P = X.shape
    isometry_loss_power = lambda x: isometry_loss(x, power)
    group_brute_loss = lambda x: group_lasso_norm(np.linalg.inv(x))
    start_greedy = time.time()
    output = greedy(X, isometry_loss_power, D, [])
    end_greedy = time.time()
    loss = isometry_loss_power(X[:, output])

    data_transformed = exponential_transformation(X, power=power)
    start_basis_pursuit = time.time()
    beta = group_basis_pursuit(data_transformed)
    end_basis_pursuit = time.time()
    basis_pursuit_indices = np.where(np.linalg.norm(beta, axis=1))[0]
    nbp = len(basis_pursuit_indices)
    brute_complexity = comb(nbp, D)
    print(f"Brute force complexity {brute_complexity} from D={D} and nbp={nbp}")
    if brute_complexity <= limit:
        start_two_stage = time.time()
        two_stage_output = basis_pursuit_indices[
            np.asarray(brute(X[:, basis_pursuit_indices], isometry_loss_power, D))
        ]  # plainly this is too hard 178**13 combinations
        end_two_stage = time.time()
        stage_two_time = end_two_stage - start_two_stage
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
    else:
        print("Brute force is too computationally expensive")
        two_stage_loss = np.nan
        random_two_stage_loss = np.nan
        greedy_multitask_norm_two_stage = np.nan
        stage_two_time = np.nan

    if compute_brute:
        if brute_complexity > limit:
            print("Brute force is too computationally expensive")
            brute_loss = np.nan
            brute_isometry_loss = np.nan
        else:
            brute_solution = brute(data_transformed[:, :], group_brute_loss, D)
            brute_loss = group_brute_loss(data_transformed[:, brute_solution])

            brute_isometry_solution = brute(X, isometry_loss_power, D)
            brute_isometry_loss = isometry_loss_power(X[:, brute_isometry_solution])
    else:
        brute_loss = np.nan
        brute_isometry_loss = np.nan

    print("stage two time", stage_two_time)
    print("greedy time", end_greedy - start_greedy)
    print("basis pursuit time", end_basis_pursuit - start_basis_pursuit)

    return (
        loss,
        len(basis_pursuit_indices),
        two_stage_loss,
        random_two_stage_loss,
        greedy_multitask_norm_two_stage,
        brute_loss,
        brute_isometry_loss,
        end_basis_pursuit - start_basis_pursuit,
        stage_two_time,
        end_greedy - start_greedy,
        two_stage_multitask,
    )


def run_resampling_experiment(data, D, frac=0.5, R=25, compute_brute=False, power=1.0):

    losses = []
    support_cardinalities_basis_pursuit = []
    two_stage_losses = []
    random_two_stage_losses = []
    greedy_multitask_norms_two_stage = []
    brute_isometry_losses = []
    brute_losses = []
    greedy_times = []
    basis_pursuit_times = []
    stage_two_times = []
    P = data.shape[0]
    co_occurence_matrix = np.zeros((P, P))
    for i in range(R):
        np.random.seed(i)
        # Select 25 out of 50 indices
        random_indices = np.random.choice(range(P), 25, replace=False)
        X = data.to_numpy()[random_indices].transpose()[:D, :]  # .5
        (
            loss,
            support_cardinality_basis_pursuit,
            two_stage_loss,
            random_two_stage_loss,
            greedy_multitask_norm_two_stage,
            brute_loss,
            brute_isometry_loss,
            basis_pursuit_time,
            stage_two_time,
            greedy_time,
            two_stage_multitask,
        ) = analyze_data(X, compute_brute=compute_brute, power=power)
        losses.append(loss)
        support_cardinalities_basis_pursuit.append(support_cardinality_basis_pursuit)
        two_stage_losses.append(two_stage_loss)
        greedy_multitask_norms_two_stage.append(greedy_multitask_norm_two_stage)
        random_two_stage_losses.append(random_two_stage_loss)
        brute_losses.append(brute_loss)
        brute_isometry_losses.append(brute_isometry_loss)
        basis_pursuit_times.append(basis_pursuit_time)
        stage_two_times.append(stage_two_time)
        greedy_times.append(greedy_time)
        from itertools import combinations

        co_occurences = combinations(two_stage_multitask, 2)
        for co in co_occurences:
            co_occurence_matrix[random_indices[co[0]], random_indices[co[1]]] += 1
            co_occurence_matrix[random_indices[co[1]], random_indices[co[0]]] += 1

    results_df = pd.DataFrame(
        {
            "Losses": losses,
            "Support Cardinalities (Basis Pursuit)": support_cardinalities_basis_pursuit,
            "Two-Stage Losses": two_stage_losses,
            "Random Two-Stage Losses": random_two_stage_losses,
            "Greedy Multitask Norms (Two-Stage)": greedy_multitask_norms_two_stage,
            "Brute Isometry Losses": brute_isometry_losses,
            "Brute Losses": brute_losses,
            "Basis Pursuit Times": basis_pursuit_times,
            "Stage Two Times": stage_two_times,
            "Greedy Times": greedy_times,
        }
    )

    return results_df, co_occurence_matrix


2 + 2
