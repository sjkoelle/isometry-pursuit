from math import comb
import time

import pandas as pd
import numpy as np

from .transformation import exponential_transformation
from .algorithm import (
    greedy,
    brute,
    group_basis_pursuit,
    create_convergence_log,
    create_convergence_log_with_coefficients,
)
from .loss import isometry_loss, group_lasso_norm


def analyze_data(
    X,
    compute_brute=False,
    power=1.0,
    limit=1e9,
    max_steps=None,
    log_convergence=False,
):
    D, P = X.shape
    isometry_loss_power = lambda x, indices=None: isometry_loss(x, power, indices)
    group_brute_loss = lambda x, indices=None: group_lasso_norm(np.linalg.inv(x))
    start_greedy = time.time()
    output = greedy(X, isometry_loss_power, D, [], verbose=False)

    end_greedy = time.time()
    loss = isometry_loss_power(X[:, output])

    data_transformed = exponential_transformation(X, power=power)
    start_basis_pursuit = time.time()

    # Use CVXPY implementation
    beta, basis_pursuit_iterations = group_basis_pursuit(data_transformed)

    # Note: Convergence logging is not implemented for the main experiment flow
    # since it would require calling group_basis_pursuit multiple times with different parameters
    # For convergence analysis, use the standalone convergence logging functions
    convergence_log = None

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

    # Two-stage on greedy (new method)
    # First stage: Isometry pursuit (basis pursuit) -> basis_pursuit_indices
    # Second stage: Greedy on the basis pursuit selection
    greedy_on_bp_complexity = comb(len(basis_pursuit_indices), D)
    print(
        f"Two-stage on greedy complexity {greedy_on_bp_complexity} from D={D} and bp_size={len(basis_pursuit_indices)}"
    )
    if greedy_on_bp_complexity <= limit:
        start_two_stage_greedy = time.time()
        # Apply greedy algorithm to the basis pursuit selection
        # Use original data X (not transformed) since isometry_loss_power expects original data
        two_stage_greedy_output = basis_pursuit_indices[
            greedy(
                X[:, basis_pursuit_indices], isometry_loss_power, D, [], verbose=False
            )
        ]
        end_two_stage_greedy = time.time()
        two_stage_greedy_time = end_two_stage_greedy - start_two_stage_greedy
        two_stage_greedy_loss = isometry_loss_power(X[:, two_stage_greedy_output])
    else:
        print("Two-stage on greedy is too computationally expensive")
        two_stage_greedy_loss = np.nan
        two_stage_greedy_time = np.nan

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
    print("basis pursuit iterations", basis_pursuit_iterations)

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
        output,
        basis_pursuit_iterations,
        convergence_log,
        two_stage_greedy_loss,
        two_stage_greedy_time,
    )


def run_resampling_experiment(
    data,
    D,
    frac=0.5,
    R=25,
    compute_brute=False,
    power=1.0,
    steps=None,
    log_convergence=False,
):

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
    basis_pursuit_iterations_list = []
    convergence_logs = []
    two_stage_greedy_losses = []
    two_stage_greedy_times = []
    P = data.shape[0]
    co_occurence_matrix = np.zeros((P, P))
    greedy_co_occurence_matrix = np.zeros((P, P))

    for i in range(R):
        np.random.seed(i)
        # Select 25 out of 50 indices
        random_indices = np.random.choice(range(P), int(P / 2), replace=False)
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
            greedy_solution,
            basis_pursuit_iterations,
            convergence_log,
            two_stage_greedy_loss,
            two_stage_greedy_time,
        ) = analyze_data(
            X,
            compute_brute=compute_brute,
            power=power,
            log_convergence=log_convergence,
        )
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
        basis_pursuit_iterations_list.append(basis_pursuit_iterations)
        convergence_logs.append(convergence_log)
        two_stage_greedy_losses.append(two_stage_greedy_loss)
        two_stage_greedy_times.append(two_stage_greedy_time)
        from itertools import combinations

        co_occurences = combinations(two_stage_multitask, 2)
        greedy_co_occurences = combinations(greedy_solution, 2)
        for co in co_occurences:
            co_occurence_matrix[random_indices[co[0]], random_indices[co[1]]] += 1
            co_occurence_matrix[random_indices[co[1]], random_indices[co[0]]] += 1

        for co in greedy_co_occurences:
            greedy_co_occurence_matrix[
                random_indices[co[0]], random_indices[co[1]]
            ] += 1
            greedy_co_occurence_matrix[
                random_indices[co[1]], random_indices[co[0]]
            ] += 1

    # Subtract D from loss values to get excess loss above theoretical minimum
    results_df = pd.DataFrame(
        {
            "Losses": np.array(losses) - D,
            "Support Cardinalities (Basis Pursuit)": support_cardinalities_basis_pursuit,
            "Two-Stage Losses": np.array(two_stage_losses) - D,
            "Random Two-Stage Losses": np.array(random_two_stage_losses) - D,
            "Greedy Multitask Norms (Two-Stage)": greedy_multitask_norms_two_stage,
            "Brute Isometry Losses": (
                np.array(brute_isometry_losses) - D
                if brute_isometry_losses
                else brute_isometry_losses
            ),
            "Brute Losses": (
                np.array(brute_losses) - D if brute_losses else brute_losses
            ),
            "Basis Pursuit Times": basis_pursuit_times,
            "Stage Two Times": stage_two_times,
            "Greedy Times": greedy_times,
            "Basis Pursuit Iterations": basis_pursuit_iterations_list,
            "Two-Stage on Greedy Losses": np.array(two_stage_greedy_losses) - D,
            "Two-Stage on Greedy Times": two_stage_greedy_times,
        }
    )

    if log_convergence:
        return (
            results_df,
            co_occurence_matrix,
            greedy_co_occurence_matrix,
            convergence_logs,
        )
    return results_df, co_occurence_matrix, greedy_co_occurence_matrix


def plot_convergence(convergence_log, title="Convergence Analysis", save_path=None):
    """
    Plot convergence metrics (loss, sparsity, residual) over iterations.

    Parameters:
    -----------
    convergence_log : dict
        Dictionary containing 'iterations', 'losses', 'sparsities', 'residuals' lists
    title : str
        Title for the plot
    save_path : str, optional
        Path to save the plot
    """
    import matplotlib.pyplot as plt

    if not convergence_log or not convergence_log["iterations"]:
        print("No convergence data to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=16)

    # Plot 1: Loss over iterations
    axes[0, 0].semilogx(
        convergence_log["iterations"], convergence_log["losses"], "b-o", markersize=4
    )
    axes[0, 0].set_xlabel("Iterations")
    axes[0, 0].set_ylabel("Loss (Objective Value)")
    axes[0, 0].set_title("Loss Convergence")
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Sparsity over iterations
    axes[0, 1].semilogx(
        convergence_log["iterations"],
        convergence_log["sparsities"],
        "r-s",
        markersize=4,
    )
    axes[0, 1].set_xlabel("Iterations")
    axes[0, 1].set_ylabel("Sparsity (Non-zero Rows)")
    axes[0, 1].set_title("Sparsity Evolution")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Residual over iterations
    axes[1, 0].semilogx(
        convergence_log["iterations"], convergence_log["residuals"], "g-^", markersize=4
    )
    axes[1, 0].set_xlabel("Iterations")
    axes[1, 0].set_ylabel("Constraint Residual")
    axes[1, 0].set_title("Constraint Satisfaction")
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Loss vs Sparsity trade-off
    axes[1, 1].plot(
        convergence_log["sparsities"], convergence_log["losses"], "m-d", markersize=4
    )
    axes[1, 1].set_xlabel("Sparsity (Non-zero Rows)")
    axes[1, 1].set_ylabel("Loss (Objective Value)")
    axes[1, 1].set_title("Loss vs Sparsity Trade-off")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Convergence plot saved to {save_path}")

    plt.show()


def analyze_convergence_logs(convergence_logs, save_path=None):
    """
    Analyze and plot convergence logs from multiple runs.

    Parameters:
    -----------
    convergence_logs : list
        List of convergence log dictionaries
    save_path : str, optional
        Path to save the analysis plot
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if not convergence_logs or all(log is None for log in convergence_logs):
        print("No convergence data to analyze")
        return

    # Filter out None logs
    valid_logs = [
        log for log in convergence_logs if log is not None and log["iterations"]
    ]

    if not valid_logs:
        print("No valid convergence data to analyze")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Convergence Analysis Across Multiple Runs", fontsize=16)

    # Plot 1: Loss convergence (all runs)
    for i, log in enumerate(valid_logs):
        axes[0, 0].semilogx(
            log["iterations"], log["losses"], alpha=0.7, label=f"Run {i+1}"
        )
    axes[0, 0].set_xlabel("Iterations")
    axes[0, 0].set_ylabel("Loss (Objective Value)")
    axes[0, 0].set_title("Loss Convergence (All Runs)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Plot 2: Sparsity evolution (all runs)
    for i, log in enumerate(valid_logs):
        axes[0, 1].semilogx(
            log["iterations"], log["sparsities"], alpha=0.7, label=f"Run {i+1}"
        )
    axes[0, 1].set_xlabel("Iterations")
    axes[0, 1].set_ylabel("Sparsity (Non-zero Rows)")
    axes[0, 1].set_title("Sparsity Evolution (All Runs)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Plot 3: Final loss distribution
    final_losses = [log["losses"][-1] for log in valid_logs if log["losses"]]
    axes[1, 0].hist(
        final_losses, bins=min(10, len(final_losses)), alpha=0.7, edgecolor="black"
    )
    axes[1, 0].set_xlabel("Final Loss")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Final Loss Distribution")
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Final sparsity distribution
    final_sparsities = [
        log["sparsities"][-1] for log in valid_logs if log["sparsities"]
    ]
    axes[1, 1].hist(
        final_sparsities,
        bins=min(10, len(final_sparsities)),
        alpha=0.7,
        edgecolor="black",
    )
    axes[1, 1].set_xlabel("Final Sparsity")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Final Sparsity Distribution")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Convergence analysis saved to {save_path}")

    plt.show()

    # Print summary statistics
    print(f"\nConvergence Analysis Summary:")
    print(f"Number of runs: {len(valid_logs)}")
    print(
        f"Average final loss: {np.mean(final_losses):.4f} ± {np.std(final_losses):.4f}"
    )
    print(
        f"Average final sparsity: {np.mean(final_sparsities):.2f} ± {np.std(final_sparsities):.2f}"
    )
    print(
        f"Average iterations to converge: {np.mean([len(log['iterations']) for log in valid_logs]):.1f}"
    )
