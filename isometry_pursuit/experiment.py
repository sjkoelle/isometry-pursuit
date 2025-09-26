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

THRESHOLD = 1e-4


def analyze_data(
    X,
    compute_brute=False,
    power=1.0,
    limit=1e6,
    max_steps=None,
    log_convergence=False,
    constraint_violation=0.0,
):
    D, P = X.shape

    isometry_loss_power = lambda x, indices=None: isometry_loss(x, power, indices)
    group_brute_loss = lambda x, indices=None: group_lasso_norm(np.linalg.inv(x))
    start_greedy = time.time()
    output = greedy(X, isometry_loss_power, D, [], verbose=False)

    end_greedy = time.time()
    loss = isometry_loss_power(X[:, output])

    data_transformed = exponential_transformation(X, power=power)
    # concatenate data_transformed with an identity matrix of shape (D, D)
    data_transformed = np.concatenate([data_transformed, np.identity(D)], axis=1)
    X = np.concatenate([X, np.identity(D)], axis=1)
    # breakpoint()
    P = D + P
    print(np.linalg.svd(data_transformed, compute_uv=False))
    start_basis_pursuit = time.time()

    # Use CVXPY implementation
    beta, basis_pursuit_iterations = group_basis_pursuit(
        data_transformed, constraint_violation=constraint_violation
    )

    # Note: Convergence logging is not implemented for the main experiment flow
    # since it would require calling group_basis_pursuit multiple times with different parameters
    # For convergence analysis, use the standalone convergence logging functions
    convergence_log = None

    end_basis_pursuit = time.time()
    basis_pursuit_indices = np.where(np.linalg.norm(beta, axis=1) > THRESHOLD)[0]
    nbp = len(basis_pursuit_indices)
    brute_complexity = comb(nbp, D)
    print(f"Brute force complexity {brute_complexity} from D={D} and nbp={nbp}")

    # Calculate loss when basis pursuit achieves exactly D cardinality (no second stage needed)
    if nbp == D:
        # Basis pursuit selected exactly D features, so we can use them directly
        # breakpoint()
        d_cardinality_loss = isometry_loss_power(X[:, basis_pursuit_indices])
        print(f"Basis pursuit achieved D cardinality! Loss: {d_cardinality_loss:.4f}")
    else:
        d_cardinality_loss = np.nan
        print(f"Basis pursuit selected {nbp} features (target: {D})")

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
        two_stage_multitask = None

    # Greedy on isometry pursuit (new method)
    # First stage: Isometry pursuit (basis pursuit) -> basis_pursuit_indices
    # Second stage: Greedy on the basis pursuit selection
    greedy_on_bp_complexity = len(basis_pursuit_indices) * D

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
        print("Greedy on isometry pursuit is too computationally expensive")
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
        d_cardinality_loss,
        beta,  # Coefficient matrix from basis pursuit
    )


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


def run_tensor_experiment(
    data_tensor,
    compute_brute=False,
    power=1.0,
    steps=None,
    log_convergence=False,
    constraint_violation=0.0,
):
    """
    Run experiment using pre-resampled data tensor.

    Parameters:
    -----------
    data_tensor : numpy.ndarray
        Pre-resampled data tensor of shape (D, n_samples_per_replicate, n_replicates)
    compute_brute : bool
        Whether to compute brute force solutions
    power : float
        Power parameter for transformation
    steps : int, optional
        Maximum number of steps for optimization
    log_convergence : bool
        Whether to log convergence information

    Returns:
    --------
    results_df : pandas.DataFrame
        Results from the experiment
    co_occurrence_matrix : np.ndarray
        Co-occurrence matrix for two-stage solutions
    greedy_co_occurrence_matrix : np.ndarray
        Co-occurrence matrix for greedy solutions
    convergence_logs : list or None
        Convergence logs if log_convergence=True
    """
    import numpy as np
    import pandas as pd

    D = data_tensor.shape[0]  # Extract D from tensor shape
    R = data_tensor.shape[2]  # Use all available replicates

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
    d_cardinality_losses = []
    greedy_d_cardinality_deltas = []
    coefficient_matrices = []

    P = data_tensor.shape[1]  # Number of samples per replicate
    co_occurence_matrix = np.zeros((P, P))
    greedy_co_occurence_matrix = np.zeros((P, P))

    for i in range(R):
        # Extract data for this replicate
        X = data_tensor[:, :, i]

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
            d_cardinality_loss,
            beta_matrix,
        ) = analyze_data(
            X,
            compute_brute=compute_brute,
            power=power,
            log_convergence=log_convergence,
            constraint_violation=constraint_violation,
        )
        print("greedy loss", loss)

        losses.append(loss)
        support_cardinalities_basis_pursuit.append(support_cardinality_basis_pursuit)
        two_stage_losses.append(two_stage_loss)
        greedy_multitask_norms_two_stage.append(greedy_multitask_norm_two_stage)
        random_two_stage_losses.append(random_two_stage_loss)
        brute_losses.append(brute_loss)
        brute_isometry_losses.append(brute_isometry_loss)
        greedy_times.append(greedy_time)
        basis_pursuit_times.append(basis_pursuit_time)
        stage_two_times.append(stage_two_time)
        basis_pursuit_iterations_list.append(basis_pursuit_iterations)
        convergence_logs.append(convergence_log)
        two_stage_greedy_losses.append(two_stage_greedy_loss)
        two_stage_greedy_times.append(two_stage_greedy_time)
        d_cardinality_losses.append(d_cardinality_loss)
        greedy_d_cardinality_deltas.append(greedy_d_cardinality_delta)
        coefficient_matrices.append(beta_matrix)

        # Update co-occurrence matrices
        if two_stage_multitask is not None:
            # two_stage_multitask is an array of indices, not a coefficient matrix
            for idx in two_stage_multitask:
                for jdx in two_stage_multitask:
                    co_occurence_matrix[idx, jdx] += 1

        if greedy_solution is not None:
            # greedy_solution is an array of indices, not a coefficient matrix
            for idx in greedy_solution:
                for jdx in greedy_solution:
                    greedy_co_occurence_matrix[idx, jdx] += 1

    # Create results DataFrame
    results_data = {
        "loss": losses,
        "support_cardinality_basis_pursuit": support_cardinalities_basis_pursuit,
        "two_stage_loss": two_stage_losses,
        "random_two_stage_loss": random_two_stage_losses,
        "greedy_multitask_norm_two_stage": greedy_multitask_norms_two_stage,
        "brute_loss": brute_losses,
        "brute_isometry_loss": brute_isometry_losses,
        "greedy_time": greedy_times,
        "basis_pursuit_time": basis_pursuit_times,
        "stage_two_time": stage_two_times,
        "basis_pursuit_iterations": basis_pursuit_iterations_list,
        "two_stage_greedy_loss": two_stage_greedy_losses,
        "two_stage_greedy_time": two_stage_greedy_times,
        "d_cardinality_loss": d_cardinality_losses,
        "greedy_d_cardinality_delta": greedy_d_cardinality_deltas,
    }

    results_df = pd.DataFrame(results_data)

    if log_convergence:
        return (
            results_df,
            co_occurence_matrix,
            greedy_co_occurence_matrix,
            convergence_logs,
            coefficient_matrices,
        )
    else:
        return (
            results_df,
            co_occurence_matrix,
            greedy_co_occurence_matrix,
            coefficient_matrices,
        )
