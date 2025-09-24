import matplotlib.pyplot as plt
import numpy as np


def ecdf_plot(losses_1, losses_2, title, xlabel, ylabel, filename):
    # Compute difference and subtract D baseline
    diff = np.asarray(losses_1) - np.asarray(losses_2)

    # Sort values for ECDF
    x = np.sort(diff)
    y = np.arange(1, len(x) + 1) / len(x)

    plt.figure(figsize=(10, 7))
    plt.scatter(x, y, color="black", alpha=0.8, s=50)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)

    # Labeling
    plt.xlabel(
        xlabel, fontsize=24
    )  # e.g., "$\ell_1(\hat{S}_{g}) - \ell_1(\hat{S}_{TSIP})$"
    plt.ylabel(ylabel, fontsize=24)  # e.g., "Empirical CDF"
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.title(title, fontsize=18)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def difference_box_plot(losses_1, losses_2, title, xlabel, ylabel, filename):
    """
    Create a box plot showing the difference between two loss methods.
    This replaces the ECDF plot functionality with a box plot.
    """
    # Compute difference (same as ECDF plot)
    diff = np.asarray(losses_1) - np.asarray(losses_2)

    plt.figure(figsize=(10, 7))

    # Create box plot of the differences
    box = plt.boxplot(
        [diff], patch_artist=True, medianprops=dict(color="black", linewidth=2)
    )

    # Customize boxplot colors
    box["boxes"][0].set_facecolor("lightblue")

    # Add horizontal line at 0 (same as ECDF plot)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)

    # Labeling (same as ECDF plot)
    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    plt.xticks([1], [title], fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def box_plot(losses_1, losses_2, D, xlabels, ylabel, filename):
    """
    Create a box plot showing the difference between two loss methods.
    This is the box plot equivalent of the ECDF plot functionality.
    Note: D parameter is kept for compatibility but not used since losses
    are now pre-processed to subtract D at the experiment level.
    """
    # Compute difference (same as ECDF plot)
    diff = np.asarray(losses_1) - np.asarray(losses_2)

    plt.figure(figsize=(10, 7))

    # Create single box plot of the differences
    box = plt.boxplot(
        [diff], patch_artist=True, medianprops=dict(color="black", linewidth=2)
    )

    # Customize boxplot color
    box["boxes"][0].set_facecolor("lightblue")

    # Add horizontal line at 0 (same as ECDF plot)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)

    # Labeling
    plt.xlabel(f"{xlabels[0]} - {xlabels[1]}", fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    plt.xticks([1], ["Difference"], fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.savefig(filename, bbox_inches="tight")
    plt.close()


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
