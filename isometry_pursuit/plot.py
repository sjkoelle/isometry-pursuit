import matplotlib.pyplot as plt
import numpy as np


def get_dataset_color_mapping(dataset_names):
    """
    Create a consistent color mapping for datasets across all plotting functions.

    Parameters:
    -----------
    dataset_names : list
        List of dataset names

    Returns:
    --------
    dict
        Dictionary mapping dataset names to colors
    """
    # Standardize dataset names to ensure consistent mapping
    standardized_names = []
    for name in dataset_names:
        # Remove common suffixes to standardize names
        standardized = (
            name.replace("_standardized", "")
            .replace("_results", "")
            .replace("_convergence_all_replicates", "")
        )
        standardized_names.append(standardized)

    # Get unique standardized names in a consistent order
    unique_standardized = sorted(list(set(standardized_names)))

    # Create color mapping
    default_colors = (
        plt.rcParams["axes.prop_cycle"]
        .by_key()
        .get("color", ["C0", "C1", "C2", "C3", "C4", "C5"])
    )

    # Map each standardized name to a color
    standardized_to_color = {
        name: default_colors[i % len(default_colors)]
        for i, name in enumerate(unique_standardized)
    }

    # Map original names to colors based on their standardized version
    color_mapping = {}
    for original_name in dataset_names:
        standardized = (
            original_name.replace("_standardized", "")
            .replace("_results", "")
            .replace("_convergence_all_replicates", "")
        )
        color_mapping[original_name] = standardized_to_color[standardized]

    return color_mapping


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

    # Use symlog scaling for y-axis to handle both positive and negative values
    plt.yscale("symlog")

    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def box_plot(
    data_list,
    labels=None,
    title="Box Plot",
    xlabel="Dataset",
    ylabel="Value",
    filename="boxplot.png",
    figsize=(10, 7),
    colors=None,
    show_zero_line=False,
    use_consistent_colors=True,
):
    """
    Create a box plot for multiple datasets.

    Parameters:
    -----------
    data_list : list
        List of arrays, each containing data for one dataset
    labels : list, optional
        Labels for each dataset
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    filename : str
        Output filename
    figsize : tuple
        Figure size
    colors : list, optional
        Colors for each box
    show_zero_line : bool
        Whether to show a horizontal line at zero
    """
    plt.figure(figsize=figsize)

    # Create box plot
    box = plt.boxplot(
        data_list,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        labels=labels,
    )

    # Set colors
    if colors is None:
        if use_consistent_colors and labels:
            # Use consistent color mapping based on dataset names
            color_mapping = get_dataset_color_mapping(labels)
            colors = [color_mapping[label] for label in labels]
        else:
            # Fallback to default color cycling
            default_colors = (
                plt.rcParams["axes.prop_cycle"]
                .by_key()
                .get("color", ["C0", "C1", "C2", "C3", "C4", "C5"])
            )
            # Cycle through colors if we have more datasets than default colors
            colors = [
                default_colors[i % len(default_colors)] for i in range(len(data_list))
            ]

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor(color)  # Make edge color match fill color
        patch.set_linewidth(1.5)

    # Add horizontal line at zero if requested
    if show_zero_line:
        plt.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)

    # Set labels
    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    plt.title(title, fontsize=28, fontweight="bold", pad=20)

    # Use symlog scaling for y-axis if there might be both positive and negative values
    if show_zero_line:
        plt.yscale("symlog")

    # Rotate x-axis labels if there are many datasets
    if labels and len(labels) > 4:
        plt.xticks(rotation=45, ha="right", fontsize=20)
    else:
        plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Save plot
    plt.savefig(filename, bbox_inches="tight", dpi=300)
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
