#!/usr/bin/env python3
"""
Regularization path runner.
This script runs isometry pursuit with different constraint violation values
and plots the regularization path.
"""

import click
import os
import numpy as np
import matplotlib.pyplot as plt
from isometry_pursuit.algorithm import isometry_pursuit


@click.command()
@click.option(
    "--data-file",
    required=True,
    help="Path to .npy data matrix file",
)
@click.option("--output-dir", required=True, help="Output directory name")
@click.option(
    "--constraint-violations",
    default="0.0,0.1,0.5,1.0,2.0,5.0,10.0",
    help="Comma-separated list of constraint violation values (default: 0.0,0.1,0.5,1.0,2.0,5.0,10.0)",
)
@click.option(
    "--plot",
    is_flag=True,
    help="Generate and save plots",
)
def main(
    data_file,
    output_dir,
    constraint_violations,
    plot,
):
    """Run regularization path analysis"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data tensor
    data_tensor = np.load(data_file)

    # Extract first replicate as 2D matrix
    # New format: (n_replicates, D, n_samples) -> (D, n_samples)
    X = data_tensor[0, :, :]

    # Parse constraint violations
    try:
        cv_values = [float(x.strip()) for x in constraint_violations.split(",")]
    except ValueError:
        print("Error: constraint-violations must be comma-separated floats")
        return

    print(
        f"Running regularization path with {len(cv_values)} constraint violation values"
    )
    print(f"Constraint violations: {cv_values}")
    print(f"Data shape: {X.shape}")

    # Run isometry pursuit for each constraint violation
    results = []
    beta_norms = []
    cardinalities = []

    for i, cv in enumerate(cv_values):
        print(f"\nIteration {i+1}/{len(cv_values)}: constraint_violation = {cv}")

        try:
            result = isometry_pursuit(
                X,
                constraint_violation=cv,
            )

            # Calculate norms of each column (feature)
            beta_norm = np.linalg.norm(result["beta"], axis=0)
            cardinality = len(result["basis_pursuit_indices"])

            results.append(result)
            beta_norms.append(beta_norm)
            cardinalities.append(cardinality)

            print(f"  Cardinality: {cardinality}")
            print(f"  Selected indices: {result['basis_pursuit_indices']}")

        except Exception as e:
            print(f"  Error: {e}")
            results.append(None)
            beta_norms.append(None)
            cardinalities.append(None)

    # Convert to numpy arrays for easier handling
    beta_norms_array = np.array([bn for bn in beta_norms if bn is not None])
    cardinalities_array = np.array([c for c in cardinalities if c is not None])
    valid_cv_values = [
        cv for i, cv in enumerate(cv_values) if beta_norms[i] is not None
    ]

    print(f"\nRegularization path completed!")
    print(f"Valid constraint violations: {valid_cv_values}")
    print(f"Cardinalities: {cardinalities_array}")

    # Save results
    results_file = os.path.join(output_dir, "regularization_path_results.npy")
    np.save(
        results_file,
        {
            "constraint_violations": np.array(valid_cv_values),
            "beta_norms": beta_norms_array,
            "cardinalities": cardinalities_array,
            "data_shape": X.shape,
        },
    )
    print(f"Results saved to {results_file}")

    # Generate plots if requested
    if plot and len(valid_cv_values) > 0:
        print("Generating plots...")

        # Plot 1: Cardinality vs Constraint Violation
        plt.figure(figsize=(10, 6))
        plt.plot(valid_cv_values, cardinalities_array, "bo-", linewidth=2, markersize=8)
        plt.xlabel("Constraint Violation")
        plt.ylabel("Cardinality (Number of Selected Features)")
        plt.title("Regularization Path: Cardinality vs Constraint Violation")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        cardinality_plot_file = os.path.join(
            output_dir, "cardinality_vs_constraint_violation.png"
        )
        plt.savefig(cardinality_plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Cardinality plot saved to {cardinality_plot_file}")

        # Plot 2: Feature Norms vs Constraint Violation
        plt.figure(figsize=(12, 8))
        for i in range(beta_norms_array.shape[1]):
            plt.plot(
                valid_cv_values,
                beta_norms_array[:, i],
                "o-",
                label=f"Feature {i}",
                linewidth=1.5,
                markersize=4,
            )

        plt.xlabel("Constraint Violation")
        plt.ylabel("Feature Norm (||β_i||)")
        plt.title("Regularization Path: Feature Norms vs Constraint Violation")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        norms_plot_file = os.path.join(
            output_dir, "feature_norms_vs_constraint_violation.png"
        )
        plt.savefig(norms_plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Feature norms plot saved to {norms_plot_file}")

        # Plot 3: Heatmap of Beta Matrix
        if len(valid_cv_values) > 1:
            plt.figure(figsize=(10, 8))

            # Create a matrix where each row is a constraint violation and each column is a feature
            beta_matrix = np.abs(
                beta_norms_array.T
            )  # Transpose so features are columns

            plt.imshow(beta_matrix, aspect="auto", cmap="viridis")
            plt.colorbar(label="|Feature Norm|")
            plt.xlabel("Feature Index")
            plt.ylabel("Constraint Violation Index")
            plt.title("Regularization Path Heatmap: Feature Norms")

            # Set x-axis labels
            plt.xticks(range(beta_matrix.shape[1]))

            # Set y-axis labels
            plt.yticks(
                range(len(valid_cv_values)), [f"{cv:.1f}" for cv in valid_cv_values]
            )

            plt.tight_layout()

            heatmap_plot_file = os.path.join(
                output_dir, "regularization_path_heatmap.png"
            )
            plt.savefig(heatmap_plot_file, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Heatmap plot saved to {heatmap_plot_file}")

    print(f"\nRegularization path analysis completed!")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()

