#!/usr/bin/env python3
"""
Generate Figure 1: Transformation and Loss Functions
This script creates the plots showing the exponential transformation and loss functions.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange

from isometry_pursuit.transformation import exponential_transformation
from isometry_pursuit.loss import isometry_loss


def generate_figure1(output_dir="figures"):
    """
    Generate Figure 1 plots showing transformation and loss functions.

    Parameters:
    -----------
    output_dir : str
        Directory to save the plots
    """

    print("Generating Figure 1: Transformation and Loss Functions")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Matrix values and rearrangement
    matrix_values = np.logspace(-2, 2, 100)
    rearranged_matrix_values = rearrange(matrix_values, "n -> n 1 1")

    # Use c = 1 only
    power = 1.0

    # Compute transformations and losses
    lengths = [
        exponential_transformation(rmv, power)[0][0] for rmv in rearranged_matrix_values
    ]
    losses_isometry = [isometry_loss(rmv, power) for rmv in rearranged_matrix_values]
    basis_pursuit_losses = np.asarray(lengths) ** (-1)

    line_style = "-"  # single style since only one curve

    # --- Plot the function: Length ---
    print("Creating length transformation plot...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.plot(matrix_values, lengths, color="black", linewidth=2.5, linestyle=line_style)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    ax.set_title(r"$q(t)$", fontsize=80)
    ax.set_ylabel(r"$\frac{2e}{e^{t} + e^{\frac{1}{t}}}$", fontsize=60)
    ax.set_xlabel(r"$t$", fontsize=50)
    ax.set_xscale("log")
    ax.set_xlim(1e-2, 1e2)
    ax.tick_params(labelsize=30, which="both")

    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "Figure_1b_bw.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(f"Length plot saved to {output_dir}/Figure_1b_bw.png")

    # --- Plot the function: Loss ---
    print("Creating loss function plot...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.plot(
        matrix_values,
        basis_pursuit_losses,
        color="black",
        linewidth=2.5,
        linestyle=line_style,
    )

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    ax.set_title(r"$\ell(t)$", fontsize=80)
    ax.set_ylim(0, 10)
    ax.set_ylabel(r"$\frac{e^{t} + e^{\frac{1}{t}}}{2e} - 1$", fontsize=60)
    ax.set_xlabel(r"$t$", fontsize=50)
    ax.set_xscale("log")
    ax.set_xlim(1e-2, 1e2)
    ax.tick_params(labelsize=30, which="both")

    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "Figure_1a_bw.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(f"Loss plot saved to {output_dir}/Figure_1a_bw.png")

    # --- Plot the function: Basis pursuit loss interpretation ---
    print("Creating basis pursuit loss interpretation plot...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.plot(
        matrix_values,
        basis_pursuit_losses,
        color="black",
        linewidth=2.5,
        linestyle=line_style,
    )

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    ax.set_title(r"$\|\beta\|_{1,2} \; : \; 1 = w(t)\,\beta$", fontsize=80)
    ax.set_ylim(0, 10)
    ax.set_ylabel(r"$\frac{e^{t} + e^{\frac{1}{t}}}{2e}$", fontsize=60)
    ax.set_xlabel(r"$t$", fontsize=50)
    ax.set_xscale("log")
    ax.set_xlim(1e-2, 1e2)
    ax.tick_params(labelsize=30, which="both")

    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "Figure_1c_bw.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(
        f"Basis pursuit loss interpretation plot saved to {output_dir}/Figure_1c_bw.png"
    )

    print("Figure 1 generation completed!")


def main():
    parser = argparse.ArgumentParser(description="Generate Figure 1 plots")
    parser.add_argument(
        "--output-dir", default="figures", help="Output directory for plots"
    )

    args = parser.parse_args()

    generate_figure1(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
