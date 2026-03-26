#!/usr/bin/env python3
"""
Generate heatmaps of isometry_loss and group_lasso_norm over a 2D grid of
basis configurations: ratio of vector lengths (x-axis) x angle between vectors (y-axis).

The 2x2 matrix is constructed as:
  v1 = [1, 0]   (unit vector)
  v2 = [r*cos(theta), r*sin(theta)]  (length r, angle theta from v1)
  X = [v1 | v2]
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from isometry_pursuit.loss import isometry_loss, group_lasso_norm
from isometry_pursuit.algorithm import group_basis_pursuit
from isometry_pursuit.transformation import exponential_transformation


def compute_grids(n_ratio=40, n_angle=40, max_ratio=5.0):
    ratios = np.linspace(1.0, max_ratio, n_ratio)
    # Exclude endpoints 0 and pi to avoid singular matrices
    angles = np.linspace(0.05, np.pi - 0.05, n_angle)

    isometry_loss_grid = np.full((n_angle, n_ratio), np.nan)
    group_lasso_grid = np.full((n_angle, n_ratio), np.nan)

    total = n_ratio * n_angle
    with tqdm(total=total, desc="Computing grid") as pbar:
        for i, theta in enumerate(angles):
            for j, r in enumerate(ratios):
                # Build 2x2 matrix: columns are the two basis vectors
                X = np.array([[1.0, r * np.cos(theta)],
                               [0.0, r * np.sin(theta)]])

                # Isometry loss (applied to raw X)
                loss_val = isometry_loss(X)
                isometry_loss_grid[i, j] = loss_val if np.isfinite(loss_val) else np.nan

                # Group lasso norm after basis pursuit on transformed X
                try:
                    X_t = exponential_transformation(X)
                    beta, _, obj = group_basis_pursuit(X_t, constraint_violation=0.0)
                    group_lasso_grid[i, j] = obj
                except Exception:
                    group_lasso_grid[i, j] = np.nan

                pbar.update(1)

    return ratios, angles, isometry_loss_grid, group_lasso_grid


def plot_heatmaps(ratios, angles, isometry_loss_grid, group_lasso_grid, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    angle_labels = [r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"]
    angle_ticks = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, grid, title, fname in [
        (axes[0], isometry_loss_grid, "Isometry Loss", "basis_isometry_loss_heatmap.png"),
        (axes[1], group_lasso_grid, "Group Lasso Norm", "basis_group_lasso_heatmap.png"),
    ]:
        log_grid = np.log1p(grid)
        vmin = np.nanpercentile(log_grid, 1)
        vmax = np.nanpercentile(log_grid, 99)
        im = ax.pcolormesh(
            ratios, angles, log_grid,
            cmap="viridis", vmin=vmin, vmax=vmax, shading="auto"
        )
        plt.colorbar(im, ax=ax)
        ax.set_xlabel(r"Length ratio $r = \|v_2\| / \|v_1\|$", fontsize=13)
        ax.set_ylabel(r"Angle $\theta$ between vectors", fontsize=13)
        ax.set_title(title + " (log scale)", fontsize=15)
        ax.set_yticks(angle_ticks)
        ax.set_yticklabels(angle_labels, fontsize=11)

    plt.tight_layout()
    combined_path = os.path.join(output_dir, "basis_heatmaps.png")
    plt.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Combined heatmap saved to {combined_path}")

    # Also save individual figures
    for grid, title, fname in [
        (isometry_loss_grid, "Isometry Loss", "basis_isometry_loss_heatmap.png"),
        (group_lasso_grid, "Group Lasso Norm", "basis_group_lasso_heatmap.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6))
        log_grid = np.log1p(grid)
        vmin = np.nanpercentile(log_grid, 1)
        vmax = np.nanpercentile(log_grid, 99)
        im = ax.pcolormesh(
            ratios, angles, log_grid,
            cmap="viridis", vmin=vmin, vmax=vmax, shading="auto"
        )
        plt.colorbar(im, ax=ax)
        ax.set_xlabel(r"Length ratio $r = \|v_2\| / \|v_1\|$", fontsize=13)
        ax.set_ylabel(r"Angle $\theta$ between vectors", fontsize=13)
        ax.set_title(title + " (log scale)", fontsize=15)
        ax.set_yticks(angle_ticks)
        ax.set_yticklabels(angle_labels, fontsize=11)
        plt.tight_layout()
        out = os.path.join(output_dir, fname)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Heatmaps of isometry_loss and group_lasso_norm over 2D basis grid"
    )
    parser.add_argument("--output-dir", default="figures", help="Output directory")
    parser.add_argument("--n-ratio", type=int, default=40, help="Grid points along ratio axis")
    parser.add_argument("--n-angle", type=int, default=40, help="Grid points along angle axis")
    parser.add_argument("--max-ratio", type=float, default=5.0, help="Max length ratio")
    args = parser.parse_args()

    ratios, angles, isometry_loss_grid, group_lasso_grid = compute_grids(
        n_ratio=args.n_ratio, n_angle=args.n_angle, max_ratio=args.max_ratio
    )
    plot_heatmaps(ratios, angles, isometry_loss_grid, group_lasso_grid, args.output_dir)


if __name__ == "__main__":
    main()
