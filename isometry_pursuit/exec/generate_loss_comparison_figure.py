#!/usr/bin/env python3
"""
Generate a comparison figure for loss functions showing good vs bad choices.
This creates a composite figure suitable for the paper.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from isometry_pursuit.loss import group_lasso_norm
from isometry_pursuit.transformation import exponential_transformation


def compute_2d_losses(r, theta):
    """Compute losses for 2x2 matrix."""
    X = np.array([[1.0, r * np.cos(theta)],
                   [0.0, r * np.sin(theta)]])

    U, svs, VT = np.linalg.svd(X)
    sv_max, sv_min = svs[0], svs[1]

    losses = {}
    losses['condition_number'] = sv_max / sv_min if sv_min > 1e-14 else np.inf
    X_norm = exponential_transformation(X)
    losses['gl_pseudoinverse'] = group_lasso_norm(np.linalg.pinv(X_norm)) if sv_min > 1e-14 else np.nan
    losses['log_det'] = np.log(sv_max * sv_min) if sv_max * sv_min > 1e-14 else np.nan

    return losses


def generate_comparison_figure(output_dir="figures", n_ratio=80, n_angle=80):
    """Generate a figure comparing good vs bad losses."""
    os.makedirs(output_dir, exist_ok=True)

    ratios = np.linspace(0.5, 3.0, n_ratio)
    angles = np.linspace(0.05, np.pi - 0.05, n_angle)

    # Compute grids
    grids = {
        'gl_pseudoinverse': np.full((n_angle, n_ratio), np.nan),
        'condition_number': np.full((n_angle, n_ratio), np.nan),
        'log_det': np.full((n_angle, n_ratio), np.nan),
    }

    from tqdm import tqdm
    total = n_ratio * n_angle
    with tqdm(total=total, desc="Computing comparison grid") as pbar:
        for i, theta in enumerate(angles):
            for j, r in enumerate(ratios):
                losses = compute_2d_losses(r, theta)
                for key in grids:
                    if key in losses:
                        grids[key][i, j] = losses[key]
                pbar.update(1)

    # Create figure: one row comparing remaining losses.
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    grid = grids['gl_pseudoinverse']
    grid_plot = grid.copy()
    grid_plot[~np.isfinite(grid_plot)] = np.nan
    im = ax.contourf(ratios, angles, grid_plot, levels=20, cmap='viridis')
    ax.plot(1.0, np.pi/2, 'r*', markersize=25, markeredgewidth=2, markeredgecolor='white')
    ax.set_xlabel('Length ratio $r$', fontsize=12)
    ax.set_ylabel('Angle $\\theta$', fontsize=12)
    ax.set_title('GL pseudoinverse norm', fontsize=13, weight='bold')
    plt.colorbar(im, ax=ax)

    ax = axes[1]
    grid = grids['condition_number']
    grid_plot = grid.copy()
    grid_plot[~np.isfinite(grid_plot)] = np.nan
    im = ax.contourf(ratios, angles, grid_plot, levels=20, cmap='viridis')
    ax.plot(1.0, np.pi/2, 'r*', markersize=25, markeredgewidth=2, markeredgecolor='white')
    ax.set_xlabel('Length ratio $r$', fontsize=12)
    ax.set_ylabel('Angle $\\theta$', fontsize=12)
    ax.set_title('Condition number', fontsize=13, weight='bold')
    plt.colorbar(im, ax=ax)

    ax = axes[2]
    grid = grids['log_det']
    grid_plot = grid.copy()
    grid_plot[~np.isfinite(grid_plot)] = np.nan
    im = ax.contourf(ratios, angles, grid_plot, levels=20, cmap='plasma')
    ax.plot(1.0, np.pi/2, 'c*', markersize=25, markeredgewidth=2, markeredgecolor='white')
    ax.set_xlabel('Length ratio $r$', fontsize=12)
    ax.set_ylabel('Angle $\\theta$', fontsize=12)
    ax.set_title('Log determinant', fontsize=13, weight='bold')
    plt.colorbar(im, ax=ax)

    plt.suptitle('Loss Function Comparison', fontsize=14, weight='bold', y=0.995)
    plt.tight_layout()

    fig.savefig(os.path.join(output_dir, 'loss_comparison_good_vs_bad.png'), dpi=300, bbox_inches='tight')
    print(f"Saved to {output_dir}/loss_comparison_good_vs_bad.png")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate loss comparison figure")
    parser.add_argument("--output-dir", default="figures", help="Output directory")
    args = parser.parse_args()

    generate_comparison_figure(args.output_dir)


if __name__ == "__main__":
    main()
