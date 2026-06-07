#!/usr/bin/env python3
"""
Visualize loss function landscapes comparing candidate losses.

1D case: 1x1 matrix X = [r], single scalar parameter
2D case: 2x2 matrix with v1 = [1, 0], v2 = [r*cos(θ), r*sin(θ)]

Generates separate figures for each loss function showing:
- Which satisfy the property: "unit-length vectors always have lower loss"
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from isometry_pursuit.loss import isometry_loss, group_lasso_norm
from isometry_pursuit.transformation import exponential_transformation, g_broadcast_transformation


def compute_1d_losses(r):
    """
    Compute all losses for 1x1 matrix X = [r].
    These should match the 2D losses for consistency.
    """
    if r < 1e-14:
        return {key: np.nan for key in ['condition_number', 'gl_pseudoinverse',
                                          'frob_pseudoinverse', 'isometry_loss',
                                          'log_det', 'nuclear_norm', 'frobenius_norm']}

    # For 1x1 matrix, singular value is just |r|
    X = np.array([[r]])

    losses = {}

    # Condition number: for 1D, always 1 (only one singular value)
    losses['condition_number'] = 1.0

    # Nuclear norm: sum of singular values = r
    losses['nuclear_norm'] = abs(r)

    # Frobenius norm squared: 1/||w(X)||_F^2 (using standard transformation)
    X_w = exponential_transformation(X)
    frob_norm_w_squared = np.linalg.norm(X_w, 'fro') ** 2
    losses['frobenius_norm'] = 1.0 / frob_norm_w_squared if frob_norm_w_squared > 1e-14 else np.inf

    # Isometry loss: operates on raw X (already correct)
    iso_loss = isometry_loss(X)
    losses['isometry_loss'] = iso_loss if np.isfinite(iso_loss) else np.nan

    # Apply n_q transformation for pseudoinverse-based losses
    X_norm = exponential_transformation(X)

    # GL norm of pseudoinverse: ||n_q^{×P}(X)^†||_{1,2}
    X_norm_pinv = np.linalg.pinv(X_norm)
    losses['gl_pseudoinverse'] = np.linalg.norm(X_norm_pinv, axis=1).sum()

    # Frobenius norm squared of pseudoinverse: ||n_q^{×P}(X)^†||_F^2
    losses['frob_pseudoinverse'] = np.linalg.norm(X_norm_pinv, 'fro') ** 2

    # Log determinant: log(r)
    losses['log_det'] = np.log(abs(r))

    return losses


def compute_2d_losses(r, theta):
    """
    Compute all losses for 2x2 matrix:
      v1 = [1, 0]
      v2 = [r*cos(θ), r*sin(θ)]
    """
    X = np.array([[1.0, r * np.cos(theta)],
                   [0.0, r * np.sin(theta)]])

    U, svs, VT = np.linalg.svd(X)
    sv_max, sv_min = svs[0], svs[1]

    losses = {}

    # Condition number
    losses['condition_number'] = sv_max / sv_min if sv_min > 1e-14 else np.inf

    # Nuclear norm
    losses['nuclear_norm'] = sv_max + sv_min

    # Frobenius norm squared: 1/||w(X)||_F^2 (using standard transformation)
    X_w = exponential_transformation(X)
    frob_norm_w_squared = np.linalg.norm(X_w, 'fro') ** 2
    losses['frobenius_norm'] = 1.0 / frob_norm_w_squared if frob_norm_w_squared > 1e-14 else np.inf

    # Isometry loss
    iso_loss = isometry_loss(X)
    losses['isometry_loss'] = iso_loss if np.isfinite(iso_loss) else np.nan

    # Apply n_q transformation for pseudoinverse-based losses
    X_norm = exponential_transformation(X)

    # GL norm of pseudoinverse and Frobenius norm squared of pseudoinverse: use n_q^{×P}(X)
    if sv_min > 1e-14:
        X_norm_pinv = np.linalg.pinv(X_norm)
        losses['gl_pseudoinverse'] = group_lasso_norm(X_norm_pinv)
        losses['frob_pseudoinverse'] = np.linalg.norm(X_norm_pinv, 'fro') ** 2
    else:
        losses['gl_pseudoinverse'] = np.nan
        losses['frob_pseudoinverse'] = np.nan

    # Log determinant
    det_val = sv_max * sv_min
    losses['log_det'] = np.log(det_val) if det_val > 1e-14 else np.nan

    return losses


def generate_combined_plots(output_dir="figures", n_ratio=60, n_angle=60, max_ratio=3.0):
    """Generate combined figures with 1D plots on top and 2D heatmaps below."""
    os.makedirs(output_dir, exist_ok=True)

    print("Generating combined loss landscape visualizations...")

    # 1D setup
    rs = np.linspace(0.3, 3.0, 200)

    # 2D setup
    ratios = np.linspace(0.5, max_ratio, n_ratio)
    angles = np.linspace(0.05, np.pi - 0.05, n_angle)

    # Define losses (exclude nuclear_norm, keep 6)
    # Order matches table: Distortion, Log det, Isometry loss, Frob. norm, Frob. pseudoinv., Isometry pursuit
    all_loss_names = ['condition_number', 'log_det', 'isometry_loss', 'frobenius_norm', 'frob_pseudoinverse', 'gl_pseudoinverse']

    loss_configs = {
        'condition_number': {'color': 'C0', 'good': True, 'cmap': 'viridis'},
        'gl_pseudoinverse': {'color': 'C1', 'good': True, 'cmap': 'viridis'},
        'frob_pseudoinverse': {'color': 'C2', 'good': True, 'cmap': 'viridis'},
        'isometry_loss': {'color': 'C3', 'good': False, 'cmap': 'plasma'},
        'log_det': {'color': 'C4', 'good': False, 'cmap': 'plasma'},
        'frobenius_norm': {'color': 'C5', 'good': False, 'cmap': 'plasma'},
    }

    # Compute all 2D grids first
    print(f"Computing 2D grids ({n_ratio} x {n_angle} resolution)...")
    all_grids = {key: np.full((n_angle, n_ratio), np.nan) for key in all_loss_names}

    from tqdm import tqdm
    total = n_ratio * n_angle
    with tqdm(total=total, desc="Computing grids") as pbar:
        for i, theta in enumerate(angles):
            for j, r in enumerate(ratios):
                losses = compute_2d_losses(r, theta)
                for key in all_loss_names:
                    if key in losses:
                        all_grids[key][i, j] = losses[key]
                pbar.update(1)

    # Generate separate 1D and 2D plots for each loss (same height for alignment)
    print("Generating 1D and 2D plots...")

    # Fixed size for both 1D and 2D plots: width=6, height=2/3*width=4
    plot_width = 6
    plot_height = 4

    for loss_name in all_loss_names:
        # Compute 1D losses
        losses_1d = [compute_1d_losses(r)[loss_name] for r in rs]
        losses_1d = np.array(losses_1d)

        config = loss_configs[loss_name]
        status = "✓ Good property" if config['good'] else "✗ No good property"

        # 1D plot
        fig, ax = plt.subplots(figsize=(plot_width, plot_height))
        valid = np.isfinite(losses_1d)
        ax.plot(rs[valid], losses_1d[valid], linewidth=3, color=config['color'])
        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='r = 1 (optimal)')
        ax.set_xlabel('Length r', fontsize=11)
        ax.set_ylabel('Loss value', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        fig.savefig(os.path.join(output_dir, f'loss_1d_{loss_name}.png'), dpi=300, bbox_inches='tight')
        print(f"  Saved loss_1d_{loss_name}.png")
        plt.close()

        # 2D heatmap
        fig, ax = plt.subplots(figsize=(plot_width, plot_height))
        grid = all_grids[loss_name]
        grid_plot = grid.copy()
        grid_plot[~np.isfinite(grid_plot)] = np.nan

        # For isometry_loss with extreme values, clip to 95th percentile for better visualization
        if loss_name == 'isometry_loss':
            finite_values = grid_plot[np.isfinite(grid_plot)]
            if len(finite_values) > 0:
                clip_value = np.percentile(finite_values, 95)
                grid_plot = np.clip(grid_plot, None, clip_value)

        cmap = config['cmap']
        im = ax.contourf(ratios, angles, grid_plot, levels=20, cmap=cmap)
        cbar = plt.colorbar(im, ax=ax, label='Loss value')
        ax.plot(1.0, np.pi/2, 'r*', markersize=20, label='Ideal (r=1, θ=90°)')
        ax.set_xlabel('Length ratio r', fontsize=11)
        ax.set_ylabel('Angle θ (radians)', fontsize=11)
        ax.legend(fontsize=10)
        fig.savefig(os.path.join(output_dir, f'loss_2d_{loss_name}.png'), dpi=300, bbox_inches='tight')
        print(f"  Saved loss_2d_{loss_name}.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize loss function landscapes")
    parser.add_argument("--output-dir", default="figures", help="Output directory")
    parser.add_argument("--n-ratio", type=int, default=60, help="2D grid resolution (ratio)")
    parser.add_argument("--n-angle", type=int, default=60, help="2D grid resolution (angle)")

    args = parser.parse_args()

    print("Generating loss landscape visualizations...\n")
    generate_combined_plots(args.output_dir, args.n_ratio, args.n_angle)
    print("\nDone! Open figures/ to see loss comparisons.")


if __name__ == "__main__":
    main()
