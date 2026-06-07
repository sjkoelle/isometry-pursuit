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


def normalize_vector(v):
    """Apply symmetric normalization q(v) * v where q(v) = (e^||v|| + e^{1/||v||}) / (2e)"""
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-14:
        return v
    q = (np.exp(norm_v) + np.exp(1.0 / norm_v)) / (2 * np.e)
    return q * v


def normalize_matrix(X):
    """Apply column-wise normalization w(X)"""
    X_normalized = X.copy()
    for i in range(X.shape[1]):
        X_normalized[:, i] = normalize_vector(X[:, i])
    return X_normalized


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

    # Apply normalization for relevant losses
    X_norm = normalize_matrix(X)

    # Frobenius norm: ||w(X)||_F
    losses['frobenius_norm'] = np.linalg.norm(X_norm, 'fro')

    # Isometry loss: g(r) where g(t) = (e^t + e^{1/t})/(2e)
    losses['isometry_loss'] = (np.exp(abs(r)) + np.exp(1.0/abs(r))) / (2 * np.e)

    # GL norm of pseudoinverse: ||w(X)^†||_{1,2}
    X_norm_pinv = np.linalg.pinv(X_norm)
    losses['gl_pseudoinverse'] = np.linalg.norm(X_norm_pinv, axis=1).sum()

    # Frobenius norm of pseudoinverse: ||w(X)^†||_F
    losses['frob_pseudoinverse'] = np.linalg.norm(X_norm_pinv, 'fro')

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

    # Apply normalization for relevant losses
    X_norm = normalize_matrix(X)

    # Frobenius norm: ||w(X)||_F
    losses['frobenius_norm'] = np.linalg.norm(X_norm, 'fro')

    # Isometry loss
    iso_loss = isometry_loss(X)
    losses['isometry_loss'] = iso_loss if np.isfinite(iso_loss) else np.nan

    # GL norm of pseudoinverse and Frobenius norm of pseudoinverse: use w(X)
    if sv_min > 1e-14:
        X_norm_pinv = np.linalg.pinv(X_norm)
        losses['gl_pseudoinverse'] = group_lasso_norm(X_norm_pinv)
        losses['frob_pseudoinverse'] = np.linalg.norm(X_norm_pinv, 'fro')
    else:
        losses['gl_pseudoinverse'] = np.nan
        losses['frob_pseudoinverse'] = np.nan

    # Log determinant
    det_val = sv_max * sv_min
    losses['log_det'] = np.log(det_val) if det_val > 1e-14 else np.nan

    return losses


def generate_1d_plots(output_dir="figures"):
    """Generate 1D plots for 1x1 matrix case."""
    os.makedirs(output_dir, exist_ok=True)

    print("Generating 1D plots (1x1 matrix)...")

    rs = np.linspace(0.3, 3.0, 200)

    # Define which losses have the good property (prefer r=1)
    good_losses = {'condition_number', 'gl_pseudoinverse', 'frob_pseudoinverse'}
    bad_losses = {'isometry_loss', 'log_det', 'nuclear_norm', 'frobenius_norm'}

    loss_configs = {
        'condition_number': {'color': 'C0', 'good': True},
        'gl_pseudoinverse': {'color': 'C1', 'good': True},
        'frob_pseudoinverse': {'color': 'C2', 'good': True},
        'isometry_loss': {'color': 'C3', 'good': False},
        'log_det': {'color': 'C4', 'good': False},
        'nuclear_norm': {'color': 'C5', 'good': False},
        'frobenius_norm': {'color': 'C6', 'good': False},
    }

    for loss_name, config in loss_configs.items():
        losses = [compute_1d_losses(r)[loss_name] for r in rs]
        losses = np.array(losses)

        fig, ax = plt.subplots(figsize=(10, 7))

        # Only plot finite values
        valid = np.isfinite(losses)
        ax.plot(rs[valid], losses[valid], linewidth=3, color=config['color'])

        # Mark the optimal point (r=1)
        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='r = 1 (optimal)')

        status = "✓ Good property" if config['good'] else "✗ No good property"
        ax.set_title(f"{loss_name} {status}", fontsize=16, weight='bold')
        ax.set_xlabel('Length r', fontsize=14)
        ax.set_ylabel('Loss value', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'loss_1d_{loss_name}.png'), dpi=300, bbox_inches='tight')
        print(f"  Saved loss_1d_{loss_name}.png")
        plt.close()


def generate_2d_heatmaps(output_dir="figures", n_ratio=60, n_angle=60, max_ratio=3.0):
    """Generate 2D heatmaps for 2x2 matrix case."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating 2D heatmaps ({n_ratio} x {n_angle} grid)...")

    ratios = np.linspace(0.5, max_ratio, n_ratio)
    angles = np.linspace(0.05, np.pi - 0.05, n_angle)

    # Define good and bad losses
    good_losses = {'condition_number', 'gl_pseudoinverse', 'frob_pseudoinverse'}
    bad_losses = {'isometry_loss', 'log_det', 'nuclear_norm', 'frobenius_norm'}
    all_loss_names = list(good_losses) + list(bad_losses)

    loss_configs = {
        'condition_number': {'good': True, 'cmap': 'viridis'},
        'gl_pseudoinverse': {'good': True, 'cmap': 'viridis'},
        'frob_pseudoinverse': {'good': True, 'cmap': 'viridis'},
        'isometry_loss': {'good': False, 'cmap': 'plasma'},
        'log_det': {'good': False, 'cmap': 'plasma'},
        'nuclear_norm': {'good': False, 'cmap': 'plasma'},
        'frobenius_norm': {'good': False, 'cmap': 'plasma'},
    }

    # Compute all grids
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

    # Generate one figure per loss
    for loss_name in all_loss_names:
        grid = all_grids[loss_name]
        grid_plot = grid.copy()
        grid_plot[~np.isfinite(grid_plot)] = np.nan

        fig, ax = plt.subplots(figsize=(10, 8))

        cmap = loss_configs[loss_name]['cmap']
        im = ax.contourf(ratios, angles, grid_plot, levels=20, cmap=cmap)
        cbar = plt.colorbar(im, ax=ax, label='Loss value')

        # Mark the ideal point: r=1 (unit-length), θ=π/2 (orthogonal)
        ax.plot(1.0, np.pi/2, 'r*', markersize=20, label='Ideal (r=1, θ=90°)')

        ax.set_xlabel('Length ratio r', fontsize=13)
        ax.set_ylabel('Angle θ (radians)', fontsize=13)

        status = "✓ Good property" if loss_configs[loss_name]['good'] else "✗ No good property"
        ax.set_title(f"{loss_name} {status}", fontsize=14, weight='bold')
        ax.legend(fontsize=11)

        plt.tight_layout()
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
    generate_1d_plots(args.output_dir)
    generate_2d_heatmaps(args.output_dir, args.n_ratio, args.n_angle)
    print("\nDone! Open figures/ to see loss comparisons.")


if __name__ == "__main__":
    main()
