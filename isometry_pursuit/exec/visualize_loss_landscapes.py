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
from matplotlib.colors import SymLogNorm

from isometry_pursuit.loss import isometry_loss, group_lasso_norm
from isometry_pursuit.transformation import exponential_transformation


def s_divergence_to_identity(X):
    """S-divergence between X^T X and the identity."""
    gram = X.T @ X
    sign_gram, logdet_gram = np.linalg.slogdet(gram)
    if sign_gram <= 0:
        return np.nan

    identity = np.eye(gram.shape[0])
    sign_mid, logdet_mid = np.linalg.slogdet((gram + identity) / 2.0)
    if sign_mid <= 0:
        return np.nan

    return logdet_mid - 0.5 * logdet_gram


def quadratic_log_singular_value_sum(X):
    """Sum of g(sigma_d) for g(t) = 2t^2 - log(t)."""
    singular_values = np.linalg.svd(X, compute_uv=False)
    if np.any(singular_values <= 1e-14):
        return np.nan

    return np.sum(2.0 * singular_values ** 2 - np.log(singular_values))


def normalized_logdet_volume(X):
    """Negative log singular-value volume after n_q column normalization."""
    X_norm = exponential_transformation(X)
    singular_values = np.linalg.svd(X_norm, compute_uv=False)
    if np.any(singular_values <= 1e-14):
        return np.nan
    return -np.sum(np.log(singular_values))


def squared_log_singular_value_sum(X):
    """Sum of squared log singular values."""
    singular_values = np.linalg.svd(X, compute_uv=False)
    if np.any(singular_values <= 1e-14):
        return np.nan

    return np.sum(np.log(singular_values) ** 2)


def absolute_log_singular_value_sum(X):
    """Sum of absolute log singular values."""
    singular_values = np.linalg.svd(X, compute_uv=False)
    if np.any(singular_values <= 1e-14):
        return np.nan

    return np.sum(np.abs(np.log(singular_values)))


def compute_1d_losses(r):
    """
    Compute all losses for 1x1 matrix X = [r].
    These should match the 2D losses for consistency.
    """
    if r < 1e-14:
        return {key: np.nan for key in ['condition_number', 'gl_pseudoinverse',
                                          'frob_pseudoinverse', 'isometry_loss',
                                          'nuclear_norm',
                                          's_divergence_identity',
                                          'quadratic_log_singular_values',
                                          'log_singular_values',
                                          'absolute_log_singular_values',
                                          'normalized_logdet_volume']}

    # For 1x1 matrix, singular value is just |r|
    X = np.array([[r]])

    losses = {}

    d = X.shape[0]

    # Condition number: for 1D, always 1 (only one singular value)
    losses['condition_number'] = 1.0 - 1.0

    # Nuclear norm: sum of singular values = r
    losses['nuclear_norm'] = abs(r)

    # Isometry loss: operates on raw X (already correct)
    iso_loss = isometry_loss(X)
    losses['isometry_loss'] = iso_loss - d if np.isfinite(iso_loss) else np.nan

    # Apply w^{×P} transformation for pseudoinverse-based losses
    X_norm = exponential_transformation(X)

    # GL norm of pseudoinverse: ||w^{×P}(X)^†||_{1,2}
    X_norm_pinv = np.linalg.pinv(X_norm)
    losses['gl_pseudoinverse'] = np.linalg.norm(X_norm_pinv, axis=1).sum() - d

    # Frobenius norm of pseudoinverse: ||n_q^{×P}(X)^†||_F
    losses['frob_pseudoinverse'] = np.linalg.norm(X_norm_pinv, 'fro') - np.sqrt(d)

    losses['s_divergence_identity'] = s_divergence_to_identity(X)
    losses['quadratic_log_singular_values'] = quadratic_log_singular_value_sum(X)
    losses['log_singular_values'] = squared_log_singular_value_sum(X)
    losses['absolute_log_singular_values'] = absolute_log_singular_value_sum(X)
    losses['normalized_logdet_volume'] = normalized_logdet_volume(X)

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

    d = X.shape[0]

    # Condition number
    losses['condition_number'] = sv_max / sv_min - 1.0 if sv_min > 1e-14 else np.inf

    # Nuclear norm
    losses['nuclear_norm'] = sv_max + sv_min

    # Isometry loss
    iso_loss = isometry_loss(X)
    losses['isometry_loss'] = iso_loss - d if np.isfinite(iso_loss) else np.nan

    # Apply w^{×P} transformation for pseudoinverse-based losses
    X_norm = exponential_transformation(X)

    # GL norm of pseudoinverse and Frobenius norm of pseudoinverse: use n_q^{×P}(X)
    if sv_min > 1e-14:
        X_norm_pinv = np.linalg.pinv(X_norm)
        losses['gl_pseudoinverse'] = group_lasso_norm(X_norm_pinv) - d
        losses['frob_pseudoinverse'] = np.linalg.norm(X_norm_pinv, 'fro') - np.sqrt(d)
    else:
        losses['gl_pseudoinverse'] = np.nan
        losses['frob_pseudoinverse'] = np.nan

    losses['s_divergence_identity'] = s_divergence_to_identity(X)
    losses['quadratic_log_singular_values'] = quadratic_log_singular_value_sum(X)
    losses['log_singular_values'] = squared_log_singular_value_sum(X)
    losses['absolute_log_singular_values'] = absolute_log_singular_value_sum(X)
    losses['normalized_logdet_volume'] = normalized_logdet_volume(X)

    return losses


def compute_3d_equicorrelation_losses(r, rho):
    """
    Compute all losses for 3 vectors in R^3 with lengths (1, r, 1)
    and pairwise correlations equal to rho.
    """
    lengths = np.array([1.0, r, 1.0])
    corr = np.full((3, 3), rho)
    np.fill_diagonal(corr, 1.0)
    gram = np.diag(lengths) @ corr @ np.diag(lengths)

    try:
        X = np.linalg.cholesky(gram).T
    except np.linalg.LinAlgError:
        return {key: np.nan for key in ['condition_number', 'gl_pseudoinverse',
                                          'frob_pseudoinverse', 'isometry_loss',
                                          'nuclear_norm',
                                          's_divergence_identity',
                                          'quadratic_log_singular_values',
                                          'log_singular_values',
                                          'absolute_log_singular_values',
                                          'normalized_logdet_volume']}

    U, svs, VT = np.linalg.svd(X)
    sv_max, sv_min = svs[0], svs[-1]

    losses = {}
    d = X.shape[0]

    losses['condition_number'] = sv_max / sv_min - 1.0 if sv_min > 1e-14 else np.inf
    losses['nuclear_norm'] = np.sum(svs)

    iso_loss = isometry_loss(X)
    losses['isometry_loss'] = iso_loss - d if np.isfinite(iso_loss) else np.nan

    X_norm = exponential_transformation(X)
    if sv_min > 1e-14:
        X_norm_pinv = np.linalg.pinv(X_norm)
        losses['gl_pseudoinverse'] = group_lasso_norm(X_norm_pinv) - d
        losses['frob_pseudoinverse'] = np.linalg.norm(X_norm_pinv, 'fro') - np.sqrt(d)
    else:
        losses['gl_pseudoinverse'] = np.nan
        losses['frob_pseudoinverse'] = np.nan

    losses['s_divergence_identity'] = s_divergence_to_identity(X)
    losses['quadratic_log_singular_values'] = quadratic_log_singular_value_sum(X)
    losses['log_singular_values'] = squared_log_singular_value_sum(X)
    losses['absolute_log_singular_values'] = absolute_log_singular_value_sum(X)
    losses['normalized_logdet_volume'] = normalized_logdet_volume(X)

    return losses


def generate_combined_plots(output_dir="figures", n_ratio=60, n_angle=60, max_ratio=3.0):
    """Generate combined figures with 1D plots on top and 2D heatmaps below."""
    os.makedirs(output_dir, exist_ok=True)

    print("Generating combined loss landscape visualizations...")

    # 1D setup. Sample geometrically on both sides of the optimum so the
    # sampling density matches the symlog x-axis.
    rs = np.unique(
        np.concatenate((
            np.geomspace(1e-2, 1.0, 500, endpoint=False),
            np.geomspace(1.0, 1e2, 500),
        ))
    )

    # 2D setup
    ratios = np.linspace(0.5, max_ratio, n_ratio)
    angles = np.linspace(0.05, np.pi - 0.05, n_angle)

    # Define losses (exclude nuclear_norm)
    # Order matches table.
    all_loss_names = [
        'condition_number',
        'isometry_loss',
        's_divergence_identity',
        'quadratic_log_singular_values',
        'log_singular_values',
        'absolute_log_singular_values',
        'normalized_logdet_volume',
        'frob_pseudoinverse',
        'gl_pseudoinverse',
    ]
    combined_1d_loss_names = [
        'condition_number',
        'isometry_loss',
        's_divergence_identity',
        'absolute_log_singular_values',
        'normalized_logdet_volume',
        'frob_pseudoinverse',
        'gl_pseudoinverse',
    ]

    loss_configs = {
        'condition_number': {
            'color': 'C0',
            'good': True,
            'cmap': 'viridis',
            'title': 'Distortion',
            'legend_label': r'Distortion: $\sigma_{\max}([r])/\sigma_{\min}([r]) - 1 = 0$',
            'ylabel_1d': r'$\sigma_{\max}([r])/\sigma_{\min}([r]) - 1 = 0$',
        },
        'gl_pseudoinverse': {
            'color': 'C1',
            'good': True,
            'cmap': 'viridis',
            'title': 'Isometry pursuit',
            'legend_label': r'Isometry pursuit: $q(r)^{-1} - 1 = \frac{e^r + e^{r^{-1}}}{2e} - 1$',
            'ylabel_1d': r'$q(r)^{-1} - 1 = \frac{e^r + e^{r^{-1}}}{2e} - 1$',
        },
        'frob_pseudoinverse': {
            'color': 'C2',
            'good': True,
            'cmap': 'viridis',
            'title': 'Frob. pseudoinv.',
            'legend_label': r'Frob. pseudoinv.: $q(r)^{-1} - 1 = \frac{e^r + e^{r^{-1}}}{2e} - 1$',
            'ylabel_1d': r'$q(r)^{-1} - 1 = \frac{e^r + e^{r^{-1}}}{2e} - 1$',
        },
        'isometry_loss': {
            'color': 'C3',
            'good': False,
            'cmap': 'plasma',
            'title': 'Isometry loss',
            'legend_label': r'Isometry loss: $g(r) - 1 = \frac{e^r + e^{r^{-1}}}{2e} - 1$',
            'ylabel_1d': r'$g(r) - 1 = \frac{e^r + e^{r^{-1}}}{2e} - 1$',
        },
        's_divergence_identity': {
            'color': 'C6',
            'good': True,
            'cmap': 'viridis',
            'title': 'S-divergence',
            'legend_label': r'S-divergence: $D(r^2,1)^2 = \log \frac{r + r^{-1}}{2}$',
            'ylabel_1d': r'$D(r^2,1)^2 = \log \frac{r + r^{-1}}{2}$',
        },
        'quadratic_log_singular_values': {
            'color': 'C8',
            'good': False,
            'cmap': 'plasma',
            'title': 'Quadratic-log',
            'legend_label': r'Quadratic-log: $\sum_d (2\sigma_d(X)^2 - \log\sigma_d(X))$',
            'ylabel_1d': r'$2r^2 - \log r$',
        },
        'normalized_logdet_volume': {
            'color': 'C7',
            'good': False,
            'cmap': 'plasma',
            'title': 'Norm. log-det',
            'legend_label': r'Norm. log-det: $-\log q(r) = \log \frac{e^r + e^{r^{-1}}}{2e}$',
            'ylabel_1d': r'$-\log q(r) = \log \frac{e^r + e^{r^{-1}}}{2e}$',
        },
        'log_singular_values': {
            'color': 'C4',
            'good': True,
            'cmap': 'viridis',
            'title': 'Squared log singular',
            'legend_label': r'Squared log singular: $\sum_d (\log\sigma_d(X))^2$',
            'ylabel_1d': r'$(\log r)^2$',
        },
        'absolute_log_singular_values': {
            'color': 'C5',
            'good': True,
            'cmap': 'viridis',
            'title': 'Abs. log singular',
            'legend_label': r'Abs. log singular: $\sum_d |\log\sigma_d(X)|$',
            'ylabel_1d': r'$|\log r|$',
        },
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

    # Generate one combined 1D plot and separate 2D plots for each loss
    print("Generating 1D and 2D plots...")

    plot_width = 6
    plot_height = 4
    shared_2d_ticks = [0, 1, 10, 100, 1000]
    shared_2d_levels = np.unique(
        np.concatenate(([0], np.linspace(0.05, 1, 10), np.logspace(0, 3, 32)))
    )
    shared_2d_norm = SymLogNorm(linthresh=1, vmin=0, vmax=1000, base=10)
    shared_2d_cmap = 'viridis'

    all_1d_losses = {
        loss_name: np.array([compute_1d_losses(r)[loss_name] for r in rs])
        for loss_name in combined_1d_loss_names
    }

    # Combined 1D plot
    fig, ax = plt.subplots(figsize=(9, 5))
    for loss_name in combined_1d_loss_names:
        config = loss_configs[loss_name]
        losses_1d = all_1d_losses[loss_name]
        valid = np.isfinite(losses_1d)
        ax.plot(
            rs[valid],
            losses_1d[valid],
            linewidth=2.5,
            color=config['color'],
            label=config['legend_label'],
        )
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label=r'$r=1$')
    ax.set_xscale('log')
    ax.set_xlim(rs.min(), rs.max())
    ax.set_ylim(-0.05, 8.0)
    ax.set_title('1D loss comparison', fontsize=13, weight='bold')
    ax.set_xlabel('Length r', fontsize=11)
    ax.set_ylabel('Loss value', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.savefig(os.path.join(output_dir, 'loss_1d_all.png'), dpi=300, bbox_inches='tight')
    print("  Saved loss_1d_all.png")
    plt.close()

    for loss_name in all_loss_names:
        config = loss_configs[loss_name]

        # 2D heatmap.
        fig, ax = plt.subplots(figsize=(plot_width, plot_height))
        grid = all_grids[loss_name]
        grid_plot = grid.copy()
        grid_plot[~np.isfinite(grid_plot)] = np.nan

        grid_plot = np.clip(grid_plot, 0, 1000)
        im = ax.contourf(
            ratios,
            angles,
            grid_plot,
            levels=shared_2d_levels,
            cmap=shared_2d_cmap,
            norm=shared_2d_norm,
        )
        cbar = plt.colorbar(im, ax=ax, label='Loss value')
        cbar.set_ticks(shared_2d_ticks)
        cbar.set_ticklabels([str(tick) for tick in shared_2d_ticks])
        ax.plot(1.0, np.pi/2, 'r*', markersize=20, label='Ideal (r=1, θ=90°)')
        ax.set_title(config['title'], fontsize=13, weight='bold')
        ax.set_xlabel('Length ratio r', fontsize=11)
        ax.set_ylabel('Angle θ (radians)', fontsize=11)
        ax.legend(fontsize=10)
        fig.savefig(os.path.join(output_dir, f'loss_2d_{loss_name}.png'), dpi=300, bbox_inches='tight')
        print(f"  Saved loss_2d_{loss_name}.png")
        plt.close()


def generate_3d_equicorrelation_plots(output_dir="figures", n_ratio=60, n_rho=60, max_ratio=3.0):
    """Generate 3D equicorrelation heatmaps over length ratio and rho."""
    os.makedirs(output_dir, exist_ok=True)

    print("Generating 3D equicorrelation loss visualizations...")

    ratios = np.linspace(0.5, max_ratio, n_ratio)
    rhos = np.linspace(-0.49, 0.99, n_rho)

    loss_names = [
        'condition_number',
        'isometry_loss',
        's_divergence_identity',
        'quadratic_log_singular_values',
        'log_singular_values',
        'absolute_log_singular_values',
        'normalized_logdet_volume',
        'frob_pseudoinverse',
        'gl_pseudoinverse',
    ]
    loss_titles = {
        'condition_number': 'Distortion',
        'isometry_loss': 'Isometry loss',
        's_divergence_identity': 'S-divergence',
        'quadratic_log_singular_values': 'Quadratic-log',
        'log_singular_values': 'Squared log singular',
        'absolute_log_singular_values': 'Abs. log singular',
        'normalized_logdet_volume': 'Norm. log-det',
        'frob_pseudoinverse': 'Frob. pseudoinv.',
        'gl_pseudoinverse': 'Isometry pursuit',
    }

    print(f"Computing 3D equicorrelation grids ({n_ratio} x {n_rho} resolution)...")
    all_grids = {key: np.full((n_rho, n_ratio), np.nan) for key in loss_names}

    from tqdm import tqdm
    total = n_ratio * n_rho
    with tqdm(total=total, desc="Computing 3D grids") as pbar:
        for i, rho in enumerate(rhos):
            for j, r in enumerate(ratios):
                losses = compute_3d_equicorrelation_losses(r, rho)
                for key in loss_names:
                    all_grids[key][i, j] = losses[key]
                pbar.update(1)

    shared_ticks = [0, 1, 10, 100, 1000]
    shared_levels = np.unique(
        np.concatenate(([0], np.linspace(0.05, 1, 10), np.logspace(0, 3, 32)))
    )
    shared_norm = SymLogNorm(linthresh=1, vmin=0, vmax=1000, base=10)
    shared_cmap = 'viridis'

    print("Generating 3D equicorrelation heatmaps...")
    for loss_name in loss_names:
        fig, ax = plt.subplots(figsize=(6, 4))
        grid_plot = all_grids[loss_name].copy()
        grid_plot[~np.isfinite(grid_plot)] = np.nan
        grid_plot = np.clip(grid_plot, 0, 1000)

        im = ax.contourf(
            ratios,
            rhos,
            grid_plot,
            levels=shared_levels,
            cmap=shared_cmap,
            norm=shared_norm,
        )
        cbar = plt.colorbar(im, ax=ax, label='Loss value')
        cbar.set_ticks(shared_ticks)
        cbar.set_ticklabels([str(tick) for tick in shared_ticks])
        ax.plot(1.0, 0.0, 'r*', markersize=20, label=r'Ideal $(r=1, \rho=0)$')
        ax.set_title(f"3D {loss_titles[loss_name]}", fontsize=13, weight='bold')
        ax.set_xlabel('Length ratio r = ||v2|| / ||v1||', fontsize=11)
        ax.set_ylabel(r'Pairwise correlation $\rho$', fontsize=11)
        ax.legend(fontsize=10)
        fig.savefig(
            os.path.join(output_dir, f'loss_3d_equicorr_{loss_name}.png'),
            dpi=300,
            bbox_inches='tight',
        )
        print(f"  Saved loss_3d_equicorr_{loss_name}.png")
        plt.close()

    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)
    for ax, loss_name in zip(axes.ravel(), loss_names):
        grid_plot = all_grids[loss_name].copy()
        grid_plot[~np.isfinite(grid_plot)] = np.nan
        grid_plot = np.clip(grid_plot, 0, 1000)
        im = ax.contourf(
            ratios,
            rhos,
            grid_plot,
            levels=shared_levels,
            cmap=shared_cmap,
            norm=shared_norm,
        )
        ax.plot(1.0, 0.0, 'r*', markersize=12)
        ax.set_title(loss_titles[loss_name], fontsize=11, weight='bold')

    for ax in axes[-1, :]:
        ax.set_xlabel('Length ratio r')
    for ax in axes[:, 0]:
        ax.set_ylabel(r'Correlation $\rho$')

    fig.suptitle('3D equicorrelation loss landscapes', fontsize=15, weight='bold')
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), label='Loss value', shrink=0.9)
    cbar.set_ticks(shared_ticks)
    cbar.set_ticklabels([str(tick) for tick in shared_ticks])
    fig.savefig(
        os.path.join(output_dir, 'loss_3d_equicorr_all.png'),
        dpi=300,
        bbox_inches='tight',
    )
    print("  Saved loss_3d_equicorr_all.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize loss function landscapes")
    parser.add_argument("--output-dir", default="figures", help="Output directory")
    parser.add_argument("--n-ratio", type=int, default=60, help="2D grid resolution (ratio)")
    parser.add_argument("--n-angle", type=int, default=60, help="2D grid resolution (angle)")
    parser.add_argument("--n-rho", type=int, default=60, help="3D equicorrelation grid resolution (rho)")
    parser.add_argument("--skip-3d-equicorr", action="store_true", help="Skip 3D equicorrelation plots")

    args = parser.parse_args()

    print("Generating loss landscape visualizations...\n")
    generate_combined_plots(args.output_dir, args.n_ratio, args.n_angle)
    if not args.skip_3d_equicorr:
        generate_3d_equicorrelation_plots(args.output_dir, args.n_ratio, args.n_rho)
    print("\nDone! Open figures/ to see loss comparisons.")


if __name__ == "__main__":
    main()
