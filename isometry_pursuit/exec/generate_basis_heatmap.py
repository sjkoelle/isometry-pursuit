#!/usr/bin/env python3
"""
Generate heatmaps of group_lasso_norm over a 2D grid of
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

from isometry_pursuit.loss import group_lasso_norm
from isometry_pursuit.algorithm import group_basis_pursuit
from isometry_pursuit.transformation import exponential_transformation


def compute_grids(n_ratio=40, n_angle=40, max_ratio=5.0):
    ratios = np.linspace(1.0, max_ratio, n_ratio)
    angles = np.linspace(0.05, np.pi - 0.05, n_angle)

    group_lasso_grid = np.full((n_angle, n_ratio), np.nan)
    nuclear_norm_grid = np.full((n_angle, n_ratio), np.nan)
    distortion_grid = np.full((n_angle, n_ratio), np.nan)

    total = n_ratio * n_angle
    with tqdm(total=total, desc="Computing grid") as pbar:
        for i, theta in enumerate(angles):
            for j, r in enumerate(ratios):
                X = np.array([[1.0, r * np.cos(theta)],
                               [0.0, r * np.sin(theta)]])

                svs = np.linalg.svd(X, compute_uv=False)
                sv_max, sv_min = svs[0], svs[1]

                nuclear_norm_grid[i, j] = sv_max + sv_min

                if sv_min > 1e-14:
                    distortion_grid[i, j] = sv_max / sv_min
                else:
                    distortion_grid[i, j] = np.nan

                try:
                    X_t = exponential_transformation(X)
                    beta, _, obj = group_basis_pursuit(X_t, constraint_violation=0.0)
                    group_lasso_grid[i, j] = obj
                except Exception:
                    group_lasso_grid[i, j] = np.nan

                pbar.update(1)

    return ratios, angles, group_lasso_grid, nuclear_norm_grid, distortion_grid


def plot_heatmaps(ratios, angles, group_lasso_grid, nuclear_norm_grid, distortion_grid, output_dir, log=True):
    os.makedirs(output_dir, exist_ok=True)

    angle_labels = [r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"]
    angle_ticks = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
    suffix = "_log" if log else "_raw"
    scale_label = " (log scale)" if log else ""

    grids = [
        (group_lasso_grid, "Group Lasso Norm", f"basis_group_lasso_heatmap{suffix}.png"),
        (nuclear_norm_grid, "Nuclear Norm", f"basis_nuclear_norm_heatmap{suffix}.png"),
        (distortion_grid, r"Distortion ($\sigma_{max}/\sigma_{min}$)", f"basis_distortion_heatmap{suffix}.png"),
    ]

    def _prepare(grid):
        g = np.log1p(grid) if log else grid.copy()
        finite = g[np.isfinite(g)]
        vmin = np.percentile(finite, 1)
        vmax = np.percentile(finite, 99)
        if log:
            vmin, vmax = 1.0, 5.0
        return g, vmin, vmax

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (grid, title, _) in zip(axes.flat, grids):
        g, vmin, vmax = _prepare(grid)
        im = ax.pcolormesh(ratios, angles, g, cmap="viridis", vmin=vmin, vmax=vmax, shading="auto")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel(r"Length ratio $r$", fontsize=12)
        ax.set_ylabel(r"Angle $\theta$", fontsize=12)
        ax.set_title(title + scale_label, fontsize=13)
        ax.set_yticks(angle_ticks)
        ax.set_yticklabels(angle_labels, fontsize=10)

    plt.tight_layout()
    combined_path = os.path.join(output_dir, f"basis_heatmaps{suffix}.png")
    plt.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Combined heatmap saved to {combined_path}")

    for grid, title, fname in grids:
        fig, ax = plt.subplots(figsize=(8, 6))
        g, vmin, vmax = _prepare(grid)
        im = ax.pcolormesh(ratios, angles, g, cmap="viridis", vmin=vmin, vmax=vmax, shading="auto")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel(r"Length ratio $r = \|v_2\| / \|v_1\|$", fontsize=13)
        ax.set_ylabel(r"Angle $\theta$ between vectors", fontsize=13)
        ax.set_title(title + scale_label, fontsize=15)
        ax.set_yticks(angle_ticks)
        ax.set_yticklabels(angle_labels, fontsize=11)
        plt.tight_layout()
        out = os.path.join(output_dir, fname)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")


def plot_marginal_theta(ratios, angles, group_lasso_grid, nuclear_norm_grid, distortion_grid, output_dir, log=True):
    xticks = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
    xlabels = [r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"]
    suffix = "_log" if log else "_raw"

    slices = [
        (group_lasso_grid[:, 0],   "Group Lasso Norm", "C1", "--"),
        (nuclear_norm_grid[:, 0],  "Nuclear Norm", "C2", "-."),
        (distortion_grid[:, 0],    r"Distortion ($\sigma_{max}/\sigma_{min}$)", "C3", ":"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax = axes[0]
    for vals, label, color, ls in slices:
        ax.plot(angles, vals, label=label, linewidth=2, color=color, linestyle=ls)
    if log:
        ax.set_yscale("log")
    ax.set_xlabel(r"Angle $\theta$", fontsize=13)
    ax.set_ylabel("Value" + (" (log scale)" if log else ""), fontsize=13)
    ax.set_title(r"Marginal vs $\theta$ at $r = 1$", fontsize=15)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=11)
    ax.legend(fontsize=11)

    gl_slice = group_lasso_grid[:, 0]
    ax = axes[1]
    for vals, label, color, ls in slices:
        if label != "Group Lasso Norm":
            ax.plot(angles, vals / gl_slice, label=f"{label} / GL Norm", linewidth=2, color=color, linestyle=ls)
    if log:
        ax.set_yscale("log")
    ax.set_xlabel(r"Angle $\theta$", fontsize=13)
    ax.set_ylabel("Ratio to Group Lasso Norm" + (" (log scale)" if log else ""), fontsize=12)
    ax.set_title(r"Ratio to Group Lasso Norm at $r = 1$", fontsize=15)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=11)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.5, label="ratio = 1")
    ax.legend(fontsize=11)

    plt.tight_layout()
    out = os.path.join(output_dir, f"basis_marginal_theta{suffix}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def compute_grids_xy(n=50, xy_range=2.0):
    """Grid over (x, y) of v2, with v1 = [1, 0], X = [[1, x], [0, y]]."""
    coords = np.linspace(-xy_range, xy_range, n)
    xs, ys = coords, coords

    group_lasso_grid   = np.full((n, n), np.nan)
    nuclear_norm_grid  = np.full((n, n), np.nan)
    distortion_grid    = np.full((n, n), np.nan)

    with tqdm(total=n * n, desc="Computing xy grid") as pbar:
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                if abs(y) < 1e-6:
                    pbar.update(1)
                    continue
                X = np.array([[1.0, x], [0.0, y]])
                svs = np.linalg.svd(X, compute_uv=False)
                sv_max, sv_min = svs[0], svs[1]

                nuclear_norm_grid[i, j] = sv_max + sv_min
                if sv_min > 1e-14:
                    distortion_grid[i, j] = sv_max / sv_min

                try:
                    X_t = exponential_transformation(X)
                    beta, _, obj = group_basis_pursuit(X_t, constraint_violation=0.0)
                    group_lasso_grid[i, j] = obj
                except Exception:
                    pass

                pbar.update(1)

    return xs, ys, group_lasso_grid, nuclear_norm_grid, distortion_grid


def plot_heatmaps_xy(xs, ys, group_lasso_grid, nuclear_norm_grid, distortion_grid, output_dir, log=True):
    os.makedirs(output_dir, exist_ok=True)
    suffix = "_xy_log" if log else "_xy_raw"
    scale_label = " (log scale)" if log else ""

    grids = [
        (group_lasso_grid,   "Group Lasso Norm",                    f"basis_group_lasso_heatmap{suffix}.png"),
        (nuclear_norm_grid,  "Nuclear Norm",                        f"basis_nuclear_norm_heatmap{suffix}.png"),
        (distortion_grid,    r"Distortion ($\sigma_{max}/\sigma_{min}$)", f"basis_distortion_heatmap{suffix}.png"),
    ]

    def _prepare(grid):
        g = np.log1p(grid) if log else grid.copy()
        finite = g[np.isfinite(g)]
        if log:
            return g, 1.0, 5.0
        vmin, vmax = np.percentile(finite, 1), np.percentile(finite, 99)
        return g, vmin, vmax

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (grid, title, _) in zip(axes.flat, grids):
        g, vmin, vmax = _prepare(grid)
        im = ax.pcolormesh(xs, ys, g, cmap="viridis", vmin=vmin, vmax=vmax, shading="auto")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel(r"$v_{2x}$", fontsize=13)
        ax.set_ylabel(r"$v_{2y}$", fontsize=13)
        ax.set_title(title + scale_label, fontsize=13)
        ax.axhline(0, color="white", linewidth=0.5, linestyle=":")
        ax.axvline(0, color="white", linewidth=0.5, linestyle=":")
        # Mark v1 = [1, 0]
        ax.plot(1, 0, "w*", markersize=10)

    plt.suptitle(r"$v_1 = [1,\,0]$,  $v_2 = [x,\,y]$  ($\star$ marks $v_1$ tip)", fontsize=13)
    plt.tight_layout()
    combined_path = os.path.join(output_dir, f"basis_heatmaps{suffix}.png")
    plt.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Combined heatmap saved to {combined_path}")

    for grid, title, fname in grids:
        fig, ax = plt.subplots(figsize=(7, 6))
        g, vmin, vmax = _prepare(grid)
        im = ax.pcolormesh(xs, ys, g, cmap="viridis", vmin=vmin, vmax=vmax, shading="auto")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel(r"$v_{2x}$", fontsize=13)
        ax.set_ylabel(r"$v_{2y}$", fontsize=13)
        ax.set_title(title + scale_label, fontsize=14)
        ax.axhline(0, color="white", linewidth=0.5, linestyle=":")
        ax.axvline(0, color="white", linewidth=0.5, linestyle=":")
        ax.plot(1, 0, "w*", markersize=10)
        plt.tight_layout()
        out = os.path.join(output_dir, fname)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Heatmaps of group_lasso_norm over 2D basis grid"
    )
    parser.add_argument("--output-dir", default="figures", help="Output directory")
    parser.add_argument("--n-ratio", type=int, default=40, help="Grid points along ratio axis")
    parser.add_argument("--n-angle", type=int, default=40, help="Grid points along angle axis")
    parser.add_argument("--max-ratio", type=float, default=5.0, help="Max length ratio")
    parser.add_argument("--n-xy", type=int, default=60, help="Grid points per axis for xy plots")
    parser.add_argument("--xy-range", type=float, default=2.0, help="xy plot range [-r, r]")
    args = parser.parse_args()

    ratios, angles, group_lasso_grid, nuclear_norm_grid, distortion_grid = compute_grids(
        n_ratio=args.n_ratio, n_angle=args.n_angle, max_ratio=args.max_ratio
    )
    for log in (True, False):
        plot_heatmaps(ratios, angles, group_lasso_grid, nuclear_norm_grid, distortion_grid, args.output_dir, log=log)
        plot_marginal_theta(ratios, angles, group_lasso_grid, nuclear_norm_grid, distortion_grid, args.output_dir, log=log)

    xs, ys, gl_xy, nn_xy, dist_xy = compute_grids_xy(n=args.n_xy, xy_range=args.xy_range)
    for log in (True, False):
        plot_heatmaps_xy(xs, ys, gl_xy, nn_xy, dist_xy, args.output_dir, log=log)


if __name__ == "__main__":
    main()
