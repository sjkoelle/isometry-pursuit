"""Analytic and numerical minimizers for the four-column rotation example."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .cloud_experiments import solve_matrix


def dictionary(theta):
    return np.column_stack((np.eye(2), [np.cos(theta), np.sin(theta)],
                            [np.cos(2*theta), np.sin(2*theta)]))


def weight_endpoint(theta):
    """Far endpoint of the exact optimal segment, away from theta=pi/4."""
    if theta < np.pi/4:
        return np.array([1., 1., 0., 0.])
    q = -np.sin(2*theta)/np.sin(4*theta)
    a = np.cos(theta)**2 + q*np.cos(2*theta)**2
    b = np.sin(theta)**2 + q*np.sin(2*theta)**2
    t = min(1/a, 1/b)
    return np.array([1-a*t, 1-b*t, t, q*t])


def generate_rotation_sweep(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    angles = np.arange(10., 86., 3.)  # Avoid coincident columns at 45 degrees.
    records = []
    for degrees in angles:
        theta = np.deg2rad(degrees)
        x = dictionary(theta)
        c = weight_endpoint(theta)
        exact = c[:, None]*x.T
        if np.min(c) < -1e-12 or not np.allclose(x@exact, np.eye(2), atol=1e-12):
            raise ValueError('Analytic minimizer is not feasible')
        for solver in ('scs', 'admm'):
            settings = dict(solver=solver, initialization='zero', seed=0,
                            eps_abs=1e-9, eps_rel=1e-9, max_iters=100000,
                            **({'scale': .1} if solver=='scs' else {'rho': 1.}))
            result = solve_matrix(x, settings, time_limit=10)
            beta = result['beta']
            residual = float(np.linalg.norm(x@beta-np.eye(2)))
            loss = float(np.linalg.norm(beta, axis=1).sum())
            if result['status'] != 'solved' or residual > 1e-6 or abs(loss-2) > 1e-6:
                raise ValueError(f'Unverified solve at {degrees}, {solver}: {result["status"]}')
            records.append(dict(degrees=float(degrees), solver=solver, settings=settings,
                                beta=beta.tolist(), row_norms=np.linalg.norm(beta, axis=1).tolist(),
                                residual=residual, loss=loss, iterations=result['basis_pursuit_iterations']))
    fig = plt.figure(figsize=(10.8, 6.5))
    grid = fig.add_gridspec(2, 3, height_ratios=(1.05, 1), hspace=.55, wspace=.3)
    axis = fig.add_subplot(grid[0, :])
    right = np.linspace(45.001, 85, 400)
    maximum = [weight_endpoint(np.deg2rad(d))[2:].sum() for d in right]
    axis.fill_between(right, 0, maximum, color='.85', label='Range over all minimizers')
    axis.plot([10, 45], [0, 0], color='.45', linewidth=2)
    for solver, color, marker in [('scs', '#4c78a8', 'o'), ('admm', '#f58518', 'x')]:
        subset = [r for r in records if r['solver']==solver]
        for side in ('left', 'right'):
            branch = [r for r in subset if (r['degrees'] < 45) == (side == 'left')]
            axis.plot([r['degrees'] for r in branch], [sum(r['row_norms'][2:]) for r in branch],
                      color=color, marker=marker, markersize=4, linewidth=1,
                      label=solver.upper() if side == 'left' else None)
    axis.axvline(45, color='.4', linestyle='--', linewidth=.9)
    axis.text(27, 1.53, 'Unique', ha='center', fontsize=10)
    axis.text(65, 1.53, 'Non-unique', ha='center', fontsize=10)
    axis.set(xlim=(10, 85), ylim=(-.05, 1.75), xlabel=r'$\theta$ (degrees)',
             ylabel=r'$\|\beta_{3.}\|_2+\|\beta_{4.}\|_2$')
    axis.legend(loc='upper left', bbox_to_anchor=(0, .83), fontsize=8, frameon=False)
    axis.grid(alpha=.15)
    colors = ('#4c78a8', '#e45756', '#54a24b', '#b279a2')
    circle = np.linspace(0, 2*np.pi, 200)
    for col, degrees in enumerate((30, 60, 75)):
        ax = fig.add_subplot(grid[1, col])
        x = dictionary(np.deg2rad(degrees))
        ax.plot(np.cos(circle), np.sin(circle), '--', color='.8', linewidth=.7)
        ax.axhline(0, color='.9', linewidth=.6); ax.axvline(0, color='.9', linewidth=.6)
        for p, color in enumerate(colors):
            ax.annotate('', xy=x[:, p], xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=color, linewidth=1.6))
            ax.text(*(1.13*x[:, p]), rf'$x_{{.{p+1}}}$', color=color, fontsize=9,
                    ha='center', va='center')
        ax.set(xlim=(-1.35, 1.35), ylim=(-1.25, 1.35), aspect='equal',
               xlabel=r'$x_{1p}$', ylabel=r'$x_{2p}$', title=rf'$\theta={degrees}^\circ$')
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=.10, right=.98, top=.96, bottom=.10)
    fig.savefig(output_dir/'four_column_rotation_sweep.pdf')
    fig.savefig(output_dir/'four_column_rotation_sweep.png', dpi=200)
    plt.close(fig)
    result = dict(records=records, scalar_uniqueness='Theorem 2 condition holds for every plotted angle; theta=45 degrees excluded.',
                  max_residual=max(r['residual'] for r in records),
                  max_objective_error=max(abs(r['loss']-2) for r in records))
    (output_dir/'four_column_rotation_sweep.json').write_text(json.dumps(result, indent=2)+'\n')
    return result
