"""Two-dimensional column-selection examples from saved search results."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .experiment import _gl_loss
from .simulation import generate_identity_instance
from .swap_plot import METHODS
from .transformation import exponential_transformation


TITLES = ('Forward → swap', 'Backward → swap',
          'IP → forward → swap', 'IP → backward → swap')
COLORS = ('#4c78a8', '#f58518', '#54a24b', '#b279a2')
PREFIXES = ('forward_backward_full', 'backward_swap_full',
            'forward_backward_restricted', 'backward_swap_restricted')


def _choose(records):
    """First agreement, first screening improvement, first screening deterioration."""
    groups = []
    for criterion in ('agreement', 'improvement', 'deterioration'):
        for record in records:
            methods = record['methods']
            delta = methods['ip_greedy_swap']['loss'] - methods['greedy_swap']['loss']
            matches = (abs(delta) <= 1e-9 if criterion == 'agreement' else
                       delta < -1e-9 if criterion == 'improvement' else delta > 1e-9)
            if matches:
                groups.append((criterion, record))
                break
        else:
            raise ValueError(f'No {criterion} example found')
    return groups


def _plot(examples, output, planted=False):
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.8))
    fig.subplots_adjust(left=.055, right=.99, top=.80, bottom=.23, wspace=.20)
    theta = np.linspace(0, 2*np.pi, 300)
    markers = ('o', 's', 'x', '+')
    sizes = (110, 65, 62, 95)
    audit = []
    for ax, (criterion, record) in zip(axes, examples):
        x = record['matrix']
        normalized = exponential_transformation(x)
        support = record['ip_support']
        limit = max(1.08, 1.08 * float(np.max(np.abs(x))))
        checks = {}
        ax.plot(np.cos(theta), np.sin(theta), '--', color='.65', linewidth=.7)
        ax.axhline(0, color='.9', linewidth=.6)
        ax.axvline(0, color='.9', linewidth=.6)
        ax.scatter(*x, color='.65', s=9, alpha=.30, linewidths=0)
        ax.scatter(*x[:, support], facecolors='none', edgecolors='.2', s=155, linewidths=.7)
        if planted:
            ax.scatter(*x[:, :2], marker='*', color='black', s=30, zorder=3)
        selected_union = sorted({s for m in METHODS for s in record['methods'][m]['indices']})
        for s in selected_union:
            ax.plot([0, x[0, s]], [0, x[1, s]], color='.5', linewidth=.9, alpha=.6)
        losses = []
        for method, color, marker, size in zip(METHODS, COLORS, markers, sizes):
            saved = record['methods'][method]
            selected = saved['indices']
            loss = _gl_loss(normalized[:, selected])
            if not np.isclose(loss, saved['loss'], rtol=1e-7, atol=1e-8):
                raise ValueError(f'Saved loss mismatch: {record["id"]}, {method}')
            if method.startswith('ip_') and not set(selected).issubset(support):
                raise ValueError('Restricted search selected outside the IP support')
            checks[method] = dict(indices=selected, saved_loss=saved['loss'], checked_loss=float(loss))
            losses.append(loss)
            style = dict(facecolors='none', edgecolors=color) if marker in ('o', 's') else dict(color=color)
            ax.scatter(*x[:, selected], marker=marker, s=size, linewidths=1.5, zorder=5, **style)
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect='equal')
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
        ax.tick_params(labelsize=8)
        ax.set_title(record['label'] + '\n' +
                     f'F: {losses[0]:.4f}   B: {losses[1]:.4f}\n' +
                     f'IP-F: {losses[2]:.4f}   IP-B: {losses[3]:.4f}', fontsize=9, pad=9)
        ax.set_xlabel(r'$x_{1p}$', fontsize=10)
        ax.set_ylabel(r'$x_{2p}$', fontsize=10)
        audit.append(dict(case=criterion, id=record['id'], ip_support=support, methods=checks))
    handles = [Line2D([], [], marker=m, color=c, markerfacecolor='none', linestyle='none',
                      markersize=7, label=t) for m, c, t in zip(markers, COLORS, TITLES)]
    handles += [Line2D([], [], marker='o', markerfacecolor='none', markeredgecolor='.2', linestyle='none', label='IP support'),
                Line2D([], [], color='.65', linestyle='--', label='Unit circle')]
    if planted:
        handles.append(Line2D([], [], marker='*', color='black', linestyle='none', label='Planted columns'))
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=9)
    output = Path(output)
    fig.savefig(output.with_suffix('.pdf'))
    fig.savefig(output.with_suffix('.png'), dpi=200)
    plt.close(fig)
    return audit


def generate_selection_examples(root, output_dir):
    root, output_dir = Path(root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ethanol = json.loads((root/'results/ethanol_swap_comparison.json').read_text())
    ip = json.loads((root/'results/ethanol_results.json').read_text())
    tensor = np.load(root/ethanol['metadata']['data_file'])
    real = []
    for r in ethanol['replicates']:
        i = r['replicate']
        real.append(dict(**r, matrix=tensor[:, :, i],
                         ip_support=ip['replicates'][i]['basis_pursuit_indices'],
                         id=f'ethanol-replicate-{i}', label=f'Replicate {i+1}'))
    runs = json.loads((root/'results/identity_swap_comparison_25x_D64/identity_experiment_runs.json').read_text())['runs']
    synthetic = []
    for r in runs:
        if r['D'] != 2:
            continue
        matrix, *_ = generate_identity_instance(2, r['corrupting_D'], r['num_corrupting_matrices'],
                                                np.random.default_rng(r['seed']))
        methods = {m: dict(indices=r[p+'_indices'], loss=r[p+'_loss']) for m, p in zip(METHODS, PREFIXES)}
        synthetic.append(dict(matrix=matrix, methods=methods, ip_support=r['selected_indices'],
                              blocks=r['num_corrupting_matrices'], replicate=r['replicate'],
                              id=f'synthetic-run-{r["run_id"]}', label=f'P={matrix.shape[1]}'))
    synthetic_examples = [(f'{b} added blocks', next(r for r in synthetic if r['blocks']==b and r['replicate']==0))
                          for b in (1, 4, 16)]
    audit = dict(selection='Ethanol: first saved agreement, improvement, and deterioration of IP-forward versus full-forward. Synthetic: first replicate at 1, 4, and 16 added blocks.',
                 ethanol=_plot(_choose(real), output_dir/'ethanol_selection_examples'),
                 synthetic=_plot(synthetic_examples, output_dir/'synthetic_selection_examples', planted=True))
    (output_dir/'selection_examples.json').write_text(json.dumps(audit, indent=2)+'\n')
    return audit
