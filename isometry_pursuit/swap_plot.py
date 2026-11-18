"""Pairwise comparison plots for the four greedy-initialized swap methods."""
from matplotlib import colormaps
import numpy as np


METHODS = ('greedy_swap', 'backward_swap', 'ip_greedy_swap', 'ip_backward_swap')
LABELS = ('Greedy\n→ swap', 'Backward\n→ swap',
          'IP → greedy\n→ swap', 'IP → backward\n→ swap')


def plot_real_regrets(results_dir, output_path):
    """Plot replicate regrets against the best of the four searches."""
    import json
    from pathlib import Path
    import matplotlib.pyplot as plt

    colors = ('#4c78a8', '#f58518', '#54a24b', '#b279a2')
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 2.7), layout='constrained')
    for ax, name in zip(axes, ('iris', 'wine', 'ethanol')):
        records = json.loads((Path(results_dir) / f'{name}_swap_comparison.json').read_text())['replicates']
        losses = np.asarray([[r['methods'][m]['loss'] for m in METHODS] for r in records])
        if not np.all(np.isfinite(losses)):
            raise ValueError(f'Nonfinite losses for {name}')
        regrets = losses - losses.min(axis=1, keepdims=True)
        regrets[np.abs(regrets) <= 1e-9] = 0
        best = regrets == 0
        unique_best = best & (best.sum(axis=1, keepdims=True) == 1)
        for j, color in enumerate(colors):
            ax.scatter(j + np.linspace(-.12, .12, len(records)), regrets[:, j],
                       color=color, alpha=np.where(unique_best[:, j], 1., .2),
                       s=15, linewidths=0)
            ax.plot([j-.22, j+.22], [regrets[:, j].mean()]*2,
                    color='black', linewidth=2)
        ax.set_xticks(range(4), LABELS, fontsize=8)
        ax.set_title(name.title(), fontsize=11)
        ax.set_ylim(bottom=-.04 * max(float(regrets.max()), .01))
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Regret against best search')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix('.png'), dpi=220)
    fig.savefig(output_path.with_suffix('.pdf'))
    plt.close(fig)


def pairwise_winrates(losses, tie_tolerance=1e-9):
    """Paired probability that the row has lower loss than the column.

    losses is (replicates, methods). Absolute differences <= tie_tolerance tie.
    A tie contributes one half, giving the paired common-language effect size
    P(row wins) + 0.5 P(tie). Returns effect sizes, strict-win counts, and tie
    counts.
    """
    losses = np.asarray(losses, dtype=float)
    if losses.ndim != 2 or not losses.shape[0] or not np.all(np.isfinite(losses)):
        raise ValueError('Expected a nonempty finite (replicates, methods) loss matrix')
    if not np.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError('tie_tolerance must be finite and nonnegative')
    differences = losses[:, :, None] - losses[:, None, :]
    wins = (differences < -tie_tolerance).sum(axis=0)
    ties = (np.abs(differences) <= tie_tolerance).sum(axis=0)
    return (wins + 0.5 * ties) / losses.shape[0], wins, ties


def pairwise_defeat_frequencies(losses, tie_tolerance=1e-9):
    """Count strict paired wins using only mutually observed finite losses."""
    losses=np.asarray(losses,dtype=float)
    if losses.ndim!=2 or not len(losses) or tie_tolerance<0 or not np.isfinite(tie_tolerance):
        raise ValueError('Expected a nonempty loss matrix and nonnegative finite tolerance')
    valid=np.isfinite(losses[:,:,None]) & np.isfinite(losses[:,None,:])
    differences=losses[:,:,None]-losses[:,None,:]
    paired=valid.sum(axis=0)
    wins=(valid & (differences < -tie_tolerance)).sum(axis=0)
    ties=(valid & (np.abs(differences)<=tie_tolerance)).sum(axis=0)
    rates=np.divide(wins,paired,out=np.full(paired.shape,np.nan,dtype=float),where=paired>0)
    return rates,wins,ties,paired


def plot_defeat_matrices(results_dir, output_dir, tie_tolerance=1e-9, synthetic_bundle=None, two_rows=False):
    """Save strict pairwise win frequencies, with ties excluded from wins."""
    import csv
    import json
    from pathlib import Path
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names=['iris','wine','ethanol']
    synthetic_rows=None
    if synthetic_bundle is not None:
        from .cloud_search_plot import join_full_synthetic_results
        _,_,synthetic_rows=join_full_synthetic_results(Path(results_dir).resolve().parent,synthetic_bundle)
        names.insert(0,'synthetic')
    if two_rows:
        if len(names)!=4:
            raise ValueError('The two-row layout requires synthetic and all three real datasets')
        fig, axes = plt.subplots(2,2,figsize=(8.2,7.2),layout='constrained')
        axes=axes.flat
    else:
        fig, axes = plt.subplots(1, len(names), figsize=(15, 5.1), layout='constrained')
    summary = {}
    axes=list(axes)
    for ax, name in zip(axes, names):
        if name=='synthetic':
            records=synthetic_rows
            keys=('forward_backward_full','backward_swap_full','forward_backward_restricted','backward_swap_restricted')
            losses=np.asarray([[r[k+'_loss'] for k in keys] for r in records],dtype=float)
        else:
            records = json.loads((Path(results_dir) / f'{name}_swap_comparison.json').read_text())['replicates']
            losses = np.asarray([[r['methods'][m]['loss'] for m in METHODS] for r in records])
            if not np.all(np.isfinite(losses)):
                raise ValueError(f'Nonfinite real-dataset losses for {name}')
        rates,wins,ties,paired=pairwise_defeat_frequencies(losses,tie_tolerance)
        im = draw_winrate_heatmap(ax, rates, f'{name.title()} (n={len(records)})')
        ax.set_xticks(range(4), ('Greedy\n→ swap', 'Backward\n→ swap',
                                'IP →\ngreedy →\nswap', 'IP →\nbackward →\nswap'),
                      fontsize=7 if len(names)==4 else 9)
        summary[name] = dict(replicates=len(records), methods=METHODS,
                             row_defeats_column=rates.tolist(), wins=wins.tolist(),
                             ties=ties.tolist(), paired_counts=paired.tolist())
        with (output_dir / f'{name}_defeat_frequencies.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['row_defeats_column', *METHODS])
            writer.writerows([method, *rates[j]] for j, method in enumerate(METHODS))
    fig.colorbar(im, ax=axes, format=PercentFormatter(1), shrink=.75,
                 label='Pairwise defeat frequency')
    fig.savefig(output_dir / 'optimizer_defeat_matrices.png', dpi=220)
    fig.savefig(output_dir / 'optimizer_defeat_matrices.pdf')
    plt.close(fig)
    (output_dir / 'defeat_frequencies.json').write_text(json.dumps(
        dict(tie_tolerance=tie_tolerance, datasets=summary), indent=2) + '\n')


def draw_winrate_heatmap(ax, rates, title, ties=None, replicates=None):
    """Draw paired superiority probabilities; leave self-comparisons blank."""
    displayed = np.asarray(rates, dtype=float).copy()
    np.fill_diagonal(displayed, np.nan)
    cmap = colormaps['RdBu'].copy()
    cmap.set_bad('#eeeeee')
    im = ax.imshow(displayed, vmin=0, vmax=1, cmap=cmap, aspect='equal')
    ax.set_xticks(range(4), LABELS, fontsize=9)
    ax.set_yticks(range(4), LABELS, fontsize=9)
    ax.set_xlabel('Losing method')
    ax.set_ylabel('Winning method')
    ax.set_title(title, pad=12)
    for i in range(4):
        for j in range(4):
            if i == j:
                ax.text(j, i, '—', ha='center', va='center', color='#666666',
                        fontsize=14)
                continue
            value = rates[i, j]
            label = f'{100*value:.1f}%'
            if ties is not None and replicates:
                label += f'\n({100*ties[i, j]/replicates:.0f}% tied)'
            ax.text(j, i, label, ha='center', va='center',
                    color='white' if value < .22 or value > .78 else 'black',
                    fontsize=10)
    return im
