"""Pairwise comparison plots for the four greedy-initialized swap methods."""
from matplotlib import colormaps
import numpy as np


METHODS = ('greedy_swap', 'backward_swap', 'ip_greedy_swap', 'ip_backward_swap')
LABELS = ('Greedy\n→ swap', 'Backward\n→ swap',
          'IP → greedy\n→ swap', 'IP → backward\n→ swap')


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


def draw_winrate_heatmap(ax, rates, title, ties=None, replicates=None):
    """Draw paired superiority probabilities; leave self-comparisons blank."""
    displayed = np.asarray(rates, dtype=float).copy()
    np.fill_diagonal(displayed, np.nan)
    cmap = colormaps['RdBu'].copy()
    cmap.set_bad('#eeeeee')
    im = ax.imshow(displayed, vmin=0, vmax=1, cmap=cmap, aspect='equal')
    ax.set_xticks(range(4), LABELS, fontsize=9)
    ax.set_yticks(range(4), LABELS, fontsize=9)
    ax.set_xlabel('Column method')
    ax.set_ylabel('Row method')
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
