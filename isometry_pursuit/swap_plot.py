"""Pairwise comparison plots for the four greedy-initialized swap methods."""
import numpy as np


METHODS = ('greedy_swap', 'backward_swap', 'ip_greedy_swap', 'ip_backward_swap')
LABELS = ('Greedy\n→ swap', 'Backward\n→ swap',
          'IP → greedy\n→ swap', 'IP → backward\n→ swap')


def pairwise_winrates(losses, tie_tolerance=1e-9):
    """Row beats column, ties scored as half a win; lower paired loss wins.

    losses is (replicates, methods). Absolute differences <= tie_tolerance tie.
    Returns fractional winrates, strict-win counts, and tie counts.
    """
    losses = np.asarray(losses, dtype=float)
    if losses.ndim != 2 or not losses.shape[0] or not np.all(np.isfinite(losses)):
        raise ValueError('Expected a nonempty finite (replicates, methods) loss matrix')
    if not np.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError('tie_tolerance must be finite and nonnegative')
    differences = losses[:, :, None] - losses[:, None, :]
    wins = (differences < -tie_tolerance).sum(axis=0)
    ties = (np.abs(differences) <= tie_tolerance).sum(axis=0)
    return (wins + .5 * ties) / losses.shape[0], wins, ties


def draw_winrate_heatmap(ax, rates, title):
    """Draw an annotated 4 by 4 heatmap with a shared 0–100% color scale."""
    im = ax.imshow(rates, vmin=0, vmax=1, cmap='RdBu', aspect='equal')
    ax.set_xticks(range(4), LABELS, fontsize=9)
    ax.set_yticks(range(4), LABELS, fontsize=9)
    ax.set_xlabel('Column method')
    ax.set_ylabel('Row method')
    ax.set_title(title, pad=12)
    for i in range(4):
        for j in range(4):
            value = rates[i, j]
            ax.text(j, i, f'{100*value:.1f}%', ha='center', va='center',
                    color='white' if abs(value-.5) > .3 else 'black', fontsize=12)
    return im
