"""Compare Figure 3b at two fixed SCS scales using identical axes."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from isometry_pursuit.cloud_search_plot import join_search_results, plot_updated_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--search-overrides', type=Path, default=Path(
        'results/cloud/swap-completion-20261116-001/merged_searches.json'))
    parser.add_argument('--output', type=Path, default=Path(
        'figures/complexity_runtime/scs_scale_01_vs_1'))
    args = parser.parse_args()
    solvers, searches, _ = join_search_results(args.root, args.search_overrides)
    fig, axes = plt.subplots(1, 2, figsize=(16.4, 6.1), sharex=True, sharey=True)
    fig.subplots_adjust(left=.055, right=.99, bottom=.17, top=.9, wspace=.1)
    for scale, ax in zip((.1, 1.), axes):
        selected = [r for r in solvers
                    if r['settings']['solver'] == 'scs' and r['settings']['scale'] == scale]
        assert len(selected) == 65
        plot_updated_runtime(args.root, selected, searches, main=True, ax=ax)
        solved = sum(r['status'] == 'solved' for r in selected)
        ax.set_title(f'SCS scale {scale:g} — {solved}/65 converged', fontsize=13)
    axes[1].set_ylabel('')
    fig.text(.5, .035,
             'Only the large-case IP points change; all other points are identical. '
             'Work uses recorded iterations; open markers denote capped runs.',
             ha='center', fontsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix('.png'), dpi=170)
    fig.savefig(args.output.with_suffix('.pdf'))
    plt.close(fig)
    print(args.output.with_suffix('.png'))


if __name__ == '__main__':
    main()
