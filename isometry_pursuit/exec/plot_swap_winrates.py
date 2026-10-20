"""Plot paired winrate heatmaps from the four-strategy experiment results."""
import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'matplotlib'))
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from isometry_pursuit.swap_plot import METHODS, draw_winrate_heatmap, pairwise_winrates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=Path('results'))
    parser.add_argument('--output-dir', type=Path, default=Path('figures/swap_winrates'))
    parser.add_argument('--tie-tolerance', type=float, default=1e-9)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = []
    for name in ('wine', 'iris', 'ethanol'):
        path = args.results_dir / f'{name}_swap_comparison.json'
        records = json.loads(path.read_text())['replicates']
        losses = [[r['methods'][m]['loss'] for m in METHODS] for r in records]
        datasets.append((name, name.title(), path, np.array(losses, dtype=float)))
    path = args.results_dir / 'identity_swap_comparison_full/identity_experiment_runs.json'
    prefixes = ('forward_backward_full', 'backward_swap_full',
                'forward_backward_restricted', 'backward_swap_restricted')
    records = json.loads(path.read_text())['runs']
    losses = [[r[p + '_loss'] for p in prefixes] for r in records]
    datasets.append(('synthetic', 'Synthetic identity (all dimensions)', path,
                     np.array(losses, dtype=float)))
    combined, axes = plt.subplots(2, 2, figsize=(13, 12), layout='constrained')
    stats = dict(methods=list(METHODS), definition='P(row loss < column loss) + 0.5 P(tie)',
                 tie_tolerance=args.tie_tolerance, datasets={})
    for ax, (name, title, path, losses) in zip(axes.flat, datasets):
        rates, wins, ties = pairwise_winrates(losses, args.tie_tolerance)
        title += f' (n={len(losses)})'
        im = draw_winrate_heatmap(ax, rates, title)
        stats['datasets'][name] = dict(source=str(path), replicates=len(losses),
                                      winrates=rates.tolist(), strict_wins=wins.tolist(),
                                      ties=ties.tolist())
        fig, individual_ax = plt.subplots(figsize=(7, 6.5), layout='constrained')
        individual_im = draw_winrate_heatmap(individual_ax, rates, title)
        fig.colorbar(individual_im, ax=individual_ax, format=PercentFormatter(1),
                     label='Row winrate (ties = ½ win)', shrink=.8)
        fig.savefig(args.output_dir / f'{name}_swap_winrates.png', dpi=200)
        fig.savefig(args.output_dir / f'{name}_swap_winrates.pdf')
        plt.close(fig)
        with (args.output_dir / f'{name}_swap_winrates.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['row_method', *METHODS])
            writer.writerows([m, *rates[i]] for i, m in enumerate(METHODS))
        print(name, np.round(rates, 3).tolist())
    combined.colorbar(im, ax=axes, format=PercentFormatter(1),
                      label='Row winrate (ties = ½ win)', shrink=.8)
    combined.suptitle('Paired final-loss winrates · row vs column · ties = ½ win', fontsize=16)
    combined.savefig(args.output_dir / 'all_swap_winrates.png', dpi=200)
    combined.savefig(args.output_dir / 'all_swap_winrates.pdf')
    plt.close(combined)
    (args.output_dir / 'swap_winrates.json').write_text(json.dumps(stats, indent=2) + '\n')


if __name__ == '__main__':
    main()
