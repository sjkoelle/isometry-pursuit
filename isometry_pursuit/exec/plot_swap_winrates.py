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
    regret_rows = []
    regret_summary = []
    stats = dict(methods=list(METHODS),
                 definition=('Paired probability of superiority: P(row loss < column loss) '
                             '+ 0.5 P(numerical tie)'),
                 tie_tolerance=args.tie_tolerance, datasets={})
    for ax, (name, title, path, losses) in zip(axes.flat, datasets):
        rates, wins, ties = pairwise_winrates(losses, args.tie_tolerance)
        regrets = losses - losses.min(axis=1, keepdims=True)
        regrets[np.abs(regrets) <= args.tie_tolerance] = 0.0
        dataset_regret_summary = {}
        for method_index, method in enumerate(METHODS):
            method_regrets = regrets[:, method_index]
            summary = dict(
                dataset=name,
                method=method,
                replicates=len(losses),
                mean_regret=float(np.mean(method_regrets)),
                median_regret=float(np.median(method_regrets)),
                p90_regret=float(np.quantile(method_regrets, .9)),
                max_regret=float(np.max(method_regrets)),
                zero_regret_rate=float(np.mean(method_regrets == 0)),
            )
            regret_summary.append(summary)
            dataset_regret_summary[method] = {
                key: value for key, value in summary.items()
                if key not in ('dataset', 'method', 'replicates')
            }
            regret_rows.extend(
                dict(dataset=name, replicate=replicate, method=method,
                     loss=float(losses[replicate, method_index]),
                     oracle_loss=float(losses[replicate].min()),
                     oracle_regret=float(method_regrets[replicate]))
                for replicate in range(len(losses))
            )
        title += f' (n={len(losses)})'
        im = draw_winrate_heatmap(ax, rates, title, ties, len(losses))
        stats['datasets'][name] = dict(source=str(path), replicates=len(losses),
                                      winrates=rates.tolist(), strict_wins=wins.tolist(),
                                      ties=ties.tolist(),
                                      oracle_regret=dataset_regret_summary)
        fig, individual_ax = plt.subplots(figsize=(7, 6.5), layout='constrained')
        individual_im = draw_winrate_heatmap(
            individual_ax, rates, title, ties, len(losses))
        fig.colorbar(individual_im, ax=individual_ax, format=PercentFormatter(1),
                     label='Paired probability row has lower loss', shrink=.8)
        fig.savefig(args.output_dir / f'{name}_swap_winrates.png', dpi=200)
        fig.savefig(args.output_dir / f'{name}_swap_winrates.pdf')
        plt.close(fig)
        with (args.output_dir / f'{name}_swap_winrates.csv').open('w', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(['row_method', *METHODS])
            writer.writerows([m, *rates[i]] for i, m in enumerate(METHODS))
        with (args.output_dir / f'{name}_swap_tie_rates.csv').open(
                'w', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(['row_method', *METHODS])
            writer.writerows(
                [m, *(ties[i] / len(losses))] for i, m in enumerate(METHODS))
        print(name, np.round(rates, 3).tolist())
    combined.colorbar(im, ax=axes, format=PercentFormatter(1),
                      label='Paired probability row has lower loss', shrink=.8)
    combined.suptitle(
        'Paired probability of superiority · lower loss wins · half credit for ties',
        fontsize=16,
    )
    combined.savefig(args.output_dir / 'all_swap_winrates.png', dpi=200)
    combined.savefig(args.output_dir / 'all_swap_winrates.pdf')
    plt.close(combined)
    for filename, rows in (
            ('oracle_regret_summary.csv', regret_summary),
            ('oracle_regrets.csv', regret_rows)):
        with (args.output_dir / filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    (args.output_dir / 'swap_winrates.json').write_text(json.dumps(stats, indent=2) + '\n')


if __name__ == '__main__':
    main()
