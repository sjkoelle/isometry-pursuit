"""Plot exact synthetic support-recovery rates by experiment condition."""
import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'matplotlib'))
import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    ('Greedy → swap', 'forward_backward_full', '#4c78a8', 'o'),
    ('Backward → swap', 'backward_swap_full', '#f58518', 's'),
    ('IP → greedy → swap', 'forward_backward_restricted', '#54a24b', '^'),
    ('IP → backward → swap', 'backward_swap_restricted', '#b279a2', 'D'),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--input', type=Path,
        default=Path('results/cloud/full-synthetic-20261116-001/figure3_main_search_runs.json'))
    parser.add_argument(
        '--output', type=Path,
        default=Path('figures/synthetic_support_recovery_grid'))
    parser.add_argument('--optimizer-results', type=Path,
                        default=Path('results/cloud/full-synthetic-20261116-001/records.json'))
    args = parser.parse_args()

    runs = json.loads(args.input.read_text())['runs']
    true_supports = {r['data_sha256']: set(r['true_indices'])
                     for r in json.loads(args.optimizer_results.read_text())['runs']}
    if len(runs) != 3250 or len({r['data_sha256'] for r in runs}) != 3250:
        raise ValueError('Expected the complete 3250-matrix cloud comparison')
    dimensions = sorted({run['D'] for run in runs})
    corrupting_by_dimension = {
        dimension: sorted({
            run['corrupting_D'] for run in runs if run['D'] == dimension
        })
        for dimension in dimensions
    }
    block_counts = sorted({run['num_corrupting_matrices'] for run in runs})
    fig, axes = plt.subplots(
        len(block_counts), len(dimensions), figsize=(13.5, 11.5),
        sharey=True, layout='constrained')
    for column, dimension in enumerate(dimensions):
        corrupting_dimensions = corrupting_by_dimension[dimension]
        for row, block_count in enumerate(block_counts):
            ax = axes[row, column]
            for label, prefix, color, marker in METHODS:
                recovery_rates = []
                omitted_dimensions = []
                for corrupting_dimension in corrupting_dimensions:
                    subset = [
                        run for run in runs
                        if run['D'] == dimension
                        and run['corrupting_D'] == corrupting_dimension
                        and run['num_corrupting_matrices'] == block_count
                    ]
                    if not subset:
                        recovery_rates.append(np.nan)
                        continue
                    method_losses = np.array([
                        run[prefix + '_loss'] for run in subset
                    ], dtype=float)
                    available = np.isfinite(method_losses)
                    exact_recovery = np.array([
                        set(run[prefix + '_indices'] or []) == true_supports[run['data_sha256']]
                        for run in subset
                    ], dtype=bool)
                    if not available.all():
                        omitted_dimensions.append(corrupting_dimension)
                    recovery_rates.append(
                        100 * exact_recovery[available].mean()
                        if available.any() else np.nan
                    )
                ax.plot(corrupting_dimensions, recovery_rates,
                        color=color, marker=marker,
                        linewidth=2, markersize=5, label=label)
                if omitted_dimensions:
                    omitted = ', '.join(str(d) for d in omitted_dimensions)
                    ax.text(.03, .06, f'Backward timed out: $d_c={omitted}$',
                            transform=ax.transAxes, fontsize=6, color=color)

            if row == 0:
                ax.set_title(rf'$D={dimension}$', fontsize=14, pad=20)
            ax.text(.03, .92, rf'$b={block_count}$',
                    transform=ax.transAxes, ha='left', va='top', fontsize=11,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=2))
            ax.set_xscale('log', base=2)
            ax.set_xticks(
                corrupting_dimensions,
                labels=[str(value) for value in corrupting_dimensions],
            )
            ax.tick_params(axis='x', labelbottom=True)
            ax.tick_params(axis='y', labelleft=(column == 0))
            ax.grid(alpha=.25)

    axes[0, 0].set_ylim(-3, 103)
    axes[0, 0].set_yticks([0, 25, 50, 75, 100])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=4, frameon=False)
    fig.supxlabel(r'Corrupting dimension $d_c$')
    fig.supylabel('Exact support recovery (%)')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix('.png'), dpi=250)
    fig.savefig(args.output.with_suffix('.pdf'))
    plt.close(fig)


if __name__ == '__main__':
    main()
