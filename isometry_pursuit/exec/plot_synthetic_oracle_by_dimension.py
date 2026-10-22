"""Plot synthetic oracle regret by target and corrupting dimensions."""
import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'matplotlib'))
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
        default=Path('results/identity_swap_comparison_25x_D64/identity_experiment_runs.json'))
    parser.add_argument(
        '--output', type=Path, default=Path('figures/synthetic_regret_grid'))
    parser.add_argument('--tie-tolerance', type=float, default=1e-9)
    args = parser.parse_args()

    runs = json.loads(args.input.read_text())['runs']
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
    maximum_regret = 0.0

    for column, dimension in enumerate(dimensions):
        corrupting_dimensions = corrupting_by_dimension[dimension]
        for row, block_count in enumerate(block_counts):
            ax = axes[row, column]
            for label, prefix, color, marker in METHODS:
                regrets = []
                omitted_dimensions = []
                for corrupting_dimension in corrupting_dimensions:
                    subset = [
                        run for run in runs
                        if run['D'] == dimension
                        and run['corrupting_D'] == corrupting_dimension
                        and run['num_corrupting_matrices'] == block_count
                    ]
                    if not subset:
                        regrets.append(np.nan)
                        continue
                    losses = np.array([
                        [run[method_prefix + '_loss'] for _, method_prefix, _, _
                         in METHODS]
                        for run in subset
                    ], dtype=float)
                    oracle = np.nanmin(losses, axis=1)
                    method_loss = np.array([
                        run[prefix + '_loss'] for run in subset
                    ], dtype=float)
                    errors = [run.get(prefix + '_error') for run in subset]
                    if any(
                        isinstance(error, str) and error.startswith('Skipped:')
                        for error in errors
                    ):
                        omitted_dimensions.append(corrupting_dimension)
                    method_regret = method_loss - oracle
                    method_regret[np.abs(method_regret) <= args.tie_tolerance] = 0
                    regrets.append(method_regret.mean())
                finite_regrets = [value for value in regrets if np.isfinite(value)]
                if finite_regrets:
                    maximum_regret = max(maximum_regret, max(finite_regrets))
                ax.plot(corrupting_dimensions, regrets, color=color, marker=marker,
                        linewidth=2, markersize=5, label=label)
                if omitted_dimensions:
                    omission_regrets = [
                        1e2 if value in omitted_dimensions else regret
                        for value, regret in zip(corrupting_dimensions, regrets)
                    ]
                    ax.plot(
                        corrupting_dimensions, omission_regrets,
                        color=color, linewidth=1.5, linestyle='--', alpha=.25,
                    )

            if row == 0:
                ax.set_title(rf'$D={dimension}$', fontsize=14, pad=20)
            ax.text(.03, .92, rf'$b={block_count}$',
                    transform=ax.transAxes, ha='left', va='top', fontsize=11,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=2))
            ax.set_xscale('log', base=2)
            ax.set_yscale('symlog', linthresh=1e-3, linscale=.7)
            ax.set_xticks(
                corrupting_dimensions,
                labels=[str(value) for value in corrupting_dimensions],
            )
            ax.tick_params(axis='x', labelbottom=True)
            ax.tick_params(axis='y', labelleft=(column == 0))
            ax.grid(alpha=.25)

    axes[0, 0].set_ylim(-3e-4, 1.12 * maximum_regret)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Line2D(
        [], [], color='gray', linestyle='--', alpha=.5,
        label='Not run (budget)',
    ))
    labels.append('Not run (budget)')
    fig.legend(handles, labels, loc='outside upper center', ncol=5, frameon=False)
    fig.supxlabel(r'Corrupting dimension $d_c$')
    fig.supylabel('Mean oracle regret')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix('.png'), dpi=250)
    fig.savefig(args.output.with_suffix('.pdf'))
    plt.close(fig)


if __name__ == '__main__':
    main()
