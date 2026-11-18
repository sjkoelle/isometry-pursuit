"""Plot support agreement with reported experiments from saved solver runs."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_optimizer_support_stability(smoke_dir, extended_dir, results_dir, output_dir):
    smoke_dir, extended_dir = Path(smoke_dir), Path(extended_dir)
    results_dir, output_dir = Path(results_dir), Path(output_dir)
    records = json.loads((smoke_dir / 'records.json').read_text())['runs']
    extended = json.loads((extended_dir / 'records.json').read_text())['runs']
    parameters = [.01, .1, 1., 10., 100.]
    colormaps = {'scs': plt.get_cmap('GnBu'), 'admm': plt.get_cmap('OrRd')}
    figure, axes = plt.subplots(1, 3, figsize=(8.2, 3.2), sharey=True)
    summaries = {}
    for axis, dataset, title in zip(axes, ('iris', 'wine', 'ethanol'), ('Iris', 'Wine', 'Ethanol')):
        original = json.loads((results_dir / f'{dataset}_results.json').read_text())['replicates'][0]
        reference = set(original['basis_pursuit_indices'])
        selected = [r for r in records if r['dataset'] == dataset]
        assert len(selected) == 10
        assert all(r['replicate'] == 0 and r['settings']['initialization'] == 'zero' for r in selected)
        assert {(r['settings']['solver'], r['settings'].get('scale', r['settings'].get('rho'))) for r in selected} == {
            (solver, p) for solver in ('scs', 'admm') for p in parameters}

        def overlap(record):
            support = set(record['basis_pursuit_indices'])
            return len(support & reference) / len(support | reference)

        initial = []
        for record in selected:
            settings = record['settings']
            solver = settings['solver']
            parameter = settings.get('scale', settings.get('rho'))
            color = colormaps[solver](.4 + .52 * parameters.index(parameter) / 4)
            same = set(record['basis_pursuit_indices']) == reference
            marker = 'o' if same else 'x'
            style = {'facecolors': 'none', 'edgecolors': color} if same else {'color': color}
            axis.scatter(parameter, overlap(record), marker=marker,
                         s=70 if solver == 'scs' else 28, linewidths=1.2, alpha=.8, **style)
            initial.append({'solver': solver, 'parameter': parameter, 'overlap': overlap(record),
                            'support': record['basis_pursuit_indices'], 'status': record['status']})
        reruns = [r for r in extended if r['dataset'] == dataset]
        for record in reruns:
            parameter = record['settings']['scale']
            earlier = next(r for r in selected if r['settings']['solver'] == 'scs' and r['settings']['scale'] == parameter)
            assert record['data_sha256'] == earlier['data_sha256']
            if overlap(record) != overlap(earlier):
                color = colormaps['scs'](.4 + .52 * parameters.index(parameter) / 4)
                axis.annotate('', xy=(parameter, overlap(record)), xytext=(parameter, overlap(earlier)),
                              arrowprops={'arrowstyle': '->', 'color': color, 'lw': 1.1})
                axis.scatter(parameter, overlap(record), marker='o', s=70,
                             facecolors='none', edgecolors=color, linewidths=1.2)
                axis.annotate('500k iterations', xy=(parameter, overlap(record)),
                              xytext=(-4, 12), textcoords='offset points', ha='right', fontsize=7)
        latest = {(r['settings']['solver'], r['settings'].get('scale', r['settings'].get('rho'))): r for r in selected}
        latest.update({(r['settings']['solver'], r['settings']['scale']): r for r in reruns})
        summaries[dataset] = {
            'replicate': 0, 'reported_support_one_based': sorted(p + 1 for p in reference),
            'initial_matches': sum(set(r['basis_pursuit_indices']) == reference for r in selected),
            'latest_matches': sum(set(r['basis_pursuit_indices']) == reference for r in latest.values()),
            'settings': len(selected), 'initial_results': initial,
            'extended_results': [{'solver': r['settings']['solver'], 'parameter': r['settings']['scale'],
                                  'overlap': overlap(r), 'status': r['status'],
                                  'max_iters': r['settings']['max_iters']} for r in reruns],
        }
        axis.set_xscale('log')
        axis.set_xticks(parameters, labels=['0.01', '0.1', '1', '10', '100'])
        axis.tick_params(labelsize=8)
        axis.set_yticks([.75, 1.])
        axis.set_ylim(.68, 1.11)
        axis.set_xlim(.006, 170)
        axis.set_title(f'{title}: {len(reference)} reported columns', fontsize=10)
        axis.axhline(1., color='.85', lw=.7, zorder=0)
        axis.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Overlap with reported support', fontsize=9)
    figure.supxlabel(r'SCS scale / ADMM $\rho$', y=.19, fontsize=9)
    handles = [Line2D([], [], marker='o', markerfacecolor='none', linestyle='none', color=colormaps['scs'](.8), label='SCS'),
               Line2D([], [], marker='o', markerfacecolor='none', linestyle='none', color=colormaps['admm'](.8), label='Direct ADMM'),
               Line2D([], [], marker='o', markerfacecolor='none', linestyle='none', color='black', label='Same support'),
               Line2D([], [], marker='x', linestyle='none', color='black', label='Different support')]
    figure.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(.5, .01))
    figure.tight_layout(rect=(0, .23, 1, 1))
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ('pdf', 'png'):
        figure.savefig(output_dir / f'optimizer_support_stability.{suffix}', dpi=180)
    plt.close(figure)
    (smoke_dir / 'paper_stability_summary.json').write_text(json.dumps(summaries, indent=2))
    return summaries
