"""Plot downloaded cloud results without needing an AWS session."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--condition', help='Dataset/condition label to select when a run contains several')
    args = parser.parse_args()
    records = json.loads((args.directory / 'records.json').read_text())['runs']
    def group(record):
        if record['dataset'] == 'synthetic':
            return f"synthetic D={record['D']}, dc={record['corrupting_D']}, blocks={record['num_corrupting_matrices']}"
        return record['dataset']
    datasets = sorted({group(r) for r in records})
    if args.condition:
        if args.condition not in datasets:
            parser.error('Unknown condition. Available: ' + '; '.join(datasets))
        records = [r for r in records if group(r) == args.condition]
        datasets = [args.condition]
    if len(datasets) != 1:
        parser.error('Use --condition to select one of: ' + '; '.join(datasets))
    figure, axis = plt.subplots(figsize=(8, 5.2))
    shapes = {True: 'o', False: 'x', None: 'D'}
    colormaps = {'scs': plt.get_cmap('GnBu'), 'admm': plt.get_cmap('OrRd')}
    parameters = sorted({r['settings'].get('scale', r['settings'].get('rho')) for r in records})
    shades = {value: .4 + .52 * index / max(1, len(parameters) - 1)
              for index, value in enumerate(parameters)}
    color_handles = []
    for solver, parameter in [(s, p) for s in ('scs', 'admm') for p in shades]:
        key = 'scale' if solver == 'scs' else 'rho'
        selected = [r for r in records if r['settings']['solver'] == solver
                    and r['settings'][key] == parameter and r.get('status', '').lower() == 'solved']
        if not selected:
            continue
        color = colormaps[solver](shades[parameter])
        label = f'{solver.upper()} {key}={parameter:g}'
        color_handles.append(Line2D([], [], marker='o', linestyle='none', color=color,
                                    markerfacecolor='none', markersize=8, label=label))
        for record in selected:
            marker = shapes[record.get('contains_true_basis')]
            style = ({'color': color} if marker in ('x', '+')
                     else {'facecolors': 'none', 'edgecolors': color})
            axis.scatter(record['support_cardinality'], record['objective_value'],
                         marker=marker, s=150 if solver == 'scs' else 75,
                         alpha=.3, linewidths=1.5, **style)
    statuses = {r.get('contains_true_basis') for r in records}
    labels = {True: 'Planted optimum retained', False: 'Planted optimum missed', None: 'Containment unknown'}
    shape_handles = [Line2D([], [], marker=shapes[state], linestyle='none', color='black',
                           markerfacecolor='none', markersize=7, label=labels[state])
                     for state in (True, False, None) if state in statuses]
    solved = sum(r.get('status', '').lower() == 'solved' for r in records)
    axis.set_title(f'{datasets[0]}: {solved}/{len(records)} solved')
    axis.set_xlabel('Retained support cardinality')
    axis.set_ylabel('Basis-pursuit objective')
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.ticklabel_format(axis='y', style='plain', useOffset=False)
    axis.margins(x=.12, y=.18)
    figure.legend(handles=color_handles + shape_handles, loc='lower center', ncol=3,
                  frameon=False, fontsize=9, bbox_to_anchor=(.5, .01))
    figure.tight_layout(rect=(0, .25, 1, 1))
    suffix = '_' + ''.join(c if c.isalnum() else '_' for c in args.condition) if args.condition else ''
    output = args.directory / f'optimizer_stability{suffix}.png'
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(output)


if __name__ == '__main__':
    main()
