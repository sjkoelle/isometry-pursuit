"""Run all four swap methods using the IP supports from an existing tensor run."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from isometry_pursuit.experiment import compare_swap_initializations
from isometry_pursuit.transformation import exponential_transformation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ip-results', type=Path, required=True)
    parser.add_argument('--output-path', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    source = json.loads(args.ip_results.read_text())
    metadata = source['metadata']
    data_file = metadata['data_file']
    tensor = np.load(data_file)
    if list(tensor.shape) != metadata['data_tensor_shape']:
        raise ValueError('Source result tensor shape does not match data')
    replicates = source['replicates']
    if len(replicates) != tensor.shape[2]:
        raise ValueError('Source results must contain all tensor replicates')
    output = dict(metadata=dict(data_file=data_file, data_tensor_shape=list(tensor.shape),
                               ip_results=str(args.ip_results), seed=args.seed,
                               ip_solver_metadata=metadata,
                               timing='Initialization and swaps; excludes cached IP solve',
                               ip_search='Restricted to IP support',
                               max_swap_rounds='Dictionary size'), replicates=[])
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    for i, record in enumerate(replicates):
        if record['replicate'] != i:
            raise ValueError('Source replicates must be ordered by replicate index')
        X = exponential_transformation(tensor[:, :, i])
        methods = compare_swap_initializations(X, record['basis_pursuit_indices'],
                                               random_state=args.seed + i)
        output['replicates'].append(dict(replicate=i, methods=methods))
        # Checkpoint completed replicates so long experiments remain inspectable.
        args.output_path.write_text(json.dumps(output, indent=2) + '\n')
        print(f'{args.ip_results.stem} {i+1}/{len(replicates)} ' +
              ' '.join(f'{name}={m["loss"]:.6g}' if m['loss'] is not None else
                       f'{name}=unavailable' for name, m in methods.items()), flush=True)
    rows = []
    for name in output['replicates'][0]['methods']:
        valid = [r['methods'][name] for r in output['replicates']
                 if r['methods'][name]['status'] == 'ok']
        rows.append(dict(method=name, valid_replicates=len(valid),
                         mean_loss=float(np.mean([m['loss'] for m in valid])) if valid else None,
                         sd_loss=float(np.std([m['loss'] for m in valid])) if valid else None,
                         improved_by_swap=sum(m['loss'] < m['initial_loss'] - 1e-9 for m in valid),
                         mean_time_seconds=float(np.mean([m['time_seconds'] for m in valid])) if valid else None))
    output['summary'] = rows
    args.output_path.write_text(json.dumps(output, indent=2) + '\n')
    with args.output_path.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__':
    main()
