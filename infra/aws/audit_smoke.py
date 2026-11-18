"""Check downloaded smoke-test coefficients and summarize convergence locally."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

from manage import ROOT
from isometry_pursuit.transformation import exponential_transformation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    folder = args.directory
    saved = json.loads((folder / 'records.json').read_text())
    records = saved['runs'] if isinstance(saved, dict) else saved
    tensor_names = {'iris': 'iris_standardized_resampled_tensor.npy',
                    'wine': 'wine_standardized_resampled_tensor.npy', 'ethanol': 'ethanol_tensor.npy'}
    errors, rows = [], []
    counts = defaultdict(list)
    for record in records:
        if record['dataset'] == 'synthetic':
            saved_matrix = folder / record['data_key']
            if not saved_matrix.exists():
                saved_matrix = ROOT / f"results/cloud/zero-start-smoke-setup/run{record['saved_run_id']}.npy"
            matrix = np.load(saved_matrix)
            condition = f"synthetic D={record['D']}, P={record['shape'][1]}"
        else:
            matrix = np.load(ROOT / 'data' / tensor_names[record['dataset']], mmap_mode='r')[:, :, record['replicate']]
            condition = record['dataset']
        stream = io.BytesIO();np.save(stream, matrix, allow_pickle=False)
        if hashlib.sha256(stream.getvalue()).hexdigest() != record['data_sha256']:
            errors.append(condition + ': sample hash mismatch')
        settings = record['settings']
        if settings['initialization'] != 'zero':
            errors.append(condition + ': initialization is not zero')
        if record['status'] == 'error':
            errors.append(condition + ': worker error: ' + record.get('error', ''))
            continue
        beta_path = folder / record['dataset'] / 'results' / (record['job_id'] + '_beta.npy')
        if hashlib.sha256(beta_path.read_bytes()).hexdigest() != record['beta_sha256']:
            errors.append(condition + ': coefficient hash mismatch')
        beta = np.load(beta_path, allow_pickle=False)
        if beta.shape != (matrix.shape[1], matrix.shape[0]) or not np.isfinite(beta).all():
            errors.append(condition + ': invalid coefficients')
            continue
        normalized = exponential_transformation(matrix)
        residual = float(np.linalg.norm(np.einsum('dp,pk->dk', normalized, beta) - np.eye(matrix.shape[0])))
        objective = float(np.linalg.norm(beta, axis=1).sum())
        if not math.isclose(objective, record['objective_value'], rel_tol=1e-10, abs_tol=1e-10):
            errors.append(condition + ': objective mismatch')
        if not math.isclose(residual, record['feasibility_residual_frobenius'], rel_tol=1e-6, abs_tol=1e-10):
            errors.append(condition + ': residual mismatch')
        thresholded = beta.copy()
        thresholded[np.abs(thresholded) < settings.get('entry_threshold', 1e-6)] = 0
        support = np.flatnonzero(np.linalg.norm(thresholded, axis=1) > settings.get('row_threshold', 1e-4)).tolist()
        if support != record['basis_pursuit_indices']:
            errors.append(condition + ': support mismatch')
        relative = residual / math.sqrt(matrix.shape[0])
        row = {'condition': condition, 'dataset': record['dataset'], 'D': matrix.shape[0], 'P': matrix.shape[1],
               'solver': settings['solver'], 'parameter': settings.get('scale', settings.get('rho')),
               'initialization': settings['initialization'], 'status': record['status'],
               'iterations': record['basis_pursuit_iterations'], 'seconds': record['basis_pursuit_time_seconds'],
               'objective': objective, 'support': len(support),
               'contains_true_basis': record.get('contains_true_basis'),
               'residual': residual, 'relative_residual': relative,
               'relative_objective_error': abs(objective - matrix.shape[0]) / matrix.shape[0] if record['dataset'] == 'synthetic' else None}
        counts[condition].append(row)
        rows.append(row)
    statuses = Counter(r['status'] for r in records)
    summary = {'records': len(records), 'integrity_passed': not errors, 'errors': errors,
               'statuses': dict(statuses), 'normal_convergence': statuses.get('solved', 0),
               'normal_convergence_with_relative_residual_above_1e_6': sum(
                   r['status'] == 'solved' and r['relative_residual'] > 1e-6 for r in rows),
               'conditions': {condition: {'jobs': len(group),
                   'statuses': dict(Counter(r['status'] for r in group)),
                   'maximum_seconds': max(r['seconds'] for r in group),
                   'minimum_seconds': min(r['seconds'] for r in group)} for condition, group in counts.items()}}
    (folder / 'audit.json').write_text(json.dumps(summary, indent=2))
    if rows:
        with (folder / 'audit.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
