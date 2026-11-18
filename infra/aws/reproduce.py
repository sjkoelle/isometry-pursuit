"""Reproduce one saved simulation before submitting a paired optimizer sweep."""

import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

from manage import ROOT, BUCKET, session, client

SETUP = ROOT / 'results/cloud/reproduction-setup'
BASELINE = 'reproduce-run78-baseline'
SWEEP = 'reproduce-run78-zero-sweep'


def prepare(stage):
    reference = json.loads((SETUP / 'reference.json').read_text())
    if stage == 'sweep':
        verified = json.loads((SETUP / 'baseline-verification.json').read_text())
        if not verified['passed']:
            raise RuntimeError('The baseline must reproduce before starting the sweep')
    s3 = client(session(), 's3')
    body = (SETUP / 'sample.npy').read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    source = f'sources/{digest}/simulation-run78.npy'
    s3.put_object(Bucket=BUCKET, Key=source, Body=body)
    saved = reference['saved_record']
    sample = {k: saved[k] for k in ('D', 'corrupting_D', 'num_corrupting_matrices', 'replicate')}
    sample.update(data_seed=saved['seed'], saved_run_id=saved['run_id'],
                  source_key=source, source_sha256=digest)
    settings = [reference['settings']]
    if stage == 'sweep':
        settings = []
        for solver, parameter in itertools.product(('scs', 'admm'), (.01, .1, 1., 10., 100.)):
            settings.append({'solver': solver, 'scale' if solver == 'scs' else 'rho': parameter,
                             'seed': 0, 'initialization': 'zero', 'search_seed': saved['seed'],
                             'search_on_raw': True, 'eps_abs': 1e-8, 'eps_rel': 1e-8,
                             'max_iters': 100000})
    config = {'synthetic': [sample], 'optimizers': settings, 'searches': True}
    raw = json.dumps(config, sort_keys=True).encode()
    key = 'configs/' + hashlib.sha256(raw).hexdigest() + '.json'
    s3.put_object(Bucket=BUCKET, Key=key, Body=raw)
    submission = {'config_key': key, 'datasets': ['synthetic']}
    (ROOT / 'results/cloud/submission.json').write_text(json.dumps(submission, indent=2))
    config_name = 'baseline' if stage == 'baseline' else 'zero-sweep'
    (SETUP / f'{config_name}-config.json').write_text(json.dumps(config, indent=2))
    print(json.dumps({'run_id': BASELINE if stage == 'baseline' else SWEEP,
                      'jobs': len(settings), 'reserved_usd': .1 + .0013 * len(settings),
                      'source_sha256': digest}))


def verify(stage, run_id=None):
    import numpy as np
    from isometry_pursuit.transformation import exponential_transformation

    name = run_id or (BASELINE if stage == 'baseline' else SWEEP)
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in name):
        raise ValueError('Invalid run ID')
    folder = ROOT / 'results/cloud' / name
    records = json.loads((folder / 'records.json').read_text())['runs']
    config = json.loads((folder / 'synthetic/config.json').read_text())
    reference = json.loads((SETUP / 'reference.json').read_text())
    saved = reference['saved_record']
    expected_hash = hashlib.sha256((SETUP / 'sample.npy').read_bytes()).hexdigest()
    errors = []
    matrix = np.load(SETUP / 'sample.npy', allow_pickle=False)
    normalized = exponential_transformation(matrix)
    if len(records) != len(config['optimizers']):
        errors.append('Unexpected number of completed records')
    if any(r['data_sha256'] != expected_hash for r in records):
        errors.append('Input matrix hash mismatch')
    for record in records:
        if record.get('status') == 'error':
            errors.append(record.get('error', 'worker error'))
            continue
        path = folder / 'synthetic/results' / (record['job_id'] + '_beta.npy')
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['beta_sha256']:
            errors.append('Coefficient file hash mismatch')
        beta = np.load(path, allow_pickle=False)
        if beta.shape != (matrix.shape[1], matrix.shape[0]) or not np.isfinite(beta).all():
            errors.append('Invalid coefficient matrix')
            continue
        measured_loss = float(np.linalg.norm(beta, axis=1).sum())
        measured_residual = float(np.linalg.norm(normalized @ beta - np.eye(matrix.shape[0])))
        if not math.isfinite(record['objective_value']) or abs(measured_loss - record['objective_value']) > 1e-10:
            errors.append('Reported objective does not match coefficients')
        if not math.isfinite(record['feasibility_residual_frobenius']) or abs(measured_residual - record['feasibility_residual_frobenius']) > 1e-10:
            errors.append('Reported residual does not match coefficients')
        if measured_residual > 1e-6:
            errors.append('Reconstruction residual exceeds 1e-6')
        thresholded = beta.copy()
        thresholded[np.abs(thresholded) < record['settings'].get('entry_threshold', 1e-6)] = 0
        support = np.flatnonzero(np.linalg.norm(thresholded, axis=1) > record['settings'].get('row_threshold', 1e-4)).tolist()
        if support != record['basis_pursuit_indices']:
            errors.append('Reported support does not match coefficients and thresholds')
        if any(record.get(prefix + '_status') != 'complete' for prefix in
               ('forward_full', 'forward_restricted', 'backward_full', 'backward_restricted')):
            errors.append('A downstream search did not complete')
    if stage == 'baseline' and records:
        record = records[0]
        if record['basis_pursuit_indices'] != saved['selected_indices']:
            errors.append('IP support differs from saved result')
        if record['basis_pursuit_iterations'] != saved['basis_pursuit_iterations']:
            errors.append('SCS iteration count differs from saved result')
        if abs(record['thresholded_objective_value'] - saved['objective_value']) > 1e-8:
            errors.append('IP objective differs from saved result')
        for prefix, old in [('forward_full', 'forward_backward_full'),
                            ('forward_restricted', 'forward_backward_restricted'),
                            ('backward_full', 'backward_swap_full'),
                            ('backward_restricted', 'backward_swap_restricted')]:
            if set(record[prefix + '_indices']) != set(saved[old + '_indices']):
                errors.append(prefix + ' support differs')
            if abs(record[prefix + '_loss'] - saved[old + '_loss']) > 1e-8:
                errors.append(prefix + ' loss differs')
        for prefix, old in [('forward_full', 'greedy_full'), ('forward_restricted', 'greedy_restricted')]:
            if set(record[prefix + '_initial_indices']) != set(saved[old + '_indices']):
                errors.append(prefix + ' initial greedy support differs')
            if abs(record[prefix + '_initial_loss'] - saved[old + '_loss']) > 1e-8:
                errors.append(prefix + ' initial greedy loss differs')
    if stage == 'sweep':
        combinations = {(r['settings']['solver'], r['settings'].get('scale', r['settings'].get('rho')), r['settings']['seed']) for r in records}
        expected = {(s['solver'], s.get('scale', s.get('rho')), s['seed']) for s in config['optimizers']}
        if combinations != expected:
            errors.append('The optimizer parameter/seed grid is incomplete')
        if sorted(json.dumps(r['settings'], sort_keys=True) for r in records) != sorted(json.dumps(s, sort_keys=True) for s in config['optimizers']):
            errors.append('Recorded solver settings differ from the submitted configuration')
    summary = {'passed': not errors, 'errors': errors, 'records': len(records),
               'data_sha256': expected_hash, 'note': 'Baseline checks reproduce saved results; sweep checks establish pipeline integrity, not universal solver guarantees.'}
    summary['statuses'] = dict(Counter(r.get('status') for r in records))
    summary['maximum_residual'] = max((r.get('feasibility_residual_frobenius', float('inf')) for r in records), default=None)
    verification_name = 'baseline' if stage == 'baseline' else name
    (SETUP / f'{verification_name}-verification.json').write_text(json.dumps(summary, indent=2))
    rows = []
    for record in records:
        settings = record['settings']
        row = {'solver': settings['solver'], 'parameter': settings.get('scale', settings.get('rho')),
               'optimizer_seed': settings['seed'], 'data_seed': record['data_seed'],
               'status': record['status'], 'iterations': record.get('basis_pursuit_iterations'),
               'support_size': record.get('support_cardinality'), 'contains_true_basis': record.get('contains_true_basis'),
               'support': json.dumps(record.get('basis_pursuit_indices')),
               'loss': record.get('objective_value'), 'residual': record.get('feasibility_residual_frobenius'),
               'forward_full_loss': record.get('forward_full_loss'),
               'forward_restricted_loss': record.get('forward_restricted_loss'),
               'backward_full_loss': record.get('backward_full_loss'),
               'backward_restricted_loss': record.get('backward_restricted_loss'),
               'solve_seconds': record.get('basis_pursuit_time_seconds')}
        rows.append(row)
    if rows:
        with (folder / 'comparison.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'verify'])
    parser.add_argument('stage', choices=['baseline', 'sweep'])
    parser.add_argument('--run-id', help='Verify a specific completed execution, including historical runs')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.stage)
    else:
        verify(args.stage, run_id=args.run_id)
