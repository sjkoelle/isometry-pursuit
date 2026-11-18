"""Prepare a bounded zero-start smoke test across six frozen samples."""

import hashlib
import io
import itertools
import json

import numpy as np

from manage import ROOT, BUCKET, session, client
from isometry_pursuit.simulation import generate_identity_instance


def main():
    folder = ROOT / 'results/cloud/zero-start-smoke-setup'
    folder.mkdir(exist_ok=True)
    s3 = client(session(), 's3')

    def freeze(body, filename):
        digest = hashlib.sha256(body).hexdigest()
        key = f'sources/{digest}/{filename}'
        s3.put_object(Bucket=BUCKET, Key=key, Body=body)
        return {'source_key': key, 'source_sha256': digest}

    real = []
    for name, filename in [('iris', 'iris_standardized_resampled_tensor.npy'),
                           ('wine', 'wine_standardized_resampled_tensor.npy'), ('ethanol', 'ethanol_tensor.npy')]:
        source = freeze((ROOT / 'data' / filename).read_bytes(), filename)
        real.append({'dataset': name, 'replicates': [0], **source})
    runs = json.loads((ROOT / 'results/identity_swap_comparison_25x_D64/identity_experiment_runs.json').read_text())['runs']
    selected = [next(r for r in runs if r['run_id'] == 78),
                next(r for r in runs if r['D'] == 16 and r['corrupting_D'] == 8
                     and r['num_corrupting_matrices'] == 8 and r['replicate'] == 0),
                max(runs, key=lambda r: (r['D'], r['corrupting_D'], r['num_corrupting_matrices'], -r['replicate']))]
    synthetic = []
    for record in selected:
        matrix, *_ = generate_identity_instance(record['D'], record['corrupting_D'],
                                                record['num_corrupting_matrices'],
                                                np.random.default_rng(record['seed']))
        stream = io.BytesIO()
        np.save(stream, matrix, allow_pickle=False)
        body = stream.getvalue()
        (folder / f"run{record['run_id']}.npy").write_bytes(body)
        source = freeze(body, f"simulation-run{record['run_id']}.npy")
        synthetic.append({**{k: record[k] for k in ('D', 'corrupting_D', 'num_corrupting_matrices', 'replicate')},
                          'data_seed': record['seed'], 'saved_run_id': record['run_id'], **source})
    settings = [dict(solver=solver, **{'scale' if solver == 'scs' else 'rho': parameter},
                     initialization='zero', seed=0, search_seed=1234,
                     eps_abs=1e-8, eps_rel=1e-8, max_iters=100000)
                for solver, parameter in itertools.product(('scs', 'admm'), (.01, .1, 1., 10., 100.))]
    config = {'real': real, 'synthetic': synthetic, 'optimizers': settings, 'searches': False}
    raw = json.dumps(config, sort_keys=True).encode()
    key = 'configs/' + hashlib.sha256(raw).hexdigest() + '.json'
    s3.put_object(Bucket=BUCKET, Key=key, Body=raw)
    submission = {'config_key': key, 'datasets': ['iris', 'wine', 'ethanol', 'synthetic']}
    (folder / 'config.json').write_text(json.dumps(config, indent=2))
    (ROOT / 'results/cloud/submission.json').write_text(json.dumps(submission, indent=2))
    print(json.dumps({'samples': 6, 'jobs': 60, 'reservation_usd': .478,
                      'searches': False, 'saved_simulation_runs': [r['run_id'] for r in selected]}))


if __name__ == '__main__':
    main()
