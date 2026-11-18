"""Profile a private, high-resource version of the same optimizer Lambda."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
import os
import re
import time

from botocore.config import Config

from manage import ROOT, BUCKET, session, client

SETUP = ROOT / 'results/cloud/large-zero-smoke-setup'
FUNCTION = 'isometry-optimize'
RUN = 'large-zero-smoke-001'


def profile(s):
    c = client(s, 'lambda')
    before = c.get_function_configuration(FunctionName=FUNCTION)
    SETUP.mkdir(exist_ok=True)
    if (SETUP / 'profile.json').exists():
        saved = json.loads((SETUP / 'profile.json').read_text())
        if saved['CodeSha256'] == before['CodeSha256']:
            print(json.dumps(saved), flush=True)
            return
    print('Creating 3072 MB / 300 second version; the normal function will be restored.', flush=True)
    try:
        c.update_function_configuration(FunctionName=FUNCTION, MemorySize=3072, Timeout=300,
                                        RevisionId=before['RevisionId'])
        c.get_waiter('function_updated_v2').wait(FunctionName=FUNCTION, WaiterConfig={'Delay': 5, 'MaxAttempts': 60})
        version = c.publish_version(FunctionName=FUNCTION, CodeSha256=before['CodeSha256'],
                                    Description='Bounded zero-start convergence smoke test: 3 GB, 300 seconds')
        metadata = {k: version[k] for k in ('FunctionArn', 'Version', 'CodeSha256', 'MemorySize', 'Timeout')}
        (SETUP / 'profile.json').write_text(json.dumps(metadata, indent=2))
    finally:
        c.update_function_configuration(FunctionName=FUNCTION, MemorySize=before['MemorySize'], Timeout=before['Timeout'])
        c.get_waiter('function_updated_v2').wait(FunctionName=FUNCTION, WaiterConfig={'Delay': 5, 'MaxAttempts': 60})
    print(json.dumps(metadata), flush=True)


def run(s):
    metadata = json.loads((SETUP / 'profile.json').read_text())
    actual = client(s, 'lambda').get_function_configuration(FunctionName=FUNCTION, Qualifier=metadata['Version'])
    if any(actual[k] != metadata[k] for k in ('CodeSha256', 'MemorySize', 'Timeout')):
        raise RuntimeError('The published Lambda profile does not match')
    s3 = client(s, 's3')
    jobs = []
    for dataset in ('ethanol', 'synthetic'):
        body = s3.get_object(Bucket=BUCKET, Key=f'runs/zero-start-smoke-001/{dataset}/jobs.jsonl')['Body'].read()
        for job in map(json.loads, body.splitlines()):
            setting = job['settings']
            include = (dataset == 'synthetic' and job['sample']['D'] == 64)
            include |= (dataset == 'ethanol' and setting['solver'] == 'scs' and setting['scale'] in (.1, 10., 100.))
            if not include:
                continue
            job['run_prefix'] = f'runs/{RUN}/{dataset}'
            job['settings'] = {**setting, 'max_iters': 500000, 'time_limit_seconds': 240,
                               'lambda_profile': 'large-smoke', 'lambda_version': metadata['Version'],
                               'lambda_memory_mb': 3072, 'lambda_timeout_seconds': 300}
            jobs.append(job)
    if len(jobs) != 13 or any(j['settings']['initialization'] != 'zero' for j in jobs):
        raise RuntimeError('Unexpected smoke-test configuration')
    spec = importlib.util.spec_from_file_location('cloud_handlers', ROOT / 'infra/aws/handlers.py')
    handlers = importlib.util.module_from_spec(spec);spec.loader.exec_module(handlers)
    os.environ['LEDGER_TABLE'] = 'isometry-experiment-budget'
    # $0.02 covers 3 GB at the full 300-second timeout plus initialization and overhead.
    handlers.reserve(client(s, 'dynamodb'), RUN, len(jobs) * 20000)
    control = {'run_id': RUN, 'invocation': 'direct authenticated Lambda calls',
               'profile': metadata, 'reserved_usd': len(jobs) * .02, 'jobs': jobs}
    (SETUP / 'submission.json').write_text(json.dumps(control, indent=2))
    s3.put_object(Bucket=BUCKET, Key=f'runs/{RUN}/control.json', Body=json.dumps(control).encode())
    print(json.dumps({'jobs': len(jobs), 'reservation_usd': len(jobs)*.02, 'version': metadata['Version']}), flush=True)
    invoke = s.client('lambda', config=Config(connect_timeout=5, read_timeout=330,
                                             retries={'total_max_attempts': 1}, max_pool_connections=8))

    def one(job):
        started = time.perf_counter()
        response = invoke.invoke(FunctionName=metadata['FunctionArn'], InvocationType='RequestResponse',
                                 LogType='Tail', Payload=json.dumps(job).encode())
        payload = json.loads(response['Payload'].read())
        log = base64.b64decode(response.get('LogResult', '')).decode(errors='replace')
        billed = re.search(r'Billed Duration: ([\d.]+) ms', log)
        memory = re.search(r'Max Memory Used: ([\d.]+) MB', log)
        return {'dataset': job['sample']['dataset'], 'D': job['sample']['shape'][0],
                'P': job['sample']['shape'][1], 'job_id': job['job_id'],
                'solver': job['settings']['solver'],
                'parameter': job['settings'].get('scale', job['settings'].get('rho')),
                'response': payload, 'function_error': response.get('FunctionError'),
                'wall_seconds': time.perf_counter()-started,
                'billed_ms': float(billed.group(1)) if billed else None,
                'max_memory_mb': float(memory.group(1)) if memory else None}

    completed = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(one, job) for job in jobs]
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as error:
                result = {'invocation_error': type(error).__name__, 'message': str(error)}
            completed.append(result)
            (SETUP / 'invocations.json').write_text(json.dumps(completed, indent=2))
            print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['profile', 'run'])
    args = parser.parse_args()
    (profile if args.command == 'profile' else run)(session())
