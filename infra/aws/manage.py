"""Build/deploy the AWS workflow, submit paired runs, and download outputs.

Run from the repository root in the isometry_pursuit Conda environment.
"""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[2]
ACCOUNT = '339712941774'
REGION = 'us-west-2'
BUCKET = f'isometry-experiments-{ACCOUNT}-{REGION}'
STACK = 'isometry-experiments'


def session():
    result = boto3.Session(profile_name='isometry', region_name=REGION)
    identity = result.client('sts').get_caller_identity()
    if identity['Arn'] != f'arn:aws:iam::{ACCOUNT}:user/sam':
        raise RuntimeError('Expected the sam IAM identity')
    return result


def client(s, name):
    return s.client(name, config=Config(connect_timeout=5, read_timeout=30,
                                       retries={'max_attempts': 1}))


def build():
    target = Path(tempfile.mkdtemp(prefix='isometry-lambda-'))
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--target', str(target),
                    '--platform', 'manylinux_2_34_x86_64', '--platform', 'manylinux_2_28_x86_64',
                    '--platform', 'manylinux_2_27_x86_64', '--platform', 'manylinux_2_24_x86_64',
                    '--platform', 'manylinux2014_x86_64', '--python-version', '3.12',
                    '--implementation', 'cp', '--only-binary=:all:', '--no-compile',
                    '-r', str(ROOT / 'infra/aws/lambda-requirements.txt')], check=True)
    output = ROOT / 'results/cloud/lambda.zip'
    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(target.rglob('*')):
            if not path.is_file() or any(p in ('tests', '__pycache__') for p in path.parts):
                continue
            total += path.stat().st_size
            archive.write(path, str(path.relative_to(target)))
        for name in ('__init__.py', 'algorithm.py', 'simulation.py', 'transformation.py',
                     'experiment.py', 'loss.py', 'cloud_experiments.py', 'cloud_search.py'):
            archive.write(ROOT / 'isometry_pursuit' / name, 'isometry_pursuit/' + name)
        archive.write(ROOT / 'infra/aws/handlers.py', 'cloud_handlers.py')
    if total > 250 * 1024**2:
        raise RuntimeError('Lambda package exceeds the uncompressed size limit')
    print(json.dumps({'archive': str(output), 'uncompressed_bytes': total,
                      'compressed_bytes': output.stat().st_size}))


def deploy(s):
    spec = importlib.util.spec_from_file_location('aws_stack', ROOT / 'infra/aws/stack.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive = ROOT / 'results/cloud/lambda.zip'
    if not archive.exists():
        raise RuntimeError('Run build first')
    s3 = client(s, 's3')
    try:
        s3.head_bucket(Bucket=BUCKET)
    except ClientError as error:
        if error.response['Error']['Code'] not in ('404', 'NoSuchBucket'):
            raise
        s3.create_bucket(Bucket=BUCKET, CreateBucketConfiguration={'LocationConstraint': REGION})
    s3.put_public_access_block(Bucket=BUCKET, PublicAccessBlockConfiguration={
        'BlockPublicAcls': True, 'IgnorePublicAcls': True,
        'BlockPublicPolicy': True, 'RestrictPublicBuckets': True})
    s3.put_bucket_tagging(Bucket=BUCKET, Tagging={'TagSet': [{'Key': 'Project', 'Value': 'isometry-pursuit'}]})
    s3.put_bucket_lifecycle_configuration(Bucket=BUCKET, LifecycleConfiguration={'Rules': [{
        'ID': 'Expire-experiment-results-after-90-days', 'Status': 'Enabled',
        'Filter': {'Prefix': 'runs/'}, 'Expiration': {'Days': 90}}]})
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    key = f'code/{digest}.zip'
    s3.upload_file(str(archive), BUCKET, key)
    cf = client(s, 'cloudformation')
    parameters = [{'ParameterKey': k, 'ParameterValue': v} for k, v in
                  [('Bucket', BUCKET), ('CodeKey', key), ('CodeHash', digest)]]
    template = json.dumps(module.template())
    cf.validate_template(TemplateBody=template)
    options = dict(StackName=STACK, TemplateBody=template, Parameters=parameters,
                   Capabilities=['CAPABILITY_IAM'], Tags=[{'Key': 'Project', 'Value': 'isometry-pursuit'}])
    try:
        cf.describe_stacks(StackName=STACK)
    except ClientError as error:
        if 'does not exist' not in str(error):
            raise
        cf.create_stack(**options)
        print('Creating stack; run status to check progress.')
        return
    try:
        cf.update_stack(**options)
        print('Updating stack; run status to check progress.')
    except ClientError as error:
        if 'No updates are to be performed' not in str(error):
            raise
        print('Stack is unchanged.')


def outputs(s):
    stack = client(s, 'cloudformation').describe_stacks(StackName=STACK)['Stacks'][0]
    if stack['StackStatus'] not in ('CREATE_COMPLETE', 'UPDATE_COMPLETE'):
        raise RuntimeError('Stack is not ready: ' + stack['StackStatus'])
    return {v['OutputKey']: v['OutputValue'] for v in stack['Outputs']}


def budget(s):
    out = outputs(s)
    ddb = client(s, 'dynamodb')
    try:
        ddb.put_item(TableName=out['LedgerTable'], Item={
            'id': {'S': 'lifetime'}, 'used_micro': {'N': '0'}, 'locked': {'BOOL': False}},
            ConditionExpression='attribute_not_exists(id)')
    except ddb.exceptions.ConditionalCheckFailedException:
        pass  # Never reset an existing project's allowance.
    budgets = client(s, 'budgets')
    name = 'isometry-total-100-usd'
    try:
        existing = budgets.describe_budget(AccountId=ACCOUNT, BudgetName=name)['Budget']
        if float(existing['BudgetLimit']['Amount']) != 100:
            raise RuntimeError('Existing budget has a different limit')
    except budgets.exceptions.NotFoundException:
        now = datetime.now(timezone.utc)
        budgets.create_budget(AccountId=ACCOUNT, Budget={
            'BudgetName': name, 'BudgetType': 'COST', 'TimeUnit': 'CUSTOM',
            'TimePeriod': {'Start': now, 'End': now + timedelta(days=365)},
            'BudgetLimit': {'Amount': '100', 'Unit': 'USD'}}, NotificationsWithSubscribers=[{
                'Notification': {'NotificationType': 'ACTUAL', 'ComparisonOperator': 'GREATER_THAN',
                                 'Threshold': 75, 'ThresholdType': 'ABSOLUTE_VALUE'},
                'Subscribers': [{'SubscriptionType': 'SNS', 'Address': out['BudgetTopicArn']}]}])
    print('Lifetime workload allowance initialized at $80; $100 billing budget configured with shutdown at $75 actual charges.')


def prepare(s, full=False):
    s3 = client(s, 's3')
    real = []
    for name, filename in [('iris', 'iris_standardized_resampled_tensor.npy'),
                           ('wine', 'wine_standardized_resampled_tensor.npy'), ('ethanol', 'ethanol_tensor.npy')]:
        body = (ROOT / 'data' / filename).read_bytes()
        digest = hashlib.sha256(body).hexdigest()
        key = f'sources/{digest}/{filename}'
        s3.put_object(Bucket=BUCKET, Key=key, Body=body)
        real.append({'dataset': name, 'source_key': key, 'source_sha256': digest,
                     'replicates': list(range(25)) if full else [0]})
    runs = json.loads((ROOT / 'results/identity_swap_comparison_25x_D64/identity_experiment_runs.json').read_text())['runs']
    selected = runs if full else [next(r for r in runs if r['D'] == d) for d in (2, 4, 32, 64)]
    synthetic = [{**{k: r[k] for k in ('D', 'corrupting_D', 'num_corrupting_matrices', 'replicate')},
                  'data_seed': r['seed']} for r in selected]
    optimizers = []
    for solver, value in itertools.product(('scs', 'admm'), (0.01, 0.1, 1.0, 10.0, 100.0)):
        optimizers.append({'solver': solver, 'scale' if solver == 'scs' else 'rho': value,
                           'seed': 0, 'initialization': 'zero', 'search_seed': 1234,
                           'eps_abs': 1e-8, 'eps_rel': 1e-8, 'max_iters': 100000})
    configuration = {'real': real, 'synthetic': synthetic, 'optimizers': optimizers, 'searches': True}
    body = json.dumps(configuration, sort_keys=True).encode()
    key = 'configs/' + hashlib.sha256(body).hexdigest() + '.json'
    s3.put_object(Bucket=BUCKET, Key=key, Body=body)
    path = ROOT / 'results/cloud/submission.json'
    path.write_text(json.dumps({'config_key': key, 'datasets': ['iris', 'wine', 'ethanol', 'synthetic']}, indent=2))
    count = (sum(len(r['replicates']) for r in real) + len(synthetic)) * len(optimizers)
    print(json.dumps({'submission': str(path), 'jobs': count,
                      'worst_case_reservation_usd': count * .0013 + .4}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['build', 'deploy', 'status', 'budget', 'prepare', 'start', 'download'])
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--run-id')
    args = parser.parse_args()
    if args.command == 'build':
        build()
        return
    s = session()
    if args.command == 'deploy': deploy(s)
    elif args.command == 'budget': budget(s)
    elif args.command == 'prepare': prepare(s, full=args.full)
    elif args.command == 'status':
        print(json.dumps(client(s, 'cloudformation').describe_stacks(StackName=STACK)['Stacks'][0], default=str, indent=2))
    elif args.command == 'start':
        out = outputs(s)
        # Refuse to launch unless the billing backstop exists and the ledger is unlocked.
        client(s, 'budgets').describe_budget(AccountId=ACCOUNT, BudgetName='isometry-total-100-usd')
        ledger = client(s, 'dynamodb').get_item(TableName=out['LedgerTable'], Key={'id': {'S': 'lifetime'}}, ConsistentRead=True)['Item']
        if ledger['locked']['BOOL']: raise RuntimeError('Project budget is locked')
        submission = json.loads((ROOT / 'results/cloud/submission.json').read_text())
        configuration = json.loads(client(s, 's3').get_object(
            Bucket=BUCKET, Key=submission['config_key'])['Body'].read())
        if any(setting.get('initialization', 'zero') != 'zero' for setting in configuration['optimizers']):
            raise RuntimeError('Prepare a new configuration: optimizer initialization must be zero')
        samples = sum(len(r['replicates']) for r in configuration.get('real', [])) + len(configuration.get('synthetic', []))
        required_micro = samples * len(configuration['optimizers']) * 1300 + len(submission['datasets']) * 100000
        if int(ledger['used_micro']['N']) + required_micro > 80_000_000:
            raise RuntimeError('The complete batch would exceed the remaining lifetime workload allowance')
        run_id = args.run_id or datetime.now(timezone.utc).strftime('run-%Y%m%d-%H%M%S')
        print(json.dumps(client(s, 'stepfunctions').start_execution(
            stateMachineArn=out['StateMachineArn'], name=run_id, input=json.dumps(submission)), default=str))
    elif args.command == 'download':
        if not args.run_id: parser.error('--run-id is required')
        prefix = 'runs/' + args.run_id + '/'
        dest = ROOT / 'results/cloud' / args.run_id
        if dest.parent != ROOT / 'results/cloud': raise ValueError('Invalid run ID')
        s3 = client(s, 's3')
        for page in s3.get_paginator('list_objects_v2').paginate(Bucket=BUCKET, Prefix=prefix):
            for entry in page.get('Contents', []):
                relative = Path(entry['Key'][len(prefix):])
                if relative.is_absolute() or '..' in relative.parts: raise ValueError('Invalid S3 key')
                path = dest / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(BUCKET, entry['Key'], str(path))
        records = [json.loads(p.read_text()) for p in sorted(dest.glob('*/results/*.json'))]
        (dest / 'records.json').write_text(json.dumps({'runs': records}, indent=2))
        print(dest)


if __name__ == '__main__':
    main()
