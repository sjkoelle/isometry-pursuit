"""Launch the prepared, bounded large-case grid on an immutable workflow version."""
import copy
import hashlib
import importlib.util
import json
import os

from manage import ACCOUNT, BUCKET, ROOT, client, outputs, session


RUN = 'large-grid-20261116-001'
SETUP = ROOT / 'results/cloud/large-grid-setup'
RESERVATION = 13_100_000


def main():
    if (SETUP / 'execution.json').exists():
        raise RuntimeError('This batch was already submitted; inspect execution.json')
    config = json.loads((SETUP / 'config.json').read_text())
    jobs = len(config['synthetic']) * len(config['optimizers'])
    if jobs != 650 or config['searches'] or any(
        x['initialization'] != 'zero' or x['lambda_version'] != '1'
        or x['time_limit_seconds'] != 240 for x in config['optimizers']
    ):
        raise RuntimeError('Unexpected large-batch configuration')
    s = session()
    out = outputs(s)
    sf = client(s, 'stepfunctions')
    machine = out['StateMachineArn']
    if sf.list_executions(stateMachineArn=machine, statusFilter='RUNNING')['executions']:
        raise RuntimeError('Wait for the existing workflow before changing its routing')
    budget = client(s, 'budgets').describe_budget(
        AccountId=ACCOUNT, BudgetName='isometry-total-100-usd')['Budget']
    if float(budget['BudgetLimit']['Amount']) != 100:
        raise RuntimeError('Expected the existing $100 billing backstop')
    actual_spend = float(budget.get('CalculatedSpend', {}).get('ActualSpend', {}).get('Amount', 0))
    if actual_spend + RESERVATION / 1e6 > 75:
        raise RuntimeError('Insufficient billing headroom below the shutdown threshold')
    ledger = client(s, 'dynamodb').get_item(TableName=out['LedgerTable'],
        Key={'id': {'S': 'lifetime'}}, ConsistentRead=True)['Item']
    if ledger['locked']['BOOL'] or int(ledger['used_micro']['N']) + RESERVATION > 80_000_000:
        raise RuntimeError('Insufficient lifetime workload allowance')
    profile = json.loads((ROOT / 'results/cloud/large-zero-smoke-setup/profile.json').read_text())
    actual = client(s, 'lambda').get_function_configuration(
        FunctionName='isometry-optimize', Qualifier='1')
    if any(actual[k] != profile[k] for k in ('CodeSha256', 'MemorySize', 'Timeout')):
        raise RuntimeError('Published Lambda profile changed')
    original = sf.describe_state_machine(stateMachineArn=machine)
    definition = json.loads(original['definition'])
    large = copy.deepcopy(definition)
    task = large['States']['OptimizeSavedSamples']['ItemProcessor']['States']['OptimizeDataset']['ItemProcessor']['States']['OptimizeOne']
    task['Parameters']['FunctionName'] = profile['FunctionArn']
    validation = sf.validate_state_machine_definition(definition=json.dumps(large), type='STANDARD')
    if validation['result'] != 'OK':
        raise RuntimeError(validation['diagnostics'])
    raw = json.dumps(config, sort_keys=True).encode()
    key = 'configs/' + hashlib.sha256(raw).hexdigest() + '.json'
    client(s, 's3').put_object(Bucket=BUCKET, Key=key, Body=raw)
    # Stage one already reserves $0.0013/job + $0.10. Reserve the extra
    # high-resource exposure now, so the complete batch reserves $13.10.
    spec = importlib.util.spec_from_file_location('cloud_handlers', ROOT / 'infra/aws/handlers.py')
    handlers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(handlers)
    os.environ['LEDGER_TABLE'] = out['LedgerTable']
    extra = RESERVATION - jobs * 1300 - 100000
    handlers.reserve(client(s, 'dynamodb'), RUN + '-large-profile', extra)
    role_name = original['roleArn'].rsplit('/', 1)[-1]
    policy = {'Version': '2012-10-17', 'Statement': [
        {'Effect': 'Allow', 'Action': 'lambda:InvokeFunction', 'Resource': profile['FunctionArn']},
        {'Effect': 'Allow', 'Action': 'states:StartExecution', 'Resource': machine + ':*'},
    ]}
    client(s, 'iam').put_role_policy(RoleName=role_name,
        PolicyName='isometry-large-version-1', PolicyDocument=json.dumps(policy))
    control = dict(run_id=RUN, jobs=jobs, samples=65, reserved_usd=RESERVATION/1e6,
                   config_key=key, lambda_profile=profile, searches=False,
                   original_definition=definition, role_name=role_name)
    (SETUP / 'launch-control.json').write_text(json.dumps(control, indent=2) + '\n')
    # Publish an immutable workflow version; restore normal routing immediately.
    # The running execution remains pinned to the high-resource version.
    try:
        published = sf.update_state_machine(stateMachineArn=machine,
            definition=json.dumps(large), publish=True,
            versionDescription='Bounded D32/D64 grid: 650 zero-start solver jobs; 3 GB/300 s')
        version = published['stateMachineVersionArn']
        # Simulate the exact permissions needed before submitting any compute.
        sim = client(s, 'iam').simulate_principal_policy(
            PolicySourceArn=original['roleArn'],
            ActionNames=['lambda:InvokeFunction'], ResourceArns=[profile['FunctionArn']])
        if any(x['EvalDecision'] != 'allowed' for x in sim['EvaluationResults']):
            raise RuntimeError('High-resource invocation permission is not ready')
        response = sf.start_execution(stateMachineArn=version, name=RUN,
            input=json.dumps({'config_key': key, 'datasets': ['synthetic']}))
        result = {**control, 'state_machine_version': version,
                  'executionArn': response['executionArn'], 'startDate': str(response['startDate'])}
        (SETUP / 'execution.json').write_text(json.dumps(result, indent=2) + '\n')
        client(s, 's3').put_object(Bucket=BUCKET, Key=f'runs/{RUN}/control.json',
                                Body=json.dumps(result, default=str).encode())
        print(json.dumps({k: result[k] for k in ('executionArn', 'jobs', 'samples', 'reserved_usd')}))
    finally:
        sf.update_state_machine(stateMachineArn=machine, definition=original['definition'])


if __name__ == '__main__':
    main()
