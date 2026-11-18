"""Complete the 25-replicate synthetic grid at fixed SCS scale 0.1."""
import argparse
import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from manage import ACCOUNT, BUCKET, ROOT, client, outputs, session
from search_grid import reserve


RUN = 'full-scs01-20261116-001'
SETUP = ROOT/'results/cloud/full-scs01-setup'
RESERVATION = 51_060_000


def prepare():
    original = json.loads((ROOT/'results/identity_swap_comparison_25x_D64/identity_experiment_runs.json').read_text())['runs']
    large = json.loads((ROOT/'results/cloud/large-grid-20261116-001/records.json').read_text())
    large = large['runs'] if isinstance(large, dict) else large
    existing = [r for r in large if r['settings']['solver']=='scs' and r['settings']['scale']==.1]
    assert len(existing)==65 and all(r['status']=='solved' for r in existing)
    completed = {r['saved_run_id'] for r in existing}
    missing = [r for r in original if r['run_id'] not in completed]
    assert len(missing)==3185
    setting = dict(existing[0]['settings'])
    config = dict(real=[], synthetic=[dict(D=r['D'], corrupting_D=r['corrupting_D'],
        num_corrupting_matrices=r['num_corrupting_matrices'], replicate=r['replicate'],
        data_seed=r['seed'], saved_run_id=r['run_id']) for r in missing],
        optimizers=[setting], searches=False)
    SETUP.mkdir(parents=True, exist_ok=True)
    (SETUP/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    plan = dict(total_samples=3250, reused_samples=65, new_samples=3185,
                replicates_per_condition=25, conditions=130, solver='scs', scale=.1,
                optimizer_reservation_usd=RESERVATION/1e6,
                search_allocation_usd=8, total_existing_and_planned_usd=79.9665)
    (SETUP/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(plan))


def start(s):
    if (SETUP/'execution.json').exists():
        raise RuntimeError('Already submitted; inspect the saved execution')
    config=json.loads((SETUP/'config.json').read_text())
    assert len(config['synthetic'])==3185 and len(config['optimizers'])==1
    old=json.loads((ROOT/'results/cloud/large-grid-setup/execution.json').read_text())
    version=old['state_machine_version']
    profile=client(s,'lambda').get_function_configuration(FunctionName='isometry-optimize',Qualifier='1')
    assert all(profile[k]==old['lambda_profile'][k] for k in ['CodeSha256','MemorySize','Timeout'])
    out=outputs(s);sf=client(s,'stepfunctions')
    if sf.list_executions(stateMachineArn=out['StateMachineArn'],statusFilter='RUNNING')['executions']:
        raise RuntimeError('Another workflow is running')
    expected=out['StateMachineArn'].replace(':stateMachine:',':execution:')+':'+RUN
    try:
        sf.describe_execution(executionArn=expected)
    except sf.exceptions.ExecutionDoesNotExist:
        pass
    else:
        raise RuntimeError('Execution already exists')
    budget=client(s,'budgets').describe_budget(AccountId=ACCOUNT,BudgetName='isometry-total-100-usd')['Budget']
    actual=float(budget.get('CalculatedSpend',{}).get('ActualSpend',{}).get('Amount',0))
    if float(budget['BudgetLimit']['Amount'])!=100 or actual+RESERVATION/1e6+8>75:
        raise RuntimeError('Insufficient billing headroom')
    ledger=client(s,'dynamodb').get_item(TableName=out['LedgerTable'],Key={'id':{'S':'lifetime'}},ConsistentRead=True)['Item']
    if ledger['locked']['BOOL'] or int(ledger['used_micro']['N'])+RESERVATION+8_000_000>80_000_000:
        raise RuntimeError('Insufficient lifetime allowance for optimizer and searches')
    raw=json.dumps(config,sort_keys=True).encode();key='configs/'+hashlib.sha256(raw).hexdigest()+'.json'
    client(s,'s3').put_object(Bucket=BUCKET,Key=key,Body=raw)
    # Generation reserves 1300 microdollars/job plus $0.10 separately.
    extra=RESERVATION-len(config['synthetic'])*1300-100000
    reserve(s,RUN+'-high-resource',extra)
    response=sf.start_execution(stateMachineArn=version,name=RUN,
        input=json.dumps({'config_key':key,'datasets':['synthetic']}))
    control=dict(run_id=RUN,executionArn=response['executionArn'],startDate=str(response['startDate']),
                 config_key=key,lambda_profile=old['lambda_profile'],new_samples=3185,
                 reserved_usd=RESERVATION/1e6,state_machine_version=version)
    (SETUP/'execution.json').write_text(json.dumps(control,indent=2)+'\n')
    client(s,'s3').put_object(Bucket=BUCKET,Key=f'runs/{RUN}/control.json',Body=json.dumps(control).encode())
    print(json.dumps(control))


def status(s):
    c=json.loads((SETUP/'execution.json').read_text());sf=client(s,'stepfunctions')
    e=sf.describe_execution(executionArn=c['executionArn'])
    maps=[sf.describe_map_run(mapRunArn=m['mapRunArn'])['itemCounts']
          for m in sf.list_map_runs(executionArn=c['executionArn'])['mapRuns']]
    print(json.dumps(dict(status=e['status'],maps=maps)))


def download(s):
    s3=client(s,'s3');folder=ROOT/'results/cloud'/RUN;folder.mkdir(exist_ok=True)
    keys=[o['Key'] for page in s3.get_paginator('list_objects_v2').paginate(
        Bucket=BUCKET,Prefix=f'runs/{RUN}/synthetic/results/') for o in page.get('Contents',[])
        if o['Key'].endswith('.json')]
    def read(key):
        return json.loads(s3.get_object(Bucket=BUCKET,Key=key)['Body'].read())
    with ThreadPoolExecutor(max_workers=8) as pool:
        records=list(pool.map(read,keys))
    (folder/'records.json').write_text(json.dumps({'runs':records},indent=2)+'\n')
    print(json.dumps(dict(downloaded=len(records),directory=str(folder))))


def assets(s):
    folder=ROOT/'results/cloud'/RUN
    records=json.loads((folder/'records.json').read_text())['runs']
    assert len(records)==3185
    files={}
    for r in records:
        files[r['data_key']]=(folder/r['data_key'],r['data_sha256'])
        files[r['beta_key']]=(folder/r['dataset']/'results'/(r['job_id']+'_beta.npy'),r['beta_sha256'])
    s3=client(s,'s3')
    def fetch(item):
        key,(path,digest)=item
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==digest:
            return
        body=s3.get_object(Bucket=BUCKET,Key=key)['Body'].read()
        if hashlib.sha256(body).hexdigest()!=digest:
            raise RuntimeError('Asset hash mismatch: '+key)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(body)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch,files.items()))
    print(json.dumps(dict(verified_assets=len(files),directory=str(folder))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','start','status','download','assets'])
    a=p.parse_args()
    if a.command=='prepare':
        prepare()
    else:
        globals()[a.command](session())
