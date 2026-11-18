"""Smoke-test, launch, and download the search-only follow-up to the large grid."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from manage import ACCOUNT, BUCKET, ROOT, client, outputs, session


SETUP = ROOT/'results/cloud/large-search-setup'
RUN = 'large-search-20261116-001'


def reserve(s, request, amount):
    spec = importlib.util.spec_from_file_location('cloud_handlers',ROOT/'infra/aws/handlers.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    os.environ['LEDGER_TABLE']='isometry-experiment-budget'
    module.reserve(client(s,'dynamodb'), request, amount)


def smoke(s, cardinality_only=False):
    jobs=[json.loads(line) for line in (SETUP/'jobs.jsonl').read_text().splitlines()][:2]
    reserve(s,RUN+'-smoke',120000)
    c=client(s,'lambda');results=[]
    for job in jobs:
        event={**job,'run_prefix':'runs/'+RUN+'-smoke/synthetic'}
        if cardinality_only:
            event['candidate_indices']=list(range(job['sample']['D']))
        response=c.invoke(FunctionName='isometry-search',InvocationType='RequestResponse',Payload=json.dumps(event).encode())
        result=json.loads(response['Payload'].read())
        if response.get('FunctionError') or result.get('status')!='complete':
            raise RuntimeError(result)
        body=json.loads(client(s,'s3').get_object(Bucket=BUCKET,Key=result['result_key'])['Body'].read())
        if body['loss'] is None or len(body['indices'])!=body['D']:
            raise RuntimeError('Invalid cloud search result')
        results.append(body)
    (SETUP/'smoke.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps([dict(job_id=r['job_id'],status=r['status'],loss=r['loss'],seconds=r['total_runtime_seconds']) for r in results]))


def start(s, expected_jobs=278):
    if (SETUP/'execution.json').exists():raise RuntimeError('Batch already submitted')
    if not (SETUP/'smoke.json').exists():raise RuntimeError('Run the cloud smoke test first')
    out=outputs(s);sf=client(s,'stepfunctions')
    config=client(s,'lambda').get_function_configuration(FunctionName='isometry-search')
    if (config['MemorySize'],config['Timeout'],config['Handler'])!=(3072,900,'cloud_handlers.search'):
        raise RuntimeError('Unexpected search Lambda profile')
    if client(s,'lambda').list_function_url_configs(FunctionName='isometry-search')['FunctionUrlConfigs']:
        raise RuntimeError('Search Lambda should have no public URL')
    budget=client(s,'budgets').describe_budget(AccountId=ACCOUNT,BudgetName='isometry-total-100-usd')['Budget']
    if float(budget['BudgetLimit']['Amount'])!=100:raise RuntimeError('Expected $100 budget')
    body=(SETUP/'jobs.jsonl').read_bytes();jobs=body.splitlines()
    if len(jobs)!=expected_jobs:raise RuntimeError('Unexpected search job count')
    aliases=(SETUP/'aliases.json').read_bytes()
    ddb=client(s,'dynamodb')
    ledger=ddb.get_item(TableName='isometry-experiment-budget',
        Key={'id':{'S':'lifetime'}},ConsistentRead=True)['Item']
    if ledger['locked']['BOOL']:raise RuntimeError('Project budget is locked')
    expected_execution=out['StateMachineArn'].replace(':stateMachine:',':execution:')+':'+RUN
    try:
        sf.describe_execution(executionArn=expected_execution)
    except sf.exceptions.ExecutionDoesNotExist:
        pass
    else:
        raise RuntimeError('This execution name already exists; inspect it instead of resubmitting')
    prior=ddb.get_item(TableName='isometry-experiment-budget',
        Key={'id':{'S':'reservation/'+RUN}},ConsistentRead=True).get('Item')
    if prior:
        if int(prior['reserved_micro']['N'])!=len(jobs)*60000:
            raise RuntimeError('Existing reservation has a different amount')
        print('Reusing the existing reservation; no execution has been submitted.',flush=True)
    else:
        reserve(s,RUN,len(jobs)*60000)
    prefix='runs/'+RUN
    s3=client(s,'s3');key=prefix+'/search-jobs.jsonl'
    s3.put_object(Bucket=BUCKET,Key=key,Body=body)
    s3.put_object(Bucket=BUCKET,Key=prefix+'/aliases.json',Body=aliases)
    submission={'search_jobs_key':key,'run_prefix':prefix}
    result=sf.start_execution(stateMachineArn=out['StateMachineArn'],name=RUN,input=json.dumps(submission))
    control=dict(run_id=RUN,jobs=len(jobs),reserved_usd=len(jobs)*.06,
                 executionArn=result['executionArn'],started=str(result['startDate']),
                 function_code_sha256=config['CodeSha256'],submission=submission)
    (SETUP/'execution.json').write_text(json.dumps(control,indent=2)+'\n')
    s3.put_object(Bucket=BUCKET,Key=prefix+'/control.json',Body=json.dumps(control).encode())
    print(json.dumps(control))


def status(s):
    control=json.loads((SETUP/'execution.json').read_text());sf=client(s,'stepfunctions')
    execution=sf.describe_execution(executionArn=control['executionArn'])
    maps=[sf.describe_map_run(mapRunArn=x['mapRunArn']) for x in sf.list_map_runs(executionArn=control['executionArn'])['mapRuns']]
    print(json.dumps(dict(status=execution['status'],maps=[m['itemCounts'] for m in maps])))


def download(s):
    s3=client(s,'s3');prefix='runs/'+RUN+'/synthetic/search-results/'
    keys=[x['Key'] for page in s3.get_paginator('list_objects_v2').paginate(Bucket=BUCKET,Prefix=prefix) for x in page.get('Contents',[]) if x['Key'].endswith('.json')]
    def read(k):return json.loads(s3.get_object(Bucket=BUCKET,Key=k)['Body'].read())
    with ThreadPoolExecutor(max_workers=8) as pool:records=list(pool.map(read,keys))
    folder=ROOT/'results/cloud'/RUN;folder.mkdir(parents=True,exist_ok=True)
    (folder/'records.json').write_text(json.dumps({'run_id':RUN,'runs':records},indent=2)+'\n')
    print(json.dumps(dict(downloaded=len(records),directory=str(folder))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['smoke','start','status','download'])
    parser.add_argument('--setup-dir',type=Path,default=SETUP)
    parser.add_argument('--run-id',default=RUN)
    parser.add_argument('--expected-jobs',type=int,default=278)
    parser.add_argument('--smoke-cardinality-only',action='store_true')
    args=parser.parse_args()
    SETUP=args.setup_dir
    RUN=args.run_id
    s=session()
    if args.command=='start':start(s,args.expected_jobs)
    elif args.command=='smoke':smoke(s,args.smoke_cardinality_only)
    else:globals()[args.command](s)
