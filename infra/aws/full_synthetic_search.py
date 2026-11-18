"""Run the fixed-scale synthetic searches within a prepaid $8 batch budget."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import zipfile

from manage import BUCKET, ROOT, client, outputs, session
from search_grid import reserve


RUN='full-search01-20261116-001'
SETUP=ROOT/'results/cloud/full-search01-setup'
BUNDLE=ROOT/'results/cloud/full-synthetic-20261116-001'
BUDGET='batch/'+RUN


def read_records(path):
    value=json.loads(path.read_text())
    return value['runs'] if isinstance(value,dict) else value


def deploy(s):
    SETUP.mkdir(parents=True,exist_ok=True)
    if (SETUP/'deployment.json').exists():
        raise RuntimeError('Deployment already recorded')
    l=client(s,'lambda');c=l.get_function_configuration(FunctionName='isometry-search')
    assert c['MemorySize']==3072 and c['Timeout']==900
    archive=ROOT/'results/cloud/lambda-full-search-metered.zip'
    with zipfile.ZipFile(ROOT/'results/cloud/lambda.zip') as old, zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as new:
        for info in old.infolist():
            new.writestr(info,(ROOT/'infra/aws/handlers.py').read_bytes() if info.filename=='cloud_handlers.py' else old.read(info.filename))
    digest=hashlib.sha256(archive.read_bytes()).hexdigest();key='code/'+digest+'.zip'
    client(s,'s3').upload_file(str(archive),BUCKET,key)
    l.update_function_code(FunctionName='isometry-search',S3Bucket=BUCKET,S3Key=key)
    l.get_waiter('function_updated_v2').wait(FunctionName='isometry-search')
    env=c['Environment']['Variables'];env['CODE_SHA256']=digest
    l.update_function_configuration(FunctionName='isometry-search',Handler='cloud_handlers.search_metered',Environment={'Variables':env})
    l.get_waiter('function_updated_v2').wait(FunctionName='isometry-search')
    try:
        published=l.publish_version(FunctionName='isometry-search',Description='Prepaid fixed-SCS full synthetic search stage')
    finally:
        l.update_function_configuration(FunctionName='isometry-search',Handler='cloud_handlers.search')
        l.get_waiter('function_updated_v2').wait(FunctionName='isometry-search')
    sf=client(s,'stepfunctions');machine=outputs(s)['StateMachineArn']
    old=sf.describe_state_machine(stateMachineArn=machine);definition=json.loads(old['definition'])
    updated=copy.deepcopy(definition)
    updated['States']['SearchSavedSamples']['ItemProcessor']['States']['SearchOne']['Parameters']['FunctionName']=published['FunctionArn']
    assert sf.validate_state_machine_definition(definition=json.dumps(updated),type='STANDARD')['result']=='OK'
    role=old['roleArn'].rsplit('/',1)[-1]
    client(s,'iam').put_role_policy(RoleName=role,PolicyName='isometry-full-search-version',PolicyDocument=json.dumps({
        'Version':'2012-10-17','Statement':[{'Effect':'Allow','Action':'lambda:InvokeFunction','Resource':published['FunctionArn']}]}))
    try:
        version=sf.update_state_machine(stateMachineArn=machine,definition=json.dumps(updated),publish=True,
            versionDescription='Fixed-scale full synthetic searches with prepaid batch budget')['stateMachineVersionArn']
    finally:
        sf.update_state_machine(stateMachineArn=machine,definition=old['definition'])
    reserve(s,RUN,8_000_000)
    client(s,'dynamodb').put_item(TableName='isometry-experiment-budget',Item={
        'id':{'S':BUDGET},'remaining_micro':{'N':'8000000'},'active_micro':{'N':'0'},
        'charged_micro':{'N':'0'},'jobs_started':{'N':'0'},'jobs_finished':{'N':'0'}},
        ConditionExpression='attribute_not_exists(id)')
    control=dict(run_id=RUN,lambda_arn=published['FunctionArn'],lambda_profile=dict(MemorySize=3072,Timeout=900),
                 code_sha256=digest,state_machine_version=version,budget_id=BUDGET,reserved_usd=8)
    (SETUP/'deployment.json').write_text(json.dumps(control,indent=2)+'\n')
    print(json.dumps(control))


def qualify(record,source):
    return {**record,'raw_result_job_id':record['job_id'],'source_run_id':source,
            'job_id':source+'/'+record['job_id']}


def prepare():
    old=read_records(ROOT/'results/cloud/large-grid-20261116-001/records.json')
    old=[r for r in old if r['settings']['solver']=='scs' and r['settings']['scale']==.1]
    new=read_records(ROOT/'results/cloud/full-scs01-20261116-001/records.json')
    assert len(old)==65 and len(new)==3185
    solvers=[qualify(r,'large-grid-20261116-001') for r in old]+[qualify(r,'full-scs01-20261116-001') for r in new]
    assert len({r['saved_run_id'] for r in solvers})==3250
    assert all(r['settings']['solver']=='scs' and r['settings']['scale']==.1 and r['status']!='error' for r in solvers)
    by_search={r['job_id']:r for r in read_records(ROOT/'results/cloud/large-search-20261116-001/records.json')}
    replacements=read_records(ROOT/'results/cloud/swap-completion-20261116-001/merged_searches.json')
    replacement_ids={r['job_id'] for r in replacements}
    for r in replacements:
        by_search[r['job_id']]=r
    old_alias=json.loads((ROOT/'results/cloud/large-search-setup/aliases.json').read_text())
    links={(a['optimizer_job_id'],a['search']):a['search_job_id'] for a in old_alias}
    names={'forward_full':('forward',False),'backward_full':('backward',False),
           'forward_restricted':('forward',True),'backward_restricted':('backward',True)}
    aliases=[];jobs=[];existing={};new_keys={}
    sample_fields=['D','corrupting_D','num_corrupting_matrices','replicate','data_seed',
                   'saved_run_id','shape','data_key','data_sha256','true_indices','dataset']
    for solver in sorted(solvers,key=lambda r:r['saved_run_id']):
        for name,(kind,restricted) in names.items():
            if solver['source_run_id']=='large-grid-20261116-001':
                result=by_search[links[(solver['raw_result_job_id'],name)]]
                source=('swap-completion-20261116-001' if result['job_id']=='0000077'
                        else 'backward-time-budget-20261116-001' if result['job_id'] in replacement_ids
                        else 'large-search-20261116-001')
                qualified=qualify(result,source);search_id=qualified['job_id'];existing[search_id]=qualified
            else:
                candidates=solver['basis_pursuit_indices'] if restricted else list(range(solver['shape'][1]))
                key=(solver['data_sha256'],tuple(candidates),kind,solver['data_seed'])
                if key not in new_keys:
                    job_id=f'{len(jobs):07d}';new_keys[key]=RUN+'/'+job_id
                    jobs.append(dict(bucket=BUCKET,run_prefix='runs/'+RUN+'/synthetic',job_id=job_id,
                        sample={k:solver[k] for k in sample_fields},candidate_indices=candidates,
                        settings=dict(initialization=kind,search_seed=solver['data_seed'],search_on_raw=True,
                                      max_backward_work=None,time_limit_seconds=245.36574106800015),
                        search_budget_id=BUDGET))
                search_id=new_keys[key]
            aliases.append(dict(optimizer_job_id=solver['job_id'],search=name,search_job_id=search_id))
    SETUP.mkdir(parents=True,exist_ok=True);BUNDLE.mkdir(parents=True,exist_ok=True)
    (SETUP/'jobs.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in jobs))
    (BUNDLE/'records.json').write_text(json.dumps({'runs':solvers},indent=2)+'\n')
    (BUNDLE/'aliases.json').write_text(json.dumps(aliases,indent=2)+'\n')
    (SETUP/'existing_search_records.json').write_text(json.dumps({'runs':list(existing.values())},indent=2)+'\n')
    print(json.dumps(dict(solvers=len(solvers),new_searches=len(jobs),reused_searches=len(existing),links=len(aliases))))


def smoke(s):
    deployment=json.loads((SETUP/'deployment.json').read_text())
    job=json.loads((SETUP/'jobs.jsonl').read_text().splitlines()[0])
    job['job_id']='smoke-'+job['job_id'];job['run_prefix']='runs/'+RUN+'-smoke/synthetic'
    result=client(s,'lambda').invoke(FunctionName=deployment['lambda_arn'],InvocationType='RequestResponse',Payload=json.dumps(job).encode())
    body=json.loads(result['Payload'].read())
    if result.get('FunctionError') or body.get('status')!='complete':
        raise RuntimeError(body)
    (SETUP/'smoke.json').write_text(json.dumps(body,indent=2)+'\n');print(json.dumps(body))


def start(s):
    if (SETUP/'execution.json').exists():raise RuntimeError('Search stage already submitted')
    if not (SETUP/'smoke.json').exists():raise RuntimeError('Smoke test missing')
    d=json.loads((SETUP/'deployment.json').read_text());raw=(SETUP/'jobs.jsonl').read_bytes()
    s3=client(s,'s3');key='runs/'+RUN+'/search-jobs.jsonl';s3.put_object(Bucket=BUCKET,Key=key,Body=raw)
    response=client(s,'stepfunctions').start_execution(stateMachineArn=d['state_machine_version'],name=RUN,
        input=json.dumps({'search_jobs_key':key,'run_prefix':'runs/'+RUN}))
    control={**d,'executionArn':response['executionArn'],'startDate':str(response['startDate']),'jobs':len(raw.splitlines())}
    (SETUP/'execution.json').write_text(json.dumps(control,indent=2)+'\n');print(json.dumps(control))


def status(s):
    c=json.loads((SETUP/'execution.json').read_text());sf=client(s,'stepfunctions')
    e=sf.describe_execution(executionArn=c['executionArn'])
    maps=[sf.describe_map_run(mapRunArn=m['mapRunArn'])['itemCounts'] for m in sf.list_map_runs(executionArn=c['executionArn'])['mapRuns']]
    budget=client(s,'dynamodb').get_item(TableName='isometry-experiment-budget',Key={'id':{'S':BUDGET}},ConsistentRead=True)['Item']
    print(json.dumps(dict(status=e['status'],maps=maps,budget={k:int(budget[k]['N'])/1e6 for k in ['remaining_micro','active_micro','charged_micro']})))


def recover(s):
    """Resume unfinished jobs with conflict retries; retain all old budget holds."""
    control=json.loads((SETUP/'execution.json').read_text())
    sf=client(s,'stepfunctions')
    assert sf.describe_execution(executionArn=control['executionArn'])['status']=='FAILED'
    assert not (SETUP/'recovery.json').exists(), 'Recovery already submitted'
    records=read_records(ROOT/'results/cloud'/RUN/'records.json')
    assert len(records)==20 and all(r['status']=='complete' for r in records)
    completed={r['job_id'] for r in records}
    jobs=[json.loads(line) for line in (SETUP/'jobs.jsonl').read_text().splitlines()]
    jobs=[j for j in jobs if j['job_id'] not in completed]
    assert len(jobs)==12066
    ddb=client(s,'dynamodb')
    budget=ddb.get_item(TableName='isometry-experiment-budget',Key={'id':{'S':BUDGET}},ConsistentRead=True)['Item']
    assert int(budget['active_micro']['N'])==120000
    assert int(budget['remaining_micro']['N'])>60000
    l=client(s,'lambda');c=l.get_function_configuration(FunctionName='isometry-search')
    assert c['MemorySize']==3072 and c['Timeout']==900
    archive=ROOT/'results/cloud/lambda-full-search-conflict-retry.zip'
    with zipfile.ZipFile(ROOT/'results/cloud/lambda-full-search-metered.zip') as old, zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as new:
        for info in old.infolist():
            new.writestr(info,(ROOT/'infra/aws/handlers.py').read_bytes() if info.filename=='cloud_handlers.py' else old.read(info.filename))
    digest=hashlib.sha256(archive.read_bytes()).hexdigest();s3=client(s,'s3');key='code/'+digest+'.zip'
    s3.upload_file(str(archive),BUCKET,key)
    l.update_function_code(FunctionName='isometry-search',S3Bucket=BUCKET,S3Key=key)
    l.get_waiter('function_updated_v2').wait(FunctionName='isometry-search')
    env=c['Environment']['Variables'];env['CODE_SHA256']=digest
    l.update_function_configuration(FunctionName='isometry-search',Handler='cloud_handlers.search_metered',Environment={'Variables':env})
    l.get_waiter('function_updated_v2').wait(FunctionName='isometry-search')
    try:
        published=l.publish_version(FunctionName='isometry-search',Description='Retry prepaid-budget transaction conflicts; unchanged scientific search')
    finally:
        l.update_function_configuration(FunctionName='isometry-search',Handler='cloud_handlers.search')
        l.get_waiter('function_updated_v2').wait(FunctionName='isometry-search')
    machine=outputs(s)['StateMachineArn'];old=sf.describe_state_machine(stateMachineArn=machine)
    definition=json.loads(old['definition'])
    definition['States']['SearchSavedSamples']['ItemProcessor']['States']['SearchOne']['Parameters']['FunctionName']=published['FunctionArn']
    assert sf.validate_state_machine_definition(definition=json.dumps(definition),type='STANDARD')['result']=='OK'
    role=old['roleArn'].rsplit('/',1)[-1]
    client(s,'iam').put_role_policy(RoleName=role,PolicyName='isometry-full-search-recovery-version',PolicyDocument=json.dumps({
        'Version':'2012-10-17','Statement':[{'Effect':'Allow','Action':'lambda:InvokeFunction','Resource':published['FunctionArn']}]}))
    try:
        version=sf.update_state_machine(stateMachineArn=machine,definition=json.dumps(definition),publish=True,
            versionDescription='Resume existing prepaid batch with transaction-conflict retries')['stateMachineVersionArn']
    finally:
        sf.update_state_machine(stateMachineArn=machine,definition=old['definition'])
    smoke_job=copy.deepcopy(jobs[0]);smoke_job['job_id']='recovery-smoke-'+smoke_job['job_id']
    smoke_job['run_prefix']='runs/'+RUN+'-smoke/synthetic'
    reply=l.invoke(FunctionName=published['FunctionArn'],InvocationType='RequestResponse',Payload=json.dumps(smoke_job).encode())
    body=json.loads(reply['Payload'].read())
    assert not reply.get('FunctionError') and body.get('status')=='complete', body
    (SETUP/'recovery_smoke.json').write_text(json.dumps(body,indent=2)+'\n')
    raw=''.join(json.dumps(j)+'\n' for j in jobs).encode();key='runs/'+RUN+'/recovery-jobs.jsonl'
    (SETUP/'recovery-jobs.jsonl').write_bytes(raw);s3.put_object(Bucket=BUCKET,Key=key,Body=raw)
    result=sf.start_execution(stateMachineArn=version,name=RUN+'-recovery1',
        input=json.dumps({'search_jobs_key':key,'run_prefix':'runs/'+RUN+'-recovery1'}))
    resumed={**control,'lambda_arn':published['FunctionArn'],'code_sha256':digest,'state_machine_version':version,
             'executionArn':result['executionArn'],'startDate':str(result['startDate']),'jobs':len(jobs),
             'original_executionArn':control['executionArn'],'retained_active_micro':120000}
    (SETUP/'execution_initial_failed.json').write_text(json.dumps(control,indent=2)+'\n')
    (SETUP/'recovery.json').write_text(json.dumps(resumed,indent=2)+'\n')
    (SETUP/'execution.json').write_text(json.dumps(resumed,indent=2)+'\n')
    print(json.dumps(resumed))


def download(s):
    s3=client(s,'s3');folder=ROOT/'results/cloud'/RUN;folder.mkdir(exist_ok=True)
    keys=[x['Key'] for page in s3.get_paginator('list_objects_v2').paginate(Bucket=BUCKET,Prefix='runs/'+RUN+'/synthetic/search-results/')
          for x in page.get('Contents',[]) if x['Key'].endswith('.json')]
    def read(key):return json.loads(s3.get_object(Bucket=BUCKET,Key=key)['Body'].read())
    with ThreadPoolExecutor(max_workers=8) as pool:records=list(pool.map(read,keys))
    (folder/'records.json').write_text(json.dumps({'runs':records},indent=2)+'\n')
    combined=read_records(SETUP/'existing_search_records.json')+[qualify(r,RUN) for r in records]
    (BUNDLE/'search_records.json').write_text(json.dumps({'runs':combined},indent=2)+'\n')
    print(json.dumps(dict(downloaded=len(records),combined=len(combined))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['deploy','prepare','smoke','start','status','download','recover']);a=p.parse_args()
    if a.command=='prepare':prepare()
    else:globals()[a.command](session())
