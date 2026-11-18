"""Lambda entry points. Inputs and scientific results live in S3, not state payloads."""

import hashlib
import io
import itertools
import json
import os
import platform
import time

import boto3
from botocore.exceptions import ClientError


def clients():
    return boto3.client("s3"), boto3.client("dynamodb")


def encode(value):
    def finite(item):
        if isinstance(item, dict):
            return {k: finite(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [finite(v) for v in item]
        if isinstance(item, float) and not __import__("math").isfinite(item):
            return None
        return item
    return json.dumps(finite(value), sort_keys=True, separators=(",", ":")).encode()


def put(s3, bucket, key, body, immutable=False):
    options = {"IfNoneMatch": "*"} if immutable else {}
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=body, **options)
    except ClientError as error:
        if not immutable or error.response["Error"]["Code"] not in ("PreconditionFailed", "412"):
            raise


def reserve(ddb, request_id, microdollars):
    """Reserve the full timeout cost before scheduling a batch; never refund it."""
    if microdollars > 80_000_000:
        raise ValueError("Batch exceeds the $80 lifetime workload allowance; use a smaller batch")
    ddb.transact_write_items(TransactItems=[
        {"Update": {
            "TableName": os.environ["LEDGER_TABLE"], "Key": {"id": {"S": "lifetime"}},
            "UpdateExpression": "SET used_micro = used_micro + :amount",
            "ConditionExpression": "locked = :false AND used_micro <= :remaining",
            "ExpressionAttributeValues": {":amount": {"N": str(microdollars)},
                                          ":remaining": {"N": str(80_000_000 - microdollars)},
                                          ":false": {"BOOL": False}},
        }},
        {"Put": {"TableName": os.environ["LEDGER_TABLE"],
                 "Item": {"id": {"S": "reservation/" + request_id}, "reserved_micro": {"N": str(microdollars)}},
                 "ConditionExpression": "attribute_not_exists(id)"}},
    ])


def matrix_bytes(matrix):
    import numpy as np
    buffer = io.BytesIO()
    np.save(buffer, matrix, allow_pickle=False)
    return buffer.getvalue()


def generate(event, context):
    """Stage one: freeze all samples, then publish the optimizer job manifest."""
    import numpy as np
    from isometry_pursuit.simulation import generate_identity_instance

    s3, ddb = clients()
    bucket = os.environ["EXPERIMENT_BUCKET"]
    config_raw = s3.get_object(Bucket=bucket, Key=event["config_key"])["Body"].read()
    config = json.loads(config_raw)
    run_id = event["run_id"]
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
        raise ValueError("Invalid run ID")
    settings = config["optimizers"]
    if not settings or len(settings) > 100:
        raise ValueError("Specify between one and 100 optimizer settings")
    if len({hashlib.sha256(encode(s)).hexdigest() for s in settings}) != len(settings):
        raise ValueError("Optimizer configurations must be distinct")
    for setting in settings:
        if setting["solver"] not in ("scs", "admm"):
            raise ValueError("Unknown solver")
    dataset = os.environ["DATASET_NAME"]
    real_specs = [spec for spec in config.get("real", []) if spec["dataset"] == dataset]
    synthetic = config.get("synthetic", []) if dataset == "synthetic" else []
    if len(synthetic) > 3250:
        raise ValueError("At most 3250 synthetic samples per batch")
    for spec in synthetic:
        if not (2 <= spec["D"] <= 64 and 1 <= spec["corrupting_D"] < spec["D"]
                and 1 <= spec["num_corrupting_matrices"] <= 16):
            raise ValueError("Synthetic dimensions exceed the bounded experiment grid")
    sample_count = sum(len(r["replicates"]) for r in real_specs) + len(synthetic)
    job_count = sample_count * len(settings)
    if not 0 < job_count <= 100000:
        raise ValueError("The batch must contain between one and 100000 optimization jobs")
    reservation = int(os.environ["JOB_RESERVATION_MICRO"]) * job_count + 100000
    reserve(ddb, context.aws_request_id, reservation)
    prefix = "runs/" + run_id + "/" + dataset
    put(s3, bucket, prefix + "/config.json", config_raw, immutable=True)
    samples = []

    def save_sample(matrix, metadata):
        if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
            raise ValueError("Samples must be finite two-dimensional arrays")
        body = matrix_bytes(matrix)
        digest = hashlib.sha256(body).hexdigest()
        key = "data/" + digest + ".npy"
        put(s3, bucket, key, body, immutable=True)
        samples.append({**metadata, "data_key": key, "data_sha256": digest,
                        "shape": list(matrix.shape)})

    for spec in real_specs:
        body = s3.get_object(Bucket=bucket, Key=spec["source_key"])["Body"].read()
        if hashlib.sha256(body).hexdigest() != spec["source_sha256"]:
            raise ValueError("Source tensor hash mismatch")
        tensor = np.load(io.BytesIO(body), allow_pickle=False)
        for replicate in spec["replicates"]:
            save_sample(tensor[:, :, replicate], {"dataset": spec["dataset"],
                        "replicate": replicate, "data_seed": None,
                        "source_sha256": spec["source_sha256"]})
    for spec in synthetic:
        if "source_key" in spec:
            body = s3.get_object(Bucket=bucket, Key=spec["source_key"])["Body"].read()
            if hashlib.sha256(body).hexdigest() != spec["source_sha256"]:
                raise ValueError("Frozen synthetic source hash mismatch")
            matrix = np.load(io.BytesIO(body), allow_pickle=False)
            expected = (spec["D"], spec["D"] + spec["corrupting_D"] * spec["num_corrupting_matrices"])
            if matrix.shape != expected:
                raise ValueError("Frozen synthetic source shape mismatch")
            true = list(range(spec["D"]))
        else:
            matrix, true, _, _ = generate_identity_instance(
                spec["D"], spec["corrupting_D"], spec["num_corrupting_matrices"],
                np.random.default_rng(spec["data_seed"]),
            )
        save_sample(matrix, {**spec, "dataset": "synthetic", "true_indices": true})
    jobs = []
    for index, (sample, setting) in enumerate(itertools.product(samples, settings)):
        jobs.append({"bucket": bucket, "run_prefix": prefix,
                     "job_id": f"{index:07d}", "sample": sample, "settings": setting,
                     "searches": config.get("searches", True)})
    put(s3, bucket, prefix + "/samples.json", encode(samples), immutable=True)
    jobs_key = prefix + "/jobs.jsonl"
    put(s3, bucket, jobs_key, b"\n".join(encode(job) for job in jobs) + b"\n", immutable=True)
    summary = {"bucket": bucket, "run_prefix": prefix, "jobs_key": jobs_key,
               "sample_count": len(samples), "job_count": len(jobs),
               "reserved_usd": reservation / 1e6,
               "config_sha256": hashlib.sha256(config_raw).hexdigest()}
    put(s3, bucket, prefix + "/generation.json", encode(summary), immutable=True)
    return summary


def optimize(event, context):
    """Stage two: load exactly one saved sample and run one optimizer setting."""
    import importlib.metadata
    import numpy as np
    from isometry_pursuit.cloud_experiments import optimize_sample

    s3, _ = clients()
    bucket = os.environ["EXPERIMENT_BUCKET"]
    if event["bucket"] != bucket:
        raise ValueError("Unexpected bucket")
    sample = event["sample"]
    body = s3.get_object(Bucket=bucket, Key=sample["data_key"])["Body"].read()
    if hashlib.sha256(body).hexdigest() != sample["data_sha256"]:
        raise ValueError("Sample hash mismatch")
    matrix = np.load(io.BytesIO(body), allow_pickle=False)
    prefix = event["run_prefix"] + "/results/" + event["job_id"]
    record = {**sample, "settings": event["settings"], "job_id": event["job_id"],
              "code_sha256": os.environ["CODE_SHA256"], "python": platform.python_version(),
              "versions": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "cvxpy", "scs")}}
    started = time.perf_counter()
    try:
        requested_limit = float(event['settings'].get('time_limit_seconds', 25))
        if not 0 < requested_limit <= 840:
            raise ValueError('Solver time limit must be positive and at most 840 seconds')
        limit = min(requested_limit, max(1, context.get_remaining_time_in_millis() / 1000 - 15))
        beta, result = optimize_sample(matrix, event["settings"], searches=event["searches"], time_limit=limit)
        beta_body = matrix_bytes(beta)
        put(s3, bucket, prefix + "_beta.npy", beta_body)
        record.update(result)
        record["beta_key"] = prefix + "_beta.npy"
        record["beta_sha256"] = hashlib.sha256(beta_body).hexdigest()
        if "true_indices" in sample:
            record["contains_true_basis"] = set(sample["true_indices"]).issubset(result["basis_pursuit_indices"])
    except Exception as error:
        record.update({"status": "error", "error_type": type(error).__name__, "error": str(error)[:1000]})
    record["total_runtime_seconds"] = time.perf_counter() - started
    put(s3, bucket, prefix + ".json", encode(record))
    return {"job_id": event["job_id"], "result_key": prefix + ".json", "status": record["status"]}


def search(event, context):
    """Run one greedy initializer and swap refinement on a frozen dictionary."""
    import signal
    import numpy as np
    from isometry_pursuit.cloud_search import search_saved_support

    s3, ddb = clients()
    bucket = os.environ['EXPERIMENT_BUCKET']
    if event['bucket'] != bucket:
        raise ValueError('Unexpected bucket')
    ledger = ddb.get_item(TableName=os.environ['LEDGER_TABLE'],
                         Key={'id': {'S': 'lifetime'}}, ConsistentRead=True)['Item']
    if ledger['locked']['BOOL']:
        raise RuntimeError('Project budget is locked')
    sample = event['sample']
    body = s3.get_object(Bucket=bucket, Key=sample['data_key'])['Body'].read()
    if hashlib.sha256(body).hexdigest()!=sample['data_sha256']:
        raise ValueError('Sample hash mismatch')
    matrix = np.load(io.BytesIO(body), allow_pickle=False)
    settings = event['settings']
    record = {**sample, 'job_id': event['job_id'], 'settings': settings,
              'candidate_indices': event['candidate_indices'],
              'code_sha256': os.environ['CODE_SHA256'], 'python': platform.python_version(),
              'status': 'pending'}
    def timeout(signum, frame):
        raise TimeoutError('Search wall-clock limit reached')
    old = signal.signal(signal.SIGALRM, timeout)
    seconds = min(float(settings.get('time_limit_seconds',240)),
                  context.get_remaining_time_in_millis()/1000-15)
    if not 0 < seconds <= 840:
        raise ValueError('Invalid search time budget')
    started = time.perf_counter()
    def progress(update):
        record.update(update)
        if update.get('phase') == 'swap':
            # The comparison cutoff applies to initialization, not common swap refinement.
            remaining = context.get_remaining_time_in_millis()/1000-15
            if remaining <= 0:
                raise TimeoutError('Lambda execution time exhausted before swaps')
            signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        record.update(search_saved_support(matrix, event['candidate_indices'], settings,
                                           progress=progress))
        if record.get('indices') is not None and 'true_indices' in sample:
            record['exact_support'] = set(record['indices'])==set(sample['true_indices'])
    except TimeoutError as error:
        record.update(status='time_limit', error=str(error))
        if record.get('phase')=='initialization':
            record.update(initialization_seconds=time.perf_counter()-started,
                          initialization_status='time_limit')
    except Exception as error:
        record.update(status='error', error_type=type(error).__name__, error=str(error)[:1000])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
    record['total_runtime_seconds'] = time.perf_counter()-started
    key = event['run_prefix']+'/search-results/'+event['job_id']+'.json'
    put(s3, bucket, key, encode(record))
    return {'job_id':event['job_id'], 'result_key':key, 'status':record['status']}


def search_metered(event, context):
    """Run a search within a fixed prepaid batch allocation.

    The lifetime reservation is never refunded. Inside that reservation, each
    invocation holds its full timeout exposure until it finishes; unused lease
    capacity can then fund another invocation in the same batch.
    """
    import math
    s3, ddb = clients()
    table = os.environ['LEDGER_TABLE']
    budget_id = event['search_budget_id']
    if budget_id != 'batch/full-search01-20261116-001':
        raise ValueError('Unexpected search batch allocation')
    allowance = 60000
    lease = 'lease/' + context.aws_request_id
    def transact(items, phase):
        # Concurrent searches update the same prepaid allocation. A transaction
        # conflict is transient, not evidence that the allocation is exhausted.
        import random
        token = hashlib.sha256((context.aws_request_id + phase).encode()).hexdigest()[:32]
        for attempt in range(16):
            try:
                return ddb.transact_write_items(TransactItems=items, ClientRequestToken=token)
            except ClientError as error:
                reasons = error.response.get('CancellationReasons', [])
                conflict = (error.response['Error']['Code'] == 'TransactionCanceledException'
                            and any(r.get('Code') == 'TransactionConflict' for r in reasons))
                if not conflict or attempt == 15:
                    raise
                time.sleep(random.uniform(.005, min(.5, .01 * 2**attempt)))
    try:
        transact([
            {'Update': {'TableName': table, 'Key': {'id': {'S': budget_id}},
                'UpdateExpression': 'SET remaining_micro = remaining_micro - :hold, active_micro = active_micro + :hold, jobs_started = jobs_started + :one',
                'ConditionExpression': 'remaining_micro >= :hold',
                'ExpressionAttributeValues': {':hold': {'N': str(allowance)}, ':one': {'N': '1'}}}},
            {'Put': {'TableName': table, 'Item': {'id': {'S': lease},
                'budget_id': {'S': budget_id}, 'job_id': {'S': event['job_id']},
                'held_micro': {'N': str(allowance)}, 'state': {'S': 'active'}},
                'ConditionExpression': 'attribute_not_exists(id)'}}], 'acquire')
    except ClientError as error:
        reasons = error.response.get('CancellationReasons', [])
        if (error.response['Error']['Code'] != 'TransactionCanceledException'
                or not reasons or reasons[0].get('Code') != 'ConditionalCheckFailed'):
            raise
        record = {**event['sample'], 'job_id': event['job_id'], 'settings': event['settings'],
                  'candidate_indices': event['candidate_indices'], 'status': 'budget_deferred'}
        key = event['run_prefix']+'/search-results/'+event['job_id']+'.json'
        put(s3, os.environ['EXPERIMENT_BUCKET'], key, encode(record))
        return {'job_id':event['job_id'], 'result_key':key, 'status':'budget_deferred'}
    try:
        return search(event, context)
    finally:
        # 3 GB x $0.0000166667/GB-second, plus request/orchestration allowance.
        seconds = max(0., 900-context.get_remaining_time_in_millis()/1000)
        charged = min(allowance, math.ceil(seconds*50.0001)+250)
        released = allowance-charged
        transact([
            {'Update': {'TableName': table, 'Key': {'id': {'S': budget_id}},
                'UpdateExpression': 'SET remaining_micro = remaining_micro + :release, active_micro = active_micro - :hold, charged_micro = charged_micro + :charge, jobs_finished = jobs_finished + :one',
                'ConditionExpression': 'active_micro >= :hold',
                'ExpressionAttributeValues': {':release': {'N': str(released)}, ':hold': {'N': str(allowance)},
                    ':charge': {'N': str(charged)}, ':one': {'N': '1'}}}},
            {'Update': {'TableName': table, 'Key': {'id': {'S': lease}},
                'UpdateExpression': 'SET #state = :done, charged_micro = :charge',
                'ConditionExpression': '#state = :active',
                'ExpressionAttributeNames': {'#state': 'state'},
                'ExpressionAttributeValues': {':done': {'S': 'finished'}, ':active': {'S': 'active'},
                                              ':charge': {'N': str(charged)}}}}], 'release')


def shutdown(event, context):
    """Budget notification: lock the ledger, throttle this project and stop its runs."""
    ddb = boto3.client("dynamodb")
    ddb.update_item(TableName=os.environ["LEDGER_TABLE"], Key={"id": {"S": "lifetime"}},
                    UpdateExpression="SET locked = :yes", ExpressionAttributeValues={":yes": {"BOOL": True}})
    client = boto3.client("lambda")
    for name in os.environ["FUNCTION_NAMES"].split(","):
        client.put_function_concurrency(FunctionName=name, ReservedConcurrentExecutions=0)
    states = boto3.client("stepfunctions")
    for page in states.get_paginator("list_executions").paginate(
            stateMachineArn=os.environ["STATE_MACHINE_ARN"], statusFilter="RUNNING"):
        for execution in page["executions"]:
            states.stop_execution(executionArn=execution["executionArn"], cause="Project budget shutdown")
    return {"locked": True}
