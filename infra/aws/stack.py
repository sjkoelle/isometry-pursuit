"""CloudFormation definition for the two-stage experiment workflow."""


def template():
    ref = lambda name: {"Ref": name}
    arn = lambda name: {"Fn::GetAtt": [name, "Arn"]}
    sub = lambda value: {"Fn::Sub": value}
    tags = [{"Key": "Project", "Value": "isometry-pursuit"}]

    def role(service, statements):
        return {"Type": "AWS::IAM::Role", "Properties": {
            "AssumeRolePolicyDocument": {"Version": "2012-10-17", "Statement": [{
                "Effect": "Allow", "Principal": {"Service": service}, "Action": "sts:AssumeRole"}]},
            "Policies": [{"PolicyName": "experiment-access", "PolicyDocument": {
                "Version": "2012-10-17", "Statement": statements}}], "Tags": tags}}

    def allow(actions, resources):
        return {"Effect": "Allow", "Action": actions, "Resource": resources}

    logging = allow(["logs:CreateLogStream", "logs:PutLogEvents"],
                    sub("arn:${AWS::Partition}:logs:${AWS::Region}:${AWS::AccountId}:log-group:/aws/lambda/isometry-*:*"))
    resources = {
        "Ledger": {"Type": "AWS::DynamoDB::Table", "Properties": {
            "TableName": "isometry-experiment-budget", "BillingMode": "PAY_PER_REQUEST",
            "AttributeDefinitions": [{"AttributeName": "id", "AttributeType": "S"}],
            "KeySchema": [{"AttributeName": "id", "KeyType": "HASH"}], "Tags": tags}},
        "WorkerRole": role("lambda.amazonaws.com", [logging,
            allow(["s3:GetObject", "s3:PutObject"], sub("arn:${AWS::Partition}:s3:::${Bucket}/*")),
            allow(["dynamodb:TransactWriteItems", "dynamodb:UpdateItem", "dynamodb:PutItem", "dynamodb:GetItem"], arn("Ledger"))]),
    }
    for logical, name, handler, memory, timeout, concurrency in (
        ("PrepareIris", "isometry-prepare-iris", "generate", 1536, 900, 1),
        ("PrepareWine", "isometry-prepare-wine", "generate", 1536, 900, 1),
        ("PrepareEthanol", "isometry-prepare-ethanol", "generate", 1536, 900, 1),
        ("PrepareSynthetic", "isometry-prepare-synthetic", "generate", 1536, 900, 1),
        ("Optimize", "isometry-optimize", "optimize", 1024, 45, 8),
        ("Search", "isometry-search", "search", 3072, 900, 8),
    ):
        resources[logical + "Logs"] = {"Type": "AWS::Logs::LogGroup", "Properties": {
            "LogGroupName": "/aws/lambda/" + name, "RetentionInDays": 7, "Tags": tags}}
        resources[logical] = {"Type": "AWS::Lambda::Function", "DependsOn": logical + "Logs", "Properties": {
            "FunctionName": name, "Runtime": "python3.12", "Architectures": ["x86_64"],
            "Handler": "cloud_handlers." + handler, "Role": arn("WorkerRole"),
            "Code": {"S3Bucket": ref("Bucket"), "S3Key": ref("CodeKey")},
            "MemorySize": memory, "Timeout": timeout, "ReservedConcurrentExecutions": concurrency,
            "Environment": {"Variables": {
                "EXPERIMENT_BUCKET": ref("Bucket"), "LEDGER_TABLE": ref("Ledger"),
                "JOB_RESERVATION_MICRO": "1300", "CODE_SHA256": ref("CodeHash"),
                "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}},
            "Tags": tags}}
        if handler == "generate":
            resources[logical]["Properties"]["Environment"]["Variables"]["DATASET_NAME"] = name.removeprefix("isometry-prepare-")
    resources["StatesRole"] = role("states.amazonaws.com", [
        allow("lambda:InvokeFunction", [arn("PrepareIris"), arn("PrepareWine"), arn("PrepareEthanol"), arn("PrepareSynthetic"), arn("Optimize"), arn("Search")]),
        allow("states:StartExecution", sub("arn:${AWS::Partition}:states:${AWS::Region}:${AWS::AccountId}:stateMachine:isometry-experiments")),
        allow(["states:DescribeExecution", "states:StopExecution"], sub("arn:${AWS::Partition}:states:${AWS::Region}:${AWS::AccountId}:execution:isometry-experiments*:*")),
        allow(["s3:GetObject", "s3:PutObject"], sub("arn:${AWS::Partition}:s3:::${Bucket}/runs/*")),
    ])
    definition = {
        "Comment": "Freeze data in S3, then optimize paired samples. No automatic retries.",
        "StartAt": "SelectStage", "TimeoutSeconds": 604800,
        "States": {
            "SelectStage": {"Type": "Choice", "Choices": [{
                "Variable": "$.search_jobs_key", "IsPresent": True, "Next": "SearchSavedSamples"}],
                "Default": "GenerateData"},
            "SearchSavedSamples": {"Type": "Map", "MaxConcurrency": 8,
                "ItemReader": {"Resource": "arn:aws:states:::s3:getObject",
                    "ReaderConfig": {"InputType": "JSONL"},
                    "Parameters": {"Bucket": ref("Bucket"), "Key.$": "$.search_jobs_key"}},
                "ItemProcessor": {"ProcessorConfig": {"Mode": "DISTRIBUTED", "ExecutionType": "STANDARD"},
                    "StartAt": "SearchOne", "States": {
                        "SearchOne": {"Type": "Task", "Resource": "arn:aws:states:::lambda:invoke",
                            "Parameters": {"FunctionName": arn("Search"), "Payload.$": "$"},
                            "OutputPath": "$.Payload", "End": True}}},
                "ResultWriter": {"Resource": "arn:aws:states:::s3:putObject", "Parameters": {
                    "Bucket": ref("Bucket"), "Prefix.$": "States.Format('{}/search-map-results', $.run_prefix)"}},
                "End": True},
            "GenerateData": {"Type": "Map", "ItemsPath": "$.datasets", "MaxConcurrency": 4,
                "ItemSelector": {"dataset.$": "$$.Map.Item.Value", "config_key.$": "$.config_key",
                                 "run_id.$": "$$.Execution.Name"},
                "ItemProcessor": {"ProcessorConfig": {"Mode": "INLINE"}, "StartAt": "PrepareDataset",
                    "States": {"PrepareDataset": {"Type": "Task", "Resource": "arn:aws:states:::lambda:invoke",
                        "Parameters": {"FunctionName.$": "States.Format('isometry-prepare-{}', $.dataset)",
                                       "Payload.$": "$"}, "OutputPath": "$.Payload", "End": True}}},
                "ResultPath": "$.prepared", "Next": "OptimizeSavedSamples"},
            "OptimizeSavedSamples": {"Type": "Map", "ItemsPath": "$.prepared", "MaxConcurrency": 1,
                "ItemProcessor": {"ProcessorConfig": {"Mode": "INLINE"}, "StartAt": "OptimizeDataset",
                    "States": {"OptimizeDataset": {"Type": "Map", "MaxConcurrency": 8,
                "ItemReader": {"Resource": "arn:aws:states:::s3:getObject",
                    "ReaderConfig": {"InputType": "JSONL"},
                    "Parameters": {"Bucket": ref("Bucket"), "Key.$": "$.jobs_key"}},
                "ItemProcessor": {"ProcessorConfig": {"Mode": "DISTRIBUTED", "ExecutionType": "STANDARD"},
                    "StartAt": "OptimizeOne", "States": {
                        "OptimizeOne": {"Type": "Task", "Resource": "arn:aws:states:::lambda:invoke",
                            "Parameters": {"FunctionName": arn("Optimize"), "Payload.$": "$"},
                            "OutputPath": "$.Payload", "End": True}}},
                "ResultWriter": {"Resource": "arn:aws:states:::s3:putObject", "Parameters": {
                    "Bucket": ref("Bucket"), "Prefix.$": "States.Format('{}/map-results', $.run_prefix)"}},
                "End": True}}}, "ResultPath": "$.optimization", "End": True},
        },
    }
    resources["Workflow"] = {"Type": "AWS::StepFunctions::StateMachine", "Properties": {
        "StateMachineName": "isometry-experiments", "StateMachineType": "STANDARD",
        "RoleArn": arn("StatesRole"), "Definition": definition, "Tags": tags}}
    resources["ShutdownRole"] = role("lambda.amazonaws.com", [logging,
        allow("dynamodb:UpdateItem", arn("Ledger")),
        allow("lambda:PutFunctionConcurrency", [arn("PrepareIris"), arn("PrepareWine"), arn("PrepareEthanol"), arn("PrepareSynthetic"), arn("Optimize"), arn("Search")]),
        allow("states:ListExecutions", ref("Workflow")),
        allow("states:StopExecution", sub("arn:${AWS::Partition}:states:${AWS::Region}:${AWS::AccountId}:execution:isometry-experiments*:*"))])
    resources["ShutdownLogs"] = {"Type": "AWS::Logs::LogGroup", "Properties": {
        "LogGroupName": "/aws/lambda/isometry-budget-shutdown", "RetentionInDays": 7, "Tags": tags}}
    resources["Shutdown"] = {"Type": "AWS::Lambda::Function", "DependsOn": "ShutdownLogs", "Properties": {
        "FunctionName": "isometry-budget-shutdown", "Runtime": "python3.12",
        "Handler": "cloud_handlers.shutdown", "Role": arn("ShutdownRole"), "MemorySize": 128, "Timeout": 30,
        "Code": {"S3Bucket": ref("Bucket"), "S3Key": ref("CodeKey")},
        "Environment": {"Variables": {"LEDGER_TABLE": ref("Ledger"), "STATE_MACHINE_ARN": ref("Workflow"),
            "FUNCTION_NAMES": "isometry-prepare-iris,isometry-prepare-wine,isometry-prepare-ethanol,isometry-prepare-synthetic,isometry-optimize,isometry-search"}}, "Tags": tags}}
    resources["BudgetTopic"] = {"Type": "AWS::SNS::Topic", "Properties": {"TopicName": "isometry-budget-alerts", "Tags": tags}}
    resources["BudgetTopicPolicy"] = {"Type": "AWS::SNS::TopicPolicy", "Properties": {
        "Topics": [ref("BudgetTopic")], "PolicyDocument": {"Version": "2012-10-17", "Statement": [{
            "Effect": "Allow", "Principal": {"Service": "budgets.amazonaws.com"},
            "Action": "sns:Publish", "Resource": ref("BudgetTopic"),
            "Condition": {"StringEquals": {"aws:SourceAccount": ref("AWS::AccountId")}}}]}}}
    resources["BudgetSubscription"] = {"Type": "AWS::SNS::Subscription", "Properties": {
        "TopicArn": ref("BudgetTopic"), "Protocol": "lambda", "Endpoint": arn("Shutdown")}}
    resources["ShutdownPermission"] = {"Type": "AWS::Lambda::Permission", "Properties": {
        "Action": "lambda:InvokeFunction", "FunctionName": ref("Shutdown"),
        "Principal": "sns.amazonaws.com", "SourceArn": ref("BudgetTopic")}}
    return {"AWSTemplateFormatVersion": "2010-09-09",
        "Description": "Isometry Pursuit paired experiments with cumulative workload reservations",
        "Parameters": {name: {"Type": "String"} for name in ("Bucket", "CodeKey", "CodeHash")},
        "Resources": resources, "Outputs": {
            "StateMachineArn": {"Value": ref("Workflow")}, "BudgetTopicArn": {"Value": ref("BudgetTopic")},
            "LedgerTable": {"Value": ref("Ledger")}, "Bucket": {"Value": ref("Bucket")}}}
