# Environment

Main experiment environment:

- Python: 3.10+ recommended
- Operating system: Linux/EC2 environment used for main runs
- Cloud API: Amazon Bedrock through `boto3`
- AWS region: `us-east-1`
- Dependencies: see `requirements.txt`

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Environment variables
```
export AWS_REGION=us-east-1
export AWS_DEFAULT_REGION=us-east-1
export BEDROCK_MODEL_IDS="amazon.nova-lite-v1:0,amazon.nova-pro-v1:0,anthropic.claude-3-haiku-20240307-v1:0,anthropic.claude-3-sonnet-20240229-v1:0"
```
AWS credentials should be configured outside the repository.

Do not commit AWS credentials, .env files, PEM files, raw traces, request IDs, or local cloud logs.
