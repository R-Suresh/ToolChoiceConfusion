# Reproducibility Notes

This document describes the reproducibility package for **ToolChoiceConfusion: Causal Minimal Tool Filtering for Reliable LLM Agents**.

The goal of this repository is to provide a clean, public, arXiv-compatible reproducibility package. It is not intended to be a dump of raw internal experiment artifacts.

## What is included

The repository includes:

* Experimental runner code
* Analysis scripts
* Plotting scripts
* Curated benchmark task definitions
* Curated tool registry
* Task-level metrics for the main public run
* Aggregate summary tables
* Paper-style figures
* Run configuration
* Environment notes

## What is intentionally excluded

The repository does not include unsanitized raw traces or cloud execution logs.

Excluded artifacts include:

* AWS credentials
* `.env` files
* PEM/key files
* Bedrock request IDs
* raw Bedrock/API response metadata
* raw model output traces
* local EC2 paths
* hostnames
* shell history
* personal notes
* review-submission PDFs
* OpenReview/AIML submission metadata
* any double-blind review information

## Main experiment

The main runner is:

```bash
python3 scaledExperiment.py
```

The runner evaluates six tool-exposure methods:

1. `all_tools`
2. `keyword_top_5`
3. `keyword_top_10`
4. `state_aware`
5. `full_causal_path`
6. `cmtf`

The default generated output location is:

```text
results_scaled/
```

The two primary local run outputs are:

```text
results_scaled/raw_traces.jsonl
results_scaled/task_metrics.csv
```

Only curated metrics derived from these runs should be committed publicly.

## Environment setup

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Required Python packages are listed in:

```text
requirements.txt
```

The experiment uses Amazon Bedrock through `boto3`. Configure AWS credentials outside the repository.

Example environment variables:

```bash
export AWS_REGION=us-east-1
export AWS_DEFAULT_REGION=us-east-1
export BEDROCK_MODEL_IDS="amazon.nova-lite-v1:0,amazon.nova-pro-v1:0,anthropic.claude-3-haiku-20240307-v1:0,anthropic.claude-3-sonnet-20240229-v1:0"
```

## Reproducing tables

The analysis script expects:

```text
analysis/task_metrics.csv
```

To regenerate the summary tables from a local run:

```bash
mkdir -p analysis
cp results_scaled/task_metrics.csv analysis/task_metrics.csv
python3 analyze_results.py
```

This writes:

```text
tables/summary_by_model_method.csv
tables/summary_aggregate.csv
tables/summary_aggregate.tex
```

## Reproducing figures

After generating `tables/summary_aggregate.csv`, run:

```bash
python3 plot_results.py
```

This writes figures into:

```text
figures/
```

Expected figure files include:

```text
figures/success_by_method.png
figures/wrong_tools_by_method.png
figures/premature_actions_by_method.png
figures/tools_per_step_by_method.png
figures/tokens_by_method.png
```

## Public artifact policy

For public release, include:

```text
data/tasks_102.json
data/tool_registry_100.json
results/task_metrics_main.csv
results/summary_aggregate.csv
results/summary_by_model_method.csv
tables/summary_aggregate.tex
figures/*.png
reproducibility/run_config_main.json
reproducibility/environment.md
```

Do not include:

```text
results_scaled/raw_traces.jsonl
archived_runs/
*.log
.aws/
.env
*.pem
*.key
```

## Sanitizing run artifacts

Before uploading any artifact generated on EC2 or through Bedrock, inspect it for:

* Request IDs
* Account IDs
* ARNs
* access keys
* session tokens
* local file paths
* hostnames
* IP addresses
* raw prompts
* raw model completions
* provider response metadata
* timestamps that reveal internal execution details
* review or submission metadata

For the public reproducibility package, prefer derived metrics over raw traces.

Safe public artifacts generally include:

* task IDs
* method names
* model family labels or model IDs already disclosed in the paper
* success/failure indicators
* wrong-tool counts
* premature-action counts
* average visible tools per step
* token totals
* step counts
* aggregate summaries
* final plots

Riskier artifacts include:

* raw JSONL traces
* full prompts
* full model outputs
* Bedrock response objects
* request metadata
* cloud logs
* shell history
* notebook outputs with environment paths

## Suggested validation checklist before release

Before tagging a release, run:

```bash
git status
git diff --cached
git ls-files
```

Check for accidentally tracked sensitive files:

```bash
git ls-files | grep -E '(\.pem|\.key|\.env|raw_traces|archived_runs|\.aws|\.log)$'
```

Search tracked files for common secrets or metadata patterns:

```bash
grep -RInE 'AKIA|ASIA|aws_secret|aws_access|session_token|requestId|RequestId|arn:aws|bedrock|/home/|/Users/|\.pem|BEGIN .*PRIVATE KEY' .
```

If any match appears in a public artifact, inspect it manually before committing.

## Release target

Recommended release tag:

```text
v1.0-arxiv
```

Recommended release title:

```text
v1.0-arxiv: Reproducibility package for ToolChoiceConfusion
```

Recommended release description:

```text
This release contains the public reproducibility package for the arXiv version of ToolChoiceConfusion: Causal Minimal Tool Filtering for Reliable LLM Agents.

It includes the experimental runner, analysis scripts, plotting scripts, curated task/tool artifacts, task-level metrics, aggregate summaries, figures, and reproducibility notes.

Raw traces and cloud logs are intentionally excluded.
```
