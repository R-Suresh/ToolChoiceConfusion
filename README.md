# ToolChoiceConfusion: Causal Minimal Tool Filtering

This repository contains the experimental runner for studying whether larger tool menus make LLM agents less reliable.

## Overview

The experiment compares several tool filtering methods:

- all tools
- keyword top-5
- keyword top-10
- state-aware filtering
- full causal path exposure
- causal minimal tool filtering (CMTF)

The benchmark currently uses synthetic multi-step tool-use tasks across calendar, email, and file/document domains.

## Setup

Install dependencies:

    python3 -m pip install --user --upgrade boto3

Set AWS region:

    export AWS_REGION=us-east-1
    export AWS_DEFAULT_REGION=us-east-1

## Run

Set the model list:

    export BEDROCK_MODEL_IDS="amazon.nova-lite-v1:0,amazon.nova-pro-v1:0,anthropic.claude-3-haiku-20240307-v1:0,anthropic.claude-3-sonnet-20240229-v1:0"

Run the experiment:

    python3 scaledExperiment.py

## Analysis

After a run completes, copy the generated results into an `analysis/` folder:

    analysis/task_metrics.csv
    analysis/raw_traces.jsonl

Then generate summary tables:

    python analyze_results.py

This writes:

    tables/summary_by_model_method.csv
    tables/summary_aggregate.csv
    tables/summary_aggregate.tex

To generate plots:

    python plot_results.py

This writes PNG and PDF figures into the `figures/` folder.

    results_scaled/raw_traces.jsonl
    results_scaled/task_metrics.csv

These output files are ignored by Git because they can be regenerated.

## Notes

Do not commit AWS keys, PEM files, logs, or generated result files.
