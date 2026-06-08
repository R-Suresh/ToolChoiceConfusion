# ToolChoiceConfusion: Causal Minimal Tool Filtering for Reliable LLM Agents

This repository contains the reproducibility artifacts and experimental code for:

**ToolChoiceConfusion: Causal Minimal Tool Filtering for Reliable LLM Agents**

arXiv: https://arxiv.org/abs/2606.06284

## Overview

Large language model agents increasingly rely on external tools. However, exposing a model to a larger tool menu can reduce reliability by increasing wrong-tool calls, premature actions, and token cost.

This repository studies **ToolChoiceConfusion**: the failure mode where semantically plausible but causally unnecessary tools distract an LLM agent during multi-step tool use.

The paper evaluates **Causal Minimal Tool Filtering (CMTF)**, a training-free method that exposes only the minimal next-step tool frontier needed to advance from the current task state toward the user goal.

The experiment compares six tool-exposure methods:

1. `all_tools`
2. `keyword_top_5`
3. `keyword_top_10`
4. `state_aware`
5. `full_causal_path`
6. `cmtf`

The benchmark uses synthetic multi-step tool-use tasks across calendar, email, and file/document domains.

## Repository contents

```text
.
├── README.md
├── REPRODUCIBILITY.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── .gitignore
├── code/
│   ├── scaledExperiment.py          # Main experimental runner
│   ├── analyze_results.py           # Generates aggregate CSV and LaTeX tables
│   └── plot_results.py              # Regenerates paper-style figures
├── data/
│   ├── tasks_102.json               # Curated benchmark tasks
│   └── tool_registry_100.json       # Curated tool registry used in the experiments
├── results/
│   ├── task_metrics_main.csv        # Curated task-level metrics from the main run
│   ├── summary_aggregate.csv        # Aggregate results by method
│   └── summary_by_model_method.csv  # Results by model and method
├── figures/
│   ├── success_by_method.png
│   ├── wrong_tools_by_method.png
│   ├── premature_actions_by_method.png
│   ├── tools_per_step_by_method.png
│   └── tokens_by_method.png
├── tables/
│   └── summary_aggregate.tex
└── reproducibility/
    ├── run_config_main.json
    └── environment.md
```

## Setup

Create a Python environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

The experiment runner uses Amazon Bedrock through `boto3`. Configure AWS credentials outside the repository using your normal AWS setup.

Do not commit AWS credentials, `.env` files, PEM files, local cloud configuration, or raw cloud logs.

Example environment variables:

```bash
export AWS_REGION=us-east-1
export AWS_DEFAULT_REGION=us-east-1
```

If your runner uses a configurable model list, set:

```bash
export BEDROCK_MODEL_IDS="amazon.nova-lite-v1:0,amazon.nova-pro-v1:0,anthropic.claude-3-haiku-20240307-v1:0,anthropic.claude-3-sonnet-20240229-v1:0"
```

## Running the experiment

Run the main experiment from the repository root:

```bash
python3 code/scaledExperiment.py
```

By default, the runner writes generated local outputs to:

```text
results_scaled/raw_traces.jsonl
results_scaled/task_metrics.csv
```

These generated outputs are ignored by Git.

Raw traces are not included in this public repository because they may contain model outputs, prompts, request IDs, local paths, or other run-specific metadata.

## Reproducing the analysis tables

The analysis script expects a task-level metrics CSV at:

```text
analysis/task_metrics.csv
```

To reproduce the summary tables from a completed run:

```bash
mkdir -p analysis
cp results_scaled/task_metrics.csv analysis/task_metrics.csv
python3 code/analyze_results.py
```

This writes:

```text
tables/summary_by_model_method.csv
tables/summary_aggregate.csv
tables/summary_aggregate.tex
```

The curated public version of the main task-level metrics is provided as:

```text
results/task_metrics_main.csv
```

If you want to regenerate the tables from the curated public metrics instead of a fresh run, use:

```bash
mkdir -p analysis
cp results/task_metrics_main.csv analysis/task_metrics.csv
python3 code/analyze_results.py
```

## Reproducing the figures

After generating or copying the summary tables, run:

```bash
python3 code/plot_results.py
```

This writes PNG and/or PDF figures into:

```text
figures/
```

The main paper figures can be regenerated from:

```text
tables/summary_aggregate.csv
```

## Artifact map

| Paper artifact               | Repository source                         |
| ---------------------------- | ----------------------------------------- |
| Main experiment runner       | `code/scaledExperiment.py`                |
| Analysis script              | `code/analyze_results.py`                 |
| Plotting script              | `code/plot_results.py`                    |
| Benchmark task definitions   | `data/tasks_102.json`                     |
| Tool registry                | `data/tool_registry_100.json`             |
| Main task-level metrics      | `results/task_metrics_main.csv`           |
| Aggregate method comparison  | `results/summary_aggregate.csv`           |
| Per-model/per-method summary | `results/summary_by_model_method.csv`     |
| LaTeX aggregate table        | `tables/summary_aggregate.tex`            |
| Success-by-method figure     | `figures/success_by_method.png`           |
| Wrong-tool-call figure       | `figures/wrong_tools_by_method.png`       |
| Premature-action figure      | `figures/premature_actions_by_method.png` |
| Visible-tools figure         | `figures/tools_per_step_by_method.png`    |
| Token-cost figure            | `figures/tokens_by_method.png`            |
| Main run configuration       | `reproducibility/run_config_main.json`    |
| Environment notes            | `reproducibility/environment.md`          |

## Public artifact policy

This repository intentionally includes curated reproducibility artifacts rather than raw internal run dumps.

The repository should not include:

* AWS keys or credentials
* `.env` files
* PEM/key files
* Bedrock request IDs
* raw service logs
* unsanitized model traces
* local EC2 paths
* personal notes
* review-submission metadata
* anonymized conference PDFs or OpenReview/AIML submission metadata

Raw traces should only be shared after careful inspection and sanitization. For the public arXiv reproducibility package, derived metrics and aggregate summaries are preferred over raw traces.

## Citation

If you use this repository or build on the paper, please cite:

```bibtex
@misc{sureshbabu2026toolchoiceconfusion,
  title         = {ToolChoiceConfusion: Causal Minimal Tool Filtering for Reliable LLM Agents},
  author        = {Suresh Babu, Rahul and Iyer, Laxmipriya Ganesh},
  year          = {2026},
  eprint        = {2606.06284},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2606.06284}
}
```

## License

This repository is released under the MIT License.
