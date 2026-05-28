# Reflective Prompt Tuning: Function-Calling Prompt Optimization

This repository contains the cleaned implementation for the paper
[Reflective Prompt Tuning through Language Model Function-Calling](https://arxiv.org/abs/2605.21781).

RPT automates prompt improvement by using an optimizer LLM with function-calling abilities to invoke diagnostic tools, analyze the target model’s failures, and produce structured evaluation reports. Each report, together with the history of earlier reports, is fed back to the optimizer, which iteratively refines the target prompt.

![RPT overview](figs/RPT_overview.png)

---

## Reflective Prompt Tuning

RPT is a function-calling workflow for iterative prompt optimization. Given a task, a target model, and an initial prompt program, the system repeatedly:

1. evaluates the current prompt on an optimization split,
2. diagnoses recurring failure modes,
3. summarizes calibration and task metrics,
4. asks an optimizer model to propose a prompt patch or stop,
5. selects the final prompt using held-out validation performance.

The cleaned repository supports three tasks:

- `hotpotqa`
- `livebench_math`
- `xbrl_formula`

---

## Repository Structure

```text
.
├── rpt/
│   ├── analysis/
│   │   ├── cluster_failures_and_patches.py
│   │   ├── cluster_fusion.py       # ClusterFusion topic extraction
│   │   ├── interpret_data_using_heatmaps.py
│   │   ├── paths.py                # Analysis path resolution
│   │   └── performance_summarization_and_analysis.py
│   ├── tasks/
│   │   ├── hotpotqa.py             # OpenAI optimizer for HotpotQA
│   │   ├── hotpotqa_gemini.py      # Gemini optimizer for HotpotQA
│   │   ├── livebench_math.py       # OpenAI optimizer for LiveBench Math
│   │   ├── livebench_math_gemini.py
│   │   ├── xbrl_formula.py         # OpenAI optimizer for XBRL Formula
│   │   └── xbrl_formula_gemini.py
│   ├── common.py                 # JSONL logging, JSON helpers, shared file utilities
│   ├── gemini_utils.py           # Gemini client, structured parsing, cleaned-log helpers
│   └── paths.py                  # Repository and dataset path configuration
├── data/
│   ├── hotpotqa/                 # Cached HotpotQA train/dev/test splits
│   ├── livebench_math/           # Cached LiveBench Math train/val/test splits
│   └── xbrl_formula/             # Cached XBRL Formula train/val/test splits
└── run_analysis_pipeline.sh      # Analysis pipeline entrypoint
```

Generated artifacts are ignored by git: `logs/`, `clustering_results/`, `vis_results/`, `results/`, and `analysis_reports/`.

---

## Requirements

Use Python `>= 3.10`.

Install dependencies:

```bash
pip install -r requirements.txt
```

For editable local development, optionally create a virtual environment first:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## API Keys

Set only the credentials for the backend you plan to use.

For OpenAI target or optimizer runs:

```bash
export OPENAI_API_KEY="..."
```

For Gemini optimizer runs through Vertex AI:

```bash
export GOOGLE_CLOUD_PROJECT="..."
export GOOGLE_CLOUD_LOCATION="global"
```

---

## Datasets

The repository is organized around cached local JSONL splits. Dataset licenses and redistribution terms are governed by the upstream dataset providers.

| Dataset | Local path | Split files | Upstream reference |
| --- | --- | --- | --- |
| HotpotQA | `data/hotpotqa/` | `train.jsonl`, `dev.jsonl`, `test.jsonl` | [HotpotQA](https://hotpotqa.github.io/) / [Hugging Face](https://huggingface.co/datasets/hotpotqa/hotpot_qa) |
| LiveBench Math | `data/livebench_math/` | `train.jsonl`, `val.jsonl`, `test.jsonl` | [LiveBench](https://livebench.ai/) / [Hugging Face](https://huggingface.co/datasets/livebench/math) |
| XBRL Formula | `data/xbrl_formula/` | `train.jsonl`, `val.jsonl`, `test.jsonl` | [ACE finance data](https://github.com/ace-agent/ace/tree/main/eval/finance/data) |

Dataset paths can be overridden with environment variables:

```bash
export RPT_DATA_ROOT="/path/to/data"
export RPT_HOTPOTQA_DATA_DIR="/path/to/hotpotqa"
export RPT_LIVEBENCH_MATH_DATA_DIR="/path/to/livebench_math"
export RPT_XBRL_FORMULA_DATA_DIR="/path/to/xbrl_formula"
```

---

## Data Source Attribution

The cached splits in this repository build on the following data sources:

1. HotpotQA
   * Source: [HotpotQA](https://hotpotqa.github.io/) and the [hotpotqa/hotpot_qa](https://huggingface.co/datasets/hotpotqa/hotpot_qa) Hugging Face mirror.
   * License: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
   * Local files: `data/hotpotqa/train.jsonl`, `data/hotpotqa/dev.jsonl`, and `data/hotpotqa/test.jsonl`.

2. LiveBench Math
   * Source: [LiveBench](https://livebench.ai/) and the [livebench/math](https://huggingface.co/datasets/livebench/math) Hugging Face dataset.
   * License: [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
   * Local files: `data/livebench_math/train.jsonl`, `data/livebench_math/val.jsonl`, and `data/livebench_math/test.jsonl`.

3. XBRL Formula
   * Source: [ACE finance data](https://github.com/ace-agent/ace/tree/main/eval/finance/data)
   * License: [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
   * Local files: `data/xbrl_formula/train.jsonl`, `data/xbrl_formula/val.jsonl`, `data/xbrl_formula/test.jsonl`.

Please refer to the respective upstream sources for complete licensing terms and attribution requirements.

---

## Quick Start

Run an OpenAI optimizer:

```bash
python -m rpt.tasks.hotpotqa --iters 20
python -m rpt.tasks.xbrl_formula --iters 20
python -m rpt.tasks.livebench_math --iters 20
```

Run a Gemini optimizer:

```bash
python -m rpt.tasks.hotpotqa_gemini --iters 20 --optimizer_name gemini-3.1-pro
python -m rpt.tasks.livebench_math_gemini --iters 20 --optimizer_name gemini-3.1-pro
python -m rpt.tasks.xbrl_formula_gemini --iters 20 --optimizer_name gemini-3.1-pro
```

Prepare or inspect cached LiveBench Math splits without running optimization:

```bash
python -m rpt.tasks.livebench_math --prepare_only
```

Evaluate the seed prompt only:

```bash
python -m rpt.tasks.hotpotqa --evaluate_only
python -m rpt.tasks.livebench_math --evaluate_only
```

---

## Analysis Pipeline

Run the analysis pipeline for an existing log:

```bash
./run_analysis_pipeline.sh \
  --log_path logs/xbrl_formula/gpt-5/example.jsonl \
  --task_name xbrl_formula \
  --model_name gpt-5
```

The pipeline can generate:

- failure and patch corpora,
- ClusterFusion topics,
- human-readable topic labels,
- transition and persistence summaries,
- heatmaps and prompt-length plots.

---

## Outputs

RPT runs write JSONL logs containing prompt programs, train/dev/test metrics, diagnostic reports, decisions, and final evaluations. Analysis scripts write derived artifacts into task/model/log-specific subdirectories.

Common output locations:

- `logs/`
- `clustering_results/`
- `vis_results/`
- `analysis_reports/`
- `results/`

---

## Citation

If you use this repository, please cite:

```bibtex
@article{bayat2026reflectiveprompttuning,
  title = {Reflective Prompt Tuning through Language Model Function-Calling},
  author = {Farima Fatahi Bayat and Moin Aminnaseri and Pouya Pezeshkpour and Estevam Hruschka},
  year = {2026},
  url = {https://arxiv.org/abs/2605.21781}
}
```

---

## Disclosure

Embedded in or used by this repository are open source software components, datasets, model APIs, and other third-party materials. Each component remains governed by its own license, terms of use, and redistribution conditions. Those terms continue to apply to the corresponding portions of this repository and to any downstream use.

You may receive, distribute, or modify open source code in this repository only under the terms of the applicable open source licenses. If any project terms conflict with a third-party open source or dataset license, the third-party license controls for that component or dataset.

Do not redistribute dataset materials unless the relevant dataset license permits it. If a public release requires dataset pointers instead of bundled files, remove the cached JSONL files and provide links to the original sources in the dataset table above. Derived datasets should retain attribution and links to their upstream sources.

All third-party components, datasets, and model-service integrations are provided without warranty, including implied warranties of merchantability or fitness for a particular purpose. Verify licenses, citations, and usage permissions before publishing, redistributing, or pushing this repository.
