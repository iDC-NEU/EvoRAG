# Script Guide

This repository contains a small set of tracked launchers and a larger set of
local experiment launchers. The local scripts use machine-specific paths,
datasets, graph spaces, model names, and output directories. They are ignored
by default and are documented here rather than treated as portable entry
points.

## Execution Model

Most experiment scripts follow this sequence:

```text
dataset + graph space
        |
database.insert_triples (optional: rebuild graph for iteration N)
        |
kg_modify.py --option <workflow> --llmbackend <backend>
        |
logs/<algorithm>/{data,stage,triplets,output}
        |
next iteration reads triplets/<space>_<N+1>.json
```

`kg_modify.py` dispatches the selected `--option` to `KGModify/KGModify.py`.
`evolve_basic_forward` and `evolve_basic_feedback` are separate two-stage
operations. `evolve_batch` performs both stages in one invocation: it creates
forward answers, evaluates feedback, updates triplet scores, and writes the
next iteration's triplet file when feedback is enabled.

## Local vLLM Workflow

The in-process backend is [llmragenv/LLM/vllm/client.py](../llmragenv/LLM/vllm/client.py).
It imports `vllm.LLM` directly and reads the model from the ignored local
configuration path:

```yaml
LLM:
  vllm:
    <model-name>:
      template_format: Qwen2.5 # or Llama / Qwen3
      modelpath: /absolute/path/to/model
```

It is not an OpenAI-compatible vLLM server. The client constructs an in-process
engine with model-specific parallelism and GPU-memory settings. The process
therefore needs enough local GPU capacity before it is started.

### Recommended Complete Forward/Feedback Entrypoint

`run_batch.sh` is the tracked launcher for a complete RGB loop. Its active
configuration selects `rgb`, NebulaGraph space `rgb_zyz`, the `basic_batch`
algorithm, `Qwen2.5-32B-Instruct`, and `--llmbackend vllm`. For every iteration
it imports the current scored graph and invokes `--option evolve_batch`.

`a_run_evolve_vllm_batch.sh` is the equivalent local, ignored experiment
launcher. Its active configuration runs the RGB dataset with a local vLLM model
and `--option evolve_batch` for a configured iteration range. It is the best
reference for a local experiment variant, but its variables must be reviewed
before use.

Do not confuse these scripts with:

- `a_run_evolve_vllm.sh`: active command runs only
  `evolve_basic_forward`; feedback commands are commented out.
- `run_two_stage.sh`: explicitly runs forward then feedback, but active feedback
  uses `llama_index`, not vLLM.
- `a_vllm.sh` and `llmragenv/vllm_batch.py`: throughput benchmarks using
  historical staged data and hardcoded local paths, not the EvoRAG graph-update
  workflow.

## Script Inventory

| Group | Scripts | Purpose |
| --- | --- | --- |
| Curated core launchers | `run_batch.sh`, `run_noise.sh`, `run_case_entity.sh`, `run_case_pruning.sh` | Tracked entrypoints for vLLM batch evolution, feedback-noise runs, entity-count sweeps, and pruning sweeps. |
| Local evolution variants | `a_run_evolve.sh`, `a_run_evolve_test.sh`, `a_run_evolve_vllm.sh`, `a_run_evolve_vllm_batch.sh`, `run_two_stage.sh`, `run_llm_only_batch.sh` | Forward/feedback variants, local vLLM experiments, short tests, explicit two-stage execution, and LLM-only ablations. |
| Batch backend variants | `run_batch_api.sh`, `run_batch_api_abaltion.sh`, `run_batch_api_f1.sh`, `run_batch_api_f1_hotpotqa.sh`, `run_batch_api_f1_multihop.sh`, `run_batch_api_human.sh`, `run_batch_api_human_hotpotqa.sh`, `run_batch_api_human_multihop.sh`, `run_batch_deepseek.sh`, `run_batch_other.sh`, `run_batch_test.sh` | `evolve_batch` variants for API backends, feedback-disabled/F1/human-alignment/ablation experiments, and alternate datasets. |
| HotpotQA-1M experiments | `run_batch_hotpotqa1M_train.sh`, `run_batch_hotpotqa1M_case.sh`, `run_batch_hotpotqa1M_v2_train.sh`, `run_batch_hotpotqa1M_v2_case.sh`, `run_batch_hotpotqa1M_v2_depcache_case.sh`, `run_batch_hotpotqa1M_v2_iteration_case.sh`, `run_batch_hotpotqa1M_v3_train.sh`, `run_batch_hotpotqa1M_v3_case.sh` | Large-dataset training, only-error continuation, case studies, dependency-cache tests, and V2/V3 iteration variants. The active V2/V3 scripts currently use API backends despite comments mentioning vLLM. |
| Baseline and reproduction | `run_baseline.sh`, `run_baseline_case.sh`, `run_baseline_data.sh`, `run_reproduce.sh`, `run_v2.sh`, `run_dragonball.sh`, `run_llama8b.sh`, `run_hotpotqa_1.sh`, `run_hotpotqa_1_qwen3_reproduce.sh`, `run_multihop.sh`, `run_rgb.sh`, `run_rgb_qwen3_reproduce.sh`, `run_similar_question.sh`, `run_bak.sh` | Rebuild graph data, reproduce older local/API experiments, and run RGB, MultiHop, HotpotQA, or DragonBall baselines. |
| RAGCache and redundancy | `run_ragcache.sh`, `run_ragcache_entity.sh`, `run_ragcache_for_data.sh`, `run_ragcache_vllm.sh`, `run_redundancy.sh`, `run_redundancy_my_qwen.sh`, `run_redundancy_my_qwen_v2.sh`, `run_redundancy_reduction.sh` | RAGCache baselines/data preparation and redundancy scoring/removal experiments. `run_ragcache_vllm.sh` currently has an API backend active; its vLLM blocks are comments. |
| External-RAG evaluation | `run_LightRAG_api.sh`, `run_MsGraphRAG_api.sh`, `run_msgraphrag.sh`, `run_prompt_acc_deepseek.sh`, `run_prompt_acc_time.sh` | Run accuracy/time comparisons for LightRAG, MsGraphRAG, and related methods with API or local backends. |
| Analysis and utility | `a_vllm.sh`, `run_cost_align_human.sh`, `run_statistics_change.sh`, `run_case_pruning35_data.sh`, `run_case_pruning_bak.sh`, `run_api.sh`, `run_api_rgb.sh`, `run_api_hotpotqa.sh`, `run_api_test.sh`, `run_tmp.sh` | vLLM throughput benchmark, cost/human-alignment analysis, score-change statistics, data sweeps, API graph preparation/tests, and temporary one-off work. |
| Graph service | `database/graph/nebula-start.sh` | Starts the local NebulaGraph service helper. Review its Docker and storage settings before use. |

## Script Status and Git Policy

Only the four curated core launchers are tracked. All other root launchers are
local experiment artifacts and remain ignored by `*.sh`; new `.bash`, `.zsh`,
and `.command` launchers are ignored as well. Logs, datasets, generated figures,
archives, local credentials, and the local `paper/` export directory are also
ignored.

Before promoting a local launcher into version control, remove machine-specific
paths, credentials, and fixed output directories, then add it explicitly and
document its expected dataset, model, graph space, and algorithm.
