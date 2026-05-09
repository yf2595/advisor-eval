# Advisor Strategy Evaluation Framework

This repository implements and benchmarks the **advisor strategy** for LLM
agents: a small / cheap *executor* model handles most of the trajectory and a
larger *advisor* model is consulted only when needed, governed by an
**escalation policy**. The framework is inspired by Anthropic's
[Advisor Strategy](https://claude.com/blog/the-advisor-strategy).

The reported results are produced on **HotpotQA (fullwiki)** and **GAIA**
(validation, agentic tool use). The same evaluator also exposes **GSM8K** and
**HotpotQA (distractor)** for sanity checks. Every run produces a fully
self-contained JSONL trajectory file with the per-task question, prediction,
gold answer, full tool trace, advisor guidance, cost, and latency.

> **Note on `results/`:** the JSONL files shipped in this repository are
> **illustrative samples** taken from intermediate runs during development —
> they are *not* the full set of trajectories used to compute the numbers in
> the paper. They are included so reviewers can inspect the exact record
> shape, the tool traces, and the advisor-guidance structure produced by the
> framework. To reproduce the paper's headline numbers, rerun the matrix
> scripts described in [Reproducing the paper matrices](#reproducing-the-paper-matrices).

## Repository layout

```
.
├── advisor.py                       # Advisor agent (planner / structured guidance)
├── analysis.py                      # Aggregate metrics, plots, paired analyses
├── config.yaml                      # Models, costs, dataset budgets, policy params
├── datasets_loader.py               # GSM8K / HotpotQA (distractor + fullwiki) / GAIA
├── evaluator.py                     # cheap / strong / advisor evaluation loops + judges
├── executor.py                      # Executor agent (cheap model, multi-step ReAct)
├── gaia_runner.py                   # Agentic GAIA loop with tool calling + advisor hooks
├── hotpotqa_runner.py               # Agentic HotpotQA-fullwiki loop with wiki tools
├── logger.py                        # JSONL logger, cost & latency tracker
├── policies.py                      # Escalation policies (model_driven, self_eval, ...)
├── run.py                           # Single CLI entry point
├── run_hotpot_fullwiki_matrix.py    # Reproduces the HotpotQA-fullwiki paper matrix
├── run_gaia_failure_or_conf_matrix.py # Reproduces the GAIA dual-signal matrix
├── data/                            # Stratified-sample manifests (HotpotQA-fullwiki seed 42)
├── results/                         # Illustrative JSONL trajectories + run manifest (sample, not the full paper data)
├── requirements.txt
└── README.md
```

`results/` contains a **subset of trajectories** captured during development —
included so reviewers can inspect the JSONL record shape, the tool traces,
and the advisor-guidance format that the framework produces. They are *not*
the full set of trajectories used to compute the paper's headline numbers.
The matrix scripts under
[Reproducing the paper matrices](#reproducing-the-paper-matrices) regenerate
fresh runs at the configured sample sizes; plots and aggregate tables are
then recomputed with `python run.py --analyze results/`.

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
```

Python 3.10+ is required.

GAIA and HotpotQA-fullwiki are gated on HuggingFace — either run
`huggingface-cli login` (and request access for `gaia-benchmark/GAIA`), or
pass a local JSONL export via `--gaia-local-file` / `--hotpotqa-fullwiki-local-file`.

## Quick start

A single experiment is one combination of `(dataset, method, [policy])`. If
`--samples` is omitted, the framework uses the dataset-specific default from
`config.yaml`:

| Dataset             | Default `--samples` |
| ------------------- | ------------------- |
| `gaia`              | **165** (full GAIA validation split) |
| `hotpotqa`          | **300** |
| `hotpotqa_fullwiki` | **300** |
| `gsm8k`             | 200                 |

```bash
# Cheap-only baseline on the full 165-task GAIA validation set
python run.py --dataset gaia --method cheap

# Strong-only baseline on the full GAIA validation set
python run.py --dataset gaia --method strong

# Advisor with the executor-driven policy (Anthropic-style), GAIA n=165
python run.py --dataset gaia --method advisor --policy model_driven

# HotpotQA fullwiki with the dual-signal policy, n=300
python run.py --dataset hotpotqa_fullwiki --method advisor \
              --policy failure_or_conf_t0.75
```

Pass `--samples N` to override the default for a quick smoke run, e.g.
`--samples 25`.

Each call appends one JSONL file to `results/` named
`<dataset>_<method>_<policy>_<UTCstamp>.jsonl`. Records contain the question,
prediction, gold answer, full tool trace, advisor guidance, per-call cost,
per-call latency, and metadata sufficient to reproduce the run.

To regenerate all aggregate tables and plots from existing JSONLs:

```bash
python run.py --analyze results/
```

This (re)creates `plots/` with the figures listed in the **Output** section.

## Configuration (`config.yaml`)

The default models, costs, and budgets live in `config.yaml`:

| Setting                                  | Default                                  |
| ---------------------------------------- | ---------------------------------------- |
| `models.executor`                        | `gpt-5.4-nano`                           |
| `models.advisor`                         | `gpt-5.4`                                |
| `datasets.gaia_samples`                  | 165 (full GAIA validation split)         |
| `datasets.hotpotqa_samples`              | 300                                      |
| `datasets.hotpotqa_fullwiki_samples`     | 300                                      |
| `datasets.gsm8k_samples`                 | 200                                      |
| `gaia.max_tool_calls`                    | 12                                       |
| `gaia.max_advisor_calls`                 | 2                                        |
| `hotpotqa_fullwiki.max_tool_calls`       | 12                                       |
| `hotpotqa_fullwiki.max_advisor_calls`    | 2                                        |
| `policies.threshold`                     | 0.6 (`self_eval` default)                |
| `policies.random_prob`                   | 0.3                                      |
| `policies.fixed_interval`                | 3                                        |

Token prices (per 1M tokens, taken from the official OpenAI pricing page) are
declared per model under `costs:`. The current and the legacy `gpt-4.1`
families are both listed so older JSONLs in `results/` keep producing
correct cost rollups when re-analysed.

You can override the model on the command line without editing the YAML:

```bash
python run.py --dataset gaia --method advisor --policy model_driven \
              --executor-model gpt-4.1-mini --advisor-model gpt-5.4 \
              --samples 100
```

## Escalation policies

| Policy name                    | Trigger                                                                                              |
| ------------------------------ | ---------------------------------------------------------------------------------------------------- |
| `model_driven`                 | Executor emits `[REQUEST_ADVISOR]` (Anthropic-style)                                                 |
| `self_eval` / `self_eval_t<x>` | Executor's self-rated confidence < threshold (`x`, default 0.6)                                      |
| `failure_based`                | Empty / parse-error output, repeated dead-ends, tool error                                           |
| `failure_or_conf_t<x>`         | **Dual-signal hybrid**: `failure_based` OR (confidence < `x`)                                        |
| `random_prob`                  | Random with probability `policies.random_prob`                                                       |
| `fixed_interval`               | Every `policies.fixed_interval` steps                                                                |

`failure_or_conf_t<x>` is the dual-signal policy used for the headline result
in the paper. It logs the trigger reason on every escalation
(`model_requested`, `low_evidence_final`, `auto_low_confidence_final`, etc.)
in the resulting JSONL.

## CLI reference

```
python run.py [OPTIONS]

Required-ish:
  --dataset {gsm8k,hotpotqa,hotpotqa_fullwiki,gaia}
  --method  {cheap,strong,advisor}
  --policy  {model_driven,self_eval,self_eval_t<x>,failure_based,
             failure_or_conf_t<x>,random_prob,fixed_interval,no_advisor}
  --samples N        Number of tasks. Defaults to the per-dataset value in
                     config.yaml (HotpotQA=300, GAIA=165, GSM8K=200).

Models / config:
  --config PATH                       Path to config.yaml (default: config.yaml)
  --executor-model NAME               Override config models.executor
  --advisor-model NAME                Override config models.advisor
  --cache                             Enable disk cache of API responses (.cache/)

Dataset-specific:
  --gaia-split, --gaia-dataset-id, --gaia-config-name,
  --gaia-local-file, --max-tool-calls
  --hotpotqa-fullwiki-local-file, --hotpotqa-fullwiki-seed,
  --hotpotqa-fullwiki-split

Convenience:
  --all                               Run experiments 1, 2, 3, 5 (see below)
  --advisor-only                      With --all, skip cheap/strong baselines
  --gaia-smoke-trio                   Cheap + strong + advisor(model_driven) on GAIA
  --analyze DIR                       Recompute metrics + plots from a results dir
  --check-smoke-gates DIR             CI gate: strong > cheap and advisor >= cheap
```

`--all` runs:

1. **Core baselines** — `cheap` and `strong`
2. **Policy comparison** — `random_prob`, `failure_based`, `self_eval`
3. **Self-eval threshold sweep** — `self_eval` at thresholds 0.3 / 0.5 / 0.7 / 0.9
4. **Difficulty split** — computed during analysis from (1)
5. **Ablation** — multi-step executor with the advisor disabled (`no_advisor`)

## Reproducing the paper matrices

The paper's headline numbers come from running the two matrix scripts below
end-to-end. Both scripts read their sample count from `config.yaml`, so they
default to **300 tasks** for HotpotQA-fullwiki and **165 tasks** (full
validation split) for GAIA. Pass `--samples N` to override.

> The JSONLs already in `results/` are **illustrative samples** captured
> during development at smaller `n` (≈ 100 per run), and they include reruns
> and partial sweeps. They are *not* the full paper trajectories. To
> regenerate the full set, run the scripts below; each invocation writes
> fresh JSONLs and a new manifest under `results/` without touching the
> existing files.

### HotpotQA fullwiki (default n = 300, stratified seed 42)

```bash
python run_hotpot_fullwiki_matrix.py                # 300 tasks (config default)
python run_hotpot_fullwiki_matrix.py --samples 100  # smaller smoke run
```

Runs the full matrix (`cheap`, `strong`, plus advisor under `failure_based`,
`model_driven`, `self_eval_t0.25/0.5/0.75`, `failure_or_conf_t0.75`) and
writes `results/hotpotqa_fullwiki_matrix_manifest_<UTCstamp>.json` listing
the JSONL files for that run.

### GAIA dual-signal study (default n = 165, full validation split)

```bash
python run_gaia_failure_or_conf_matrix.py               # 165 tasks (config default)
python run_gaia_failure_or_conf_matrix.py --samples 50  # smaller smoke run
```

Runs `failure_based` vs `failure_or_conf_t0.75` for three executors
(`gpt-5.4-nano`, `gpt-4.1-mini`, `gpt-4.1`) sharing the same advisor.

### Single-policy GAIA matrix used for the dual-executor study

The single-policy GAIA matrix in the paper was produced by repeated
`run.py` invocations with `--executor-model {gpt-4.1-mini, gpt-5.4-nano}`
over the policies `{cheap, strong, model_driven, failure_based,
random_prob, self_eval_t0.25, self_eval_t0.5, self_eval_t0.75}`. The
`gaia_*_20260423_*.jsonl` and `gaia_*_20260424_*.jsonl` files in
`results/` are leftover samples from those development runs.

## Output

- **`results/`** — one JSONL trajectory file per run, plus the HotpotQA
  fullwiki run manifest. Each row is one task and contains:
  `task_id`, `dataset`, `method`, `question`, `prediction`,
  `ground_truth`, `correct`, `cost_total`, `latency_total_s`,
  `advisor_calls`, `tool_calls`, `tool_errors`, `steps`, `tool_trace`,
  `advisor_guidance`, `run_metadata` (executor / advisor model, policy,
  budgets, seed). The files shipped with the repo are illustrative samples
  (see the note in [Reproducing the paper matrices](#reproducing-the-paper-matrices)),
  not the full paper data.

- **`plots/`** — regenerated by `python run.py --analyze results/`.
  Includes:
  - `three_axis_summary.png` — accuracy / cost / latency grouped bars
  - `cost_vs_accuracy.png`, `latency_vs_accuracy.png`, `cost_vs_latency.png`
  - `latency_distribution.png` — per-method latency boxplots
  - `policy_comparison.png` — accuracy by policy
  - `difficulty_split.png` — recovery rate on cheap-wrong tasks
  - `gaia_success_vs_tool_error.png`, `gaia_advisor_rescue.png` — GAIA
    diagnostics

The aggregate table is also printed to stdout when `--analyze` runs.

## Key metrics

- **Accuracy** — exact match (GSM8K numeric, HotpotQA case-insensitive
  normalised string) and GAIA's official normalised exact match, with an
  LLM-judge fallback for paraphrases.
- **Cost** — per-token pricing from `config.yaml`, split into executor and
  advisor.
- **Latency** — wall-clock `time.perf_counter()` per API call, split into
  executor and advisor; reported as mean / median / p95.
- **Advisor usage** — call count, per-step escalation rate, mean
  `advisor_calls` per task.
- **Recovery rate** — fraction of cheap-wrong tasks fixed by the advisor run.
- **Tool reliability (GAIA / HotpotQA-fullwiki)** — tool error rate,
  advisor-rescue rate, and `followed_advice` (executor took the advisor's
  first NEXT step).

## Notes on each benchmark

- **GAIA** runs in an agentic loop with `web_search`, `web_fetch_url`,
  `wiki_search`, `wiki_lookup`, `calculator`, `python_eval`, and
  `read_attachment`. Advisor escalation is checked at every planning step,
  including after tool failures. JSONL records include the full trigger
  histogram per task.
- **HotpotQA fullwiki** uses the same agentic loop with wiki-only tools and
  a stratified sample (50/50 bridge vs comparison) frozen by
  `data/hotpotqa_fullwiki_manifest_seed42.json`. The shipped JSONLs in
  `results/` were collected at n = 100 (seed 42); the current default is
  n = 300 from the same manifest.

## Reproducibility

Every JSONL trajectory the framework writes — both the illustrative samples
under `results/` and any new runs — is seedable and self-contained: every
record carries `run_metadata` with the executor model, advisor model,
policy, sampling seed, and per-task budgets. Re-running the command stored
in `run_metadata` against the same dataset version reproduces the
trajectory up to OpenAI's own non-determinism. The paper's headline numbers
are reproduced by running the matrix scripts in
[Reproducing the paper matrices](#reproducing-the-paper-matrices) at the
default sample sizes (HotpotQA-fullwiki = 300, GAIA = 165) and then
calling `python run.py --analyze results/`.
