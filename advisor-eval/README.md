# When to Ask, What to Say, and When Listening Hurts: Communication Between Asymmetric LLM Agents

Code, prompts, configurations, and trajectory logs for the anonymous NAACL 2027
submission. Anonymous repository: <https://anonymous.4open.science/r/advisor-eval-0A48>

A small **executor** model solves an agentic task with tools. It can consult a
stronger **advisor** that is only allowed to *advise*: the advisor never calls
tools and never gives the final answer, so everything it contributes must pass
through a natural-language message. We study this exchange as a communication
problem with three decisions:

| Research question | Decision | What we vary |
|---|---|---|
| **RQ1: When to ask?** | the initiation policy π(s<sub>t</sub>) | Monitor, Report (confidence < τ), Hybrid, Self-initiated, Random, No channel |
| **RQ2: What to say?** | the message format μ | DNA (DIAGNOSE / AVOID / NEXT) and its subsets, free-form, action hand-off, answer hint, Self-DNA |
| **RQ3: When does listening hurt?** | whether the executor follows the advice | help and harm per task, advisor swaps across model families |

The repository implements every configuration in the paper. It covers five
same-family executor–advisor pairs and three cross-family pairs, on **GAIA**
(full validation set, 165 tasks) and **HotpotQA** (300 questions).

**To reproduce the paper**, see
[Reproducing the paper, step by step](#reproducing-the-paper-step-by-step).
It covers installation, credentials, starting the local models with vLLM, the
command for each paper table, and rebuilding the tables from the logs.

## Main results

All numbers below are from the paper (seed 42 unless stated). The help/harm
counts compare each advised run with the executor-alone run on the same tasks.
A task is a **help** event if it is wrong alone and right with advice, and a
**harm** event if it is right alone and wrong with advice.

Pairs are named by their executor:

| Pair | Executor | Advisor |
|---|---|---|
| Nano | GPT-5.4 Nano | GPT-5.4 |
| Mini | GPT-4.1 Mini | GPT-5.4 |
| 4.1 | GPT-4.1 | GPT-5.4 |
| Qwen | Qwen3.5-9B | Qwen3.6-27B |
| OSS | gpt-oss-20b | gpt-oss-120b |

### RQ1: when the channel opens matters as much as who is listening

Table 1 of the paper: accuracy (%) by initiation policy.

| Initiation policy | GAIA Nano | GAIA Mini | GAIA 4.1 | GAIA Qwen | GAIA OSS | HotpotQA Nano | HotpotQA Mini | HotpotQA 4.1 | HotpotQA Qwen | HotpotQA OSS |
|---|---|---|---|---|---|---|---|---|---|---|
| No channel (executor alone) | 20 | 26 | 31 | 23 | 27 | 61 | 69 | 73 | 61 | 64 |
| Random | 18 | 31 | 36 | 27 | 30 | 65 | 71 | 78 | 64 | 67 |
| Monitor | 39 | 42 | 43 | 31 | 31 | 79 | 77 | 84 | 70 | 72 |
| Report (τ = 0.75) | 45 | 46 | 48 | 35 | **38** | 80 | 81 | 84 | 75 | 80 |
| Hybrid (monitor or report) | **48** | **48** | **50** | **36** | **38** | 78 | 82 | 87 | 75 | 80 |
| Self-initiated | 40 | 46 | 46 | 35 | 37 | **82** | **85** | **89** | **77** | **81** |
| Advisor alone | 55 | 55 | 55 | 44 | 49 | 78 | 78 | 78 | 76 | 76 |

- **The initiation policy can matter more than the executor.** For Nano on
  GAIA, changing the policy moves accuracy from 18% (Random) to 48% (Hybrid).
  Replacing Nano with the larger GPT-4.1 and no advice moves it only from 20%
  to 31%.
- **The best initiator depends on where the evidence of failure is.**
  - On HotpotQA, failures are mostly silent, and the executor's own request is
    best in **14 of 14** settings (every pair, seed, and advisor tested).
  - On GAIA, failures are mostly visible tool errors. There the Hybrid trigger
    is best for the GPT executors, and ties with self-initiated requests for
    the open-weight ones.
  - Across seeds 42/43/44, accuracy moves by at most 2 points (Appendix Table 7).
- **Cost.** For Nano and Mini, advised runs cost 67–81% less than running
  GPT-5.4 alone.

### RQ2: the communicative acts matter more than their surface form

Table 2 of the paper: hybrid initiation and the same budget. Cells are
accuracy (%) with (help / harm).

| Channel | Speaker | GAIA Nano | GAIA Qwen | GAIA OSS | HotpotQA Nano | HotpotQA Qwen | HotpotQA OSS | Tokens per message (Nano) |
|---|---|---|---|---|---|---|---|---|
| No channel | – | 20 | 23 | 27 | 61 | 61 | 64 | – |
| Self-DNA | executor | 20 (0/0) | 24 (2/0) | 26 (1/3) | 62 (2/0) | 63 (6/0) | 63 (0/3) | 161 |
| Free-form | advisor | 35 (36/11) | 29 (24/14) | 30 (16/11) | 67 (28/10) | 69 (35/11) | 72 (34/10) | 165 |
| Free-form, length-matched | advisor | 34 (34/11) | 29 (25/15) | 29 (16/12) | 67 (28/10) | 69 (35/11) | 71 (32/11) | 166 |
| **DNA, compact (default)** | advisor | 48 (53/7) | 36 (30/9) | **38** (26/8) | 78 (58/7) | **75** (51/9) | 80 (57/9) | **69** |
| DNA, full sentences | advisor | **50** (66/16) | **37** (32/9) | **38** (25/7) | **79** (59/6) | **75** (51/9) | **81** (60/9) | 172 |
| Action hand-off | advisor | 39 (47/16) | 30 (23/11) | 29 (16/13) | 65 (23/11) | 67 (30/12) | 69 (30/15) | 38 |
| Answer hint | advisor | 43 (48/10) | 34 (27/9) | 35 (22/9) | 74 (47/8) | 73 (45/9) | 77 (49/10) | 169 |
| Advisor alone | – | 55 | 44 | 49 | 78 | 76 | 76 | – |

- **The gain requires a stronger speaker.** When the executor writes the DNA
  message to itself, accuracy stays within 2 points of the executor alone.
- **Acts, not prose.**
  - Compact DNA and full-sentence DNA differ by at most 2 points, while
    compact DNA uses about 40% of the tokens.
  - Free-form advice stays 6–13 points below compact DNA, and matching its
    length to DNA does not close the gap.
- **Explaining beats acting or answering.**
  - Handing over the next tool call is 6–13 points below compact DNA, and
    it causes the most harm on HotpotQA.
  - Giving the advisor's own answer is 2–5 points below compact DNA.

**Every act contributes** (DNA ablation, Appendix Table 12). Full DNA is the
most accurate configuration in all eight pair–benchmark settings. Averaged over
pairs, removing an act costs:

| Act removed | Accuracy lost on GAIA (points) | Accuracy lost on HotpotQA (points) | Harm |
|---|---|---|---|
| NEXT | 6.8 | 4.7 | |
| DIAGNOSE | 4.8 | 3.7 | |
| AVOID | 2.4 | 2.7 | rises in all 8 settings (Nano on GAIA: 7 → 10) |

### RQ3: harm is rare but real, and a stronger advisor mainly adds help

- **Help and harm.** Advice rescues 16–32% of GAIA tasks and breaks 2–6%.
  With the hybrid policy, the GAIA help/harm counts are:
  - Nano 53/7, Mini 41/5, 4.1 37/6 (ratios of 6–9×);
  - Qwen 30/9, OSS 26/8 (about 3×).

  The best single-signal policy and the hybrid improve significantly over the
  executor alone in every pair (exact McNemar test, p ≤ 0.003).
- **Advisor swaps.** Table 3 of the paper keeps the executor fixed and changes
  only the advisor. GAIA uses the hybrid trigger and HotpotQA uses
  self-initiated requests. Cells are accuracy (%) with (help / harm); † marks a
  cross-family pair.

| Executor | Advisor | Advisor alone (GAIA) | GAIA | HotpotQA |
|---|---|---|---|---|
| Qwen3.5-9B | none | – | 23 | 61 |
| Qwen3.5-9B | Qwen3.6-27B | 44 | 36 (30/9) | 77 (54/6) |
| Qwen3.5-9B | GPT-5.4† | 55 | 39 (35/9) | 81 (67/7) |
| gpt-oss-20b | none | – | 27 | 64 |
| gpt-oss-20b | gpt-oss-120b | 49 | 38 (26/8) | 81 (58/7) |
| gpt-oss-20b | GPT-5.4† | 55 | 40 (29/8) | 82 (62/8) |
| GPT-5.4 Nano | none | – | 20 | 61 |
| GPT-5.4 Nano | GPT-5.4 | 55 | 48 (53/7) | 82 (67/4) |
| GPT-5.4 Nano | Qwen3.6-27B† | 44 | 43 (47/9) | 77 (55/7) |

  Across the six comparisons, a stronger advisor raises accuracy every time.
  Help changes by 3–13 tasks, while harm changes by at most 3. On GAIA, harm
  stays at 7–9 tasks whichever advisor is used.

## Reproducing the paper, step by step

Reproducing the paper means running all 211 conditions, with every task logged
to `runs/`, and then rebuilding the tables from those logs. The conditions fall
into three groups that can run independently, on different machines if needed:

| Group | Pairs | Backend | Conditions | Task runs |
|---|---|---|---|---|
| **API** | Nano, Mini, 4.1, GPT-5.4 alone | OpenAI API | 97 | 21,405 |
| **Local** | Qwen, OSS, Qwen3.6-27B alone, gpt-oss-120b alone | vLLM on your GPUs | 102 | 23,850 |
| **Cross-family** | Qwen3.5-9B → GPT-5.4, gpt-oss-20b → GPT-5.4, Nano → Qwen3.6-27B | both | 12 | 2,790 |

The short version:

```bash
# 1. install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. credentials
cp .env.example .env              # add OPENAI_API_KEY
huggingface-cli login             # after requesting access to gaia-benchmark/GAIA

# 3. check the setup (no model calls)
python scripts/run_experiment_plan.py --all --dry-run

# 4. run everything (start the vLLM servers first, see step 4 below)
python scripts/run_experiment_plan.py --all

# 5. rebuild the paper tables from the logs
python scripts/paper_tables.py    # -> results/paper_tables/*.csv
```

The rest of this section explains each step.

### Step 0: what you need

| Needed for | Requirement |
|---|---|
| Everything | Linux or macOS (Windows works for the API group; vLLM needs Linux), Python ≥ 3.10, internet access. GAIA tools use web search, page fetch, and Wikipedia; HotpotQA uses the Wikipedia API. |
| GAIA | A Hugging Face account with access to the gated dataset [`gaia-benchmark/GAIA`](https://huggingface.co/datasets/gaia-benchmark/GAIA) |
| HotpotQA | Nothing: `hotpot_qa` / `fullwiki` is public and downloads automatically |
| API group and cross-family group | An OpenAI API key with access to `gpt-5.4`, `gpt-5.4-nano`, `gpt-4.1`, and `gpt-4.1-mini` |
| Local group and cross-family group | NVIDIA GPUs with [vLLM](https://github.com/vllm-project/vllm). The paper used RTX 6000 Ada (48 GB): one GPU each for Qwen3.5-9B and gpt-oss-20b, and two each (tensor parallel) for Qwen3.6-27B and gpt-oss-120b. |

### Step 1: install

```bash
git clone https://anonymous.4open.science/r/advisor-eval   # or download and unzip the archive
cd advisor-eval
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For the local group, install vLLM on the GPU machine. Using its own
environment avoids PyTorch version conflicts:

```bash
python -m venv .venv-vllm && source .venv-vllm/bin/activate
pip install vllm
```

### Step 2: credentials and data access

```bash
cp .env.example .env
```

Then edit `.env`:

- `OPENAI_API_KEY=sk-...`: required for the API and cross-family groups.
- `MAX_TOTAL_USD=150`: each run session stops before its OpenAI spend would
  exceed this cap. Raise it for a full run.

`.env` is gitignored. Never commit it.

For GAIA, request access on the dataset page, then log in once:

```bash
huggingface-cli login
```

Both datasets download on first use. The 300-question HotpotQA sample (150
bridge and 150 comparison questions, seed 42) is written to
`data/hotpotqa_fullwiki_manifest_seed42.json`.

### Step 3: check the setup

This step makes no model calls and costs nothing:

```bash
python scripts/run_experiment_plan.py --all --dry-run   # loads every condition, its tasks and its prompts
python scripts/check_prompt_equivalence.py              # the DNA prompt files match the advisor code
```

Then send one test request to each model you will use:

```bash
python scripts/verify_models.py --openai-only     # API models
python scripts/verify_models.py --local-only      # vLLM models (after step 4)
```

### Step 4: start the vLLM servers (local and cross-family groups only)

Each local model runs as an OpenAI-compatible server on a fixed port. The ports
are set in `config.yaml` under `local_models`. Run each command in its own
terminal, or under `tmux` or `nohup`, on the GPU machine:

```bash
CUDA_VISIBLE_DEVICES=0     bash scripts/serve_vllm.sh qwen3.5-9b     # :8001
CUDA_VISIBLE_DEVICES=1,2   bash scripts/serve_vllm.sh qwen3.6-27b    # :8002 (TP=2)
CUDA_VISIBLE_DEVICES=3     bash scripts/serve_vllm.sh gpt-oss-20b    # :8003
CUDA_VISIBLE_DEVICES=4,5   bash scripts/serve_vllm.sh gpt-oss-120b   # :8004 (TP=2)
```

- You only need the servers for the pair you are running. For example, the
  Qwen pair needs the first two.
- `TP` overrides the tensor-parallel size; by default it is the number of
  visible GPUs. `MAX_MODEL_LEN` sets the context length (default 32768).
- **Servers on another machine.** If the servers run on a different machine
  from the experiment runner, point the runner at them in `.env`, for example
  `QWEN_EXECUTOR_URL=http://<gpu-host>:8001/v1`. The other variables are
  `QWEN_ADVISOR_URL`, `OSS_EXECUTOR_URL` and `OSS_ADVISOR_URL`.
- The decoding settings sent to each server are in `config.yaml`: thinking is
  disabled for Qwen, and gpt-oss runs with `reasoning_effort: low`.

Wait until each server logs that it is ready, then run
`python scripts/verify_models.py --local-only`.

### Step 5: run the experiments

Start with a smoke test of a few tasks:

```bash
python run_condition.py --condition gaia_nano_hybrid --limit 5
python run_condition.py --condition hotpot_qwen_self --limit 5    # needs the Qwen servers
```

Then run one group at a time, or everything at once. A condition with no
`--limit` runs all 165 GAIA tasks or all 300 HotpotQA questions.

```bash
# API group (OpenAI only)
python scripts/run_experiment_plan.py --only "*_nano_*" --only "*_mini_*" --only "*_gpt41_*" --only "*_gpt54_alone"

# Local group (vLLM only, no OpenAI cost)
python scripts/run_experiment_plan.py --only "*_qwen_*" --only "*_qwen27b_alone"   # Qwen servers
python scripts/run_experiment_plan.py --only "*_oss_*"  --only "*_oss120b_alone"   # gpt-oss servers

# Cross-family group (vLLM + OpenAI)
python scripts/run_experiment_plan.py --only "*-x-*"

# Or everything
python scripts/run_experiment_plan.py --all
```

To reproduce a single paper table, select it by key from
`configs/paper_tables.yaml`, optionally narrowed with `--only`:

| Paper table | Command |
|---|---|
| Table 1 (and Tables 9, 10, 11): initiation policies | `python scripts/run_experiment_plan.py --table table1_initiation` |
| Tables 5, 6: Report thresholds τ = 0.25 / 0.5 | `python scripts/run_experiment_plan.py --table table5_6_report_thresholds` |
| Table 7: seeds 42 / 43 / 44 | `python scripts/run_experiment_plan.py --table table7_seeds` |
| Table 2: speaker and channel | `python scripts/run_experiment_plan.py --table table2_channels` |
| Tables 12, 13: DNA-act ablation | `python scripts/run_experiment_plan.py --table table12_ablation` |
| Tables 3, 8: advisor swaps | `python scripts/run_experiment_plan.py --table table3_8_cross_family` |
| Only the Nano part of Table 1 | `python scripts/run_experiment_plan.py --table table1_initiation --only "*_nano_*"` |

Notes on running:

- **Runs resume.** Finished tasks and finished conditions are skipped, so you
  can stop a run at any time (Ctrl-C) and rerun the same command.
- **Spend cap.** Each session stops before its OpenAI spend would exceed
  `MAX_TOTAL_USD`. Rerun the command with a higher cap to continue.
- **Long runs.** Run inside `tmux` or with `nohup ... &`. The groups are
  independent, so they can run in parallel in separate terminals.
- **One condition.** `python run_condition.py --condition <condition_id>` runs
  a single condition.

### Step 6: rebuild the tables

```bash
python scripts/paper_tables.py        # -> results/paper_tables/<table>.csv
```

This writes one CSV per paper table, with one row per condition:

- accuracy, and accuracy on easy (GAIA Level 1) and hard (Levels 2–3) tasks;
- help and harm against the matching executor-alone run, with an exact
  McNemar p-value;
- advisor messages, and tokens per message (`tiktoken` `o200k_base`);
- cost per task, and the cost reduction relative to GPT-5.4 alone;
- latency, tool errors, and recovery efficiency (accuracy points gained per
  advisor message).

The script also lists any condition in a table that has no run yet. To compare
any two runs task by task:

```bash
python scripts/help_harm.py --cheap runs/gaia_nano_cheap/tasks.jsonl --advised runs/gaia_nano_hybrid/tasks.jsonl
```

With a fixed seed, the OpenAI models can still vary slightly from run to run.
The paper's three-seed comparison measures this: accuracy moves by at most 2
points between seeds.

## Experimental setup

These settings follow Section 4 and Appendix A of the paper, and are set in
`config.yaml` and the condition files.

- **Benchmarks.**
  - **GAIA**: the full validation set, 165 tasks (53 Level 1, 86 Level 2,
    26 Level 3), with web search, page fetch, Wikipedia, calculator, Python,
    and attachment tools.
  - **HotpotQA**: 300 fullwiki questions (150 bridge, 150 comparison),
    stratified with seed 42. There is one `wiki_search` tool, which returns
    the opening sections of the top three articles. The sample is written to
    `data/hotpotqa_fullwiki_manifest_seed42.json` on the first run.
- **Budgets.**
  - At most 16 steps, 12 tool calls, and B = 2 advisor messages per task.
  - If the step limit is reached, one forced-answer prompt makes the executor
    commit to an answer.
- **Decoding.**

  | Models | Setting |
  |---|---|
  | GPT-4.1 models | temperature 0 |
  | GPT-5.4 models | `reasoning_effort: low` |
  | Qwen models | thinking disabled |
  | gpt-oss models | `reasoning_effort: low` |

  Seed 42 is used everywhere; seeds 43 and 44 are added for the main
  comparison (Nano, Qwen, OSS × no channel / hybrid / self-initiated).
- **Initiation policies** (`policies.py`):
  - **Monitor**: fires on a tool exception, malformed output, or a repeated
    identical call.
  - **Report**: fires when the executor's self-reported confidence falls
    below τ ∈ {0.25, 0.5, 0.75}.
  - **Hybrid**: Monitor OR Report, with τ = 0.75.
  - **Self-initiated**: the executor emits a help request (`[REQUEST_ADVISOR]`)
    instead of an action.
  - **Random**: fires with p = 0.3 per step, under the same budget.
- **Advisor context.** When the channel opens, the advisor receives the task,
  the trajectory so far, any error signals, the current candidate answer, and
  the executor's request, if it made one.
- **Scoring.**
  - Accuracy is normalized exact match. GAIA uses its official answer
    normalization; HotpotQA uses SQuAD-style normalization.
  - Setting `evaluation.llm_judge: true` adds an optional `gpt-4.1-nano`
    equivalence fallback. It is off by default, and the paper does not use it.
- **Prices** (Appendix Table 4, OpenAI list prices, per 1M tokens, input /
  output):

  | Model | Input | Output |
  |---|---|---|
  | GPT-5.4 Nano | $0.20 | $1.25 |
  | GPT-4.1 Mini | $0.40 | $1.60 |
  | GPT-4.1 | $2.00 | $8.00 |
  | GPT-5.4 | $2.50 | $15.00 |

  Locally served models are costed at $0. The paper reports their GPU time
  instead: 0.45 GPU-hours per 100 tasks for the Qwen pair and 0.65 for the
  gpt-oss pair.

### Condition names

Condition ids follow `<benchmark>_<pair>_<variant>[_s<seed>]`. The benchmark
is `gaia` or `hotpot`, and the pair is one of `nano`, `mini`, `gpt41`, `qwen`,
`oss`, `qwen-x-gpt54`, `oss-x-gpt54`, or `nano-x-qwen27b`.

| Variant | Paper configuration |
|---|---|
| `cheap` | No channel (executor alone) |
| `random`, `monitor`, `hybrid`, `self` | Random, Monitor, Hybrid, Self-initiated |
| `report_t0.25`, `report_t0.5`, `report_t0.75` | Report with threshold τ |
| `hybrid_selfdna` | Self-DNA: the executor writes the DNA message to itself |
| `hybrid_<format>` | Hybrid initiation with another message format (below) |
| `gpt54_alone`, `qwen27b_alone`, `oss120b_alone` | Advisor alone (e.g. `gaia_gpt54_alone`) |

Examples: `gaia_nano_hybrid`, `hotpot_qwen_self_s43`,
`gaia_oss_hybrid_dna_compact_dn`, `hotpot_qwen-x-gpt54_self`.

### Message formats (`prompts/`)

| `message_format` | Paper name | Prompt |
|---|---|---|
| `dna_compact` (default) | DNA, compact: three short labelled fields (about 70 tokens) | `dna_compact.txt` |
| `dna_sentences` | DNA, full sentences | `dna_sentences.txt` |
| `dna_compact_dn`, `_an`, `_da`, `_n`, `_d`, `_a` | Ablation subsets (D = DIAGNOSE, A = AVOID, N = NEXT) | `dna_compact_*.txt` |
| `freeform` | Free-form, no labelled acts | `freeform.txt` |
| `freeform_lenmatched` | Free-form, limited to the length of full-sentence DNA (172 tokens) | `freeform.txt` + length limit |
| `action_handoff` | Action hand-off: the advisor sends the exact next tool call, which is **executed** | `action_handoff.txt` |
| `answer_hint` | Answer hint: the advisor states the answer it believes is correct | `answer_hint.txt` |

The ablation prompts are derived from `dna_compact.txt` by
`scripts/generate_ablation_prompts.py`. `scripts/write_condition_yamls.py`
regenerates all condition files and `configs/paper_tables.yaml`.

## Outputs

Each run writes the following files to `runs/<condition_id>/`:

| File | Contents |
|---|---|
| `meta.json` | The resolved condition, models, budgets, decoding settings, seed, and total cost |
| `tasks.jsonl` | One row per task: prediction, gold answer, `correct`, cost, latency, tool calls and errors, number of advisor messages, GAIA level or HotpotQA type |
| `messages.jsonl` | One row per advisor message: trigger, speaker, format, text, parsed DNA fields, whether a hand-off was executed |
| `trace.jsonl` | The step-by-step trajectory: executor actions, tool observations, advisor events |

The `results/*.jsonl` files and the `runs/e1_*`, `runs/e2_*` and `runs/repro_*`
directories are earlier development runs. They used smaller task subsets and
older settings, and they are not used for the tables.

## Repository layout

```
config.yaml                 budgets, datasets, decoding, vLLM endpoints, prices
run_condition.py            run one condition -> runs/<condition_id>/
evaluator.py                executor-alone / advisor-alone / advised loops and scoring
gaia_runner.py              GAIA agent loop and tools
hotpotqa_runner.py          HotpotQA agent loop (wiki_search)
executor.py, advisor.py     executor and advisor agents
policies.py                 initiation policies
datasets_loader.py          GAIA and HotpotQA loading and stratified sampling
logger.py                   cost and latency accounting
experiment/                 condition specs, message-format hooks, vLLM routing, logging
configs/conditions/         one YAML per condition (generated)
configs/paper_tables.yaml   conditions behind each paper table (generated)
prompts/                    advisor message prompts
scripts/                    table runner, table builder, statistics, model checks, vLLM launcher
runs/                       per-condition trajectory logs
```

## Notes

- **Model versions.** The OpenAI results depend on specific API model versions.
  With a fixed seed and temperature 0, API outputs can still vary slightly
  between runs. The paper measures this variation with three seeds.
- **Advanced option.** Condition files accept `loop_guards: true`, which ORs
  extra stuck-detection triggers into any policy (for example a hedged final
  answer or a nearly exhausted tool budget). It is **off** in every paper
  configuration.

## License

Code is released under the MIT License (see `LICENSE`). HotpotQA is
released under CC BY-SA 4.0. GAIA is gated and distributed under the
terms of its Hugging Face dataset card; it is not redistributed here.
Model outputs are subject to the terms of the models that produced them.
