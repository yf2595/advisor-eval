#!/usr/bin/env bash
# Serve one open-weight model with vLLM's OpenAI-compatible server.
# Ports match `local_models` in config.yaml.
#
# Usage:  bash scripts/serve_vllm.sh <qwen3.5-9b|qwen3.6-27b|gpt-oss-20b|gpt-oss-120b>
# Env:    CUDA_VISIBLE_DEVICES  GPUs to use (e.g. "0" or "0,1")
#         TP                    tensor-parallel size (default: number of visible GPUs)
#         MAX_MODEL_LEN         context length (default 32768)
#
# The paper used NVIDIA RTX 6000 Ada (48 GB) GPUs: one GPU for the 9B / 20B
# executors, two (TP=2) for the 27B / 120B advisors.
set -euo pipefail

MODEL_KEY="${1:?model key required}"
case "$MODEL_KEY" in
  qwen3.5-9b)   HF_ID="Qwen/Qwen3.5-9B";     PORT=8001 ;;
  qwen3.6-27b)  HF_ID="Qwen/Qwen3.6-27B";    PORT=8002 ;;
  gpt-oss-20b)  HF_ID="openai/gpt-oss-20b";  PORT=8003 ;;
  gpt-oss-120b) HF_ID="openai/gpt-oss-120b"; PORT=8004 ;;
  *) echo "unknown model key: $MODEL_KEY" >&2; exit 1 ;;
esac

N_GPUS=$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES:-0}")
TP="${TP:-$N_GPUS}"

exec vllm serve "$HF_ID" \
  --served-model-name "$HF_ID" \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --seed 42
