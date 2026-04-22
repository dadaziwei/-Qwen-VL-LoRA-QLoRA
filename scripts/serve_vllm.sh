#!/usr/bin/env bash
set -euo pipefail

MODEL="${EDGE_QWEN_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct-AWQ}"
PORT="${VLLM_PORT:-8001}"

# 远程 Linux GPU 上启动 vLLM OpenAI 兼容服务，供 FastAPI 代理调用。
vllm serve "$MODEL" \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.88}" \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --host 0.0.0.0 \
  --port "$PORT"
