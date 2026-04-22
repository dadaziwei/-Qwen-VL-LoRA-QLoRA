# 05 vLLM + FastAPI 部署

本章目标：理解为什么要引入 vLLM，并把 vLLM 和本项目 FastAPI 服务组合起来。

## 1. 架构说明

推荐部署架构：

```text
客户端
  |
  | /v1/chat/completions
  v
FastAPI 服务
  |
  | OpenAI 兼容请求
  v
vLLM 推理服务
  |
  v
Qwen AWQ 模型
```

FastAPI 负责：

- 统一接口格式。
- 鉴权、日志、限流等业务逻辑扩展。
- SSE 流式输出。
- 屏蔽底层推理后端差异。

vLLM 负责：

- KV cache 管理。
- 并发请求调度。
- 降低变长文本生成造成的显存碎片影响。
- 提供 OpenAI 兼容服务。

## 2. 启动 vLLM

在 Linux GPU 机器上执行：

```bash
source .venv/bin/activate
export EDGE_QWEN_MODEL=Qwen/Qwen2.5-VL-3B-Instruct-AWQ
bash scripts/serve_vllm.sh
```

脚本内部会执行：

```bash
vllm serve "$MODEL" \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.88 \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --port 8001
```

## 3. 启动 FastAPI 代理

另开一个终端：

```bash
source .venv/bin/activate
export EDGE_QWEN_BACKEND=openai
export VLLM_BASE_URL=http://127.0.0.1:8001/v1
uvicorn edge_qwen.api:app --host 0.0.0.0 --port 8000
```

此时业务方只需要访问 `8000` 端口，不需要直接感知 vLLM。

## 4. 测试请求

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "edge-qwen",
    "stream": true,
    "messages": [
      {
        "role": "user",
        "content": "解释 vLLM 在边缘端服务中的作用。"
      }
    ]
  }'
```

## 5. 参数调优建议

8GB 或 12GB 显存：

- `--max-model-len 2048`
- `--gpu-memory-utilization 0.80`
- 控制并发数量。

24GB 显存：

- `--max-model-len 4096` 或更高。
- `--gpu-memory-utilization 0.88`
- 可以做基本并发压测。

如果出现 OOM，优先降低 `max-model-len`，其次降低并发。

