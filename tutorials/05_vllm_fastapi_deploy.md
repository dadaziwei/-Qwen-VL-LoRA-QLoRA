# 05 vLLM + FastAPI 部署

这一章是我认为最像真实工程的一部分：不要让业务代码直接绑死在某个推理框架上。FastAPI 做稳定入口，vLLM 做底层推理引擎，这样以后换模型、换机器、换部署方式都更轻松。

## 1. 架构

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

- 对外提供稳定 API。
- 统一请求和响应格式。
- 处理 SSE 流式返回。
- 后续扩展鉴权、日志、限流。

vLLM 负责：

- 高效管理 KV cache。
- 调度并发请求。
- 处理变长文本生成。
- 提供 OpenAI 兼容服务。

我喜欢这个分层，因为它把“业务接口”和“推理性能”拆开了。

## 2. 启动 vLLM

在 Linux GPU 机器上：

```bash
source .venv/bin/activate
export EDGE_QWEN_MODEL=Qwen/Qwen2.5-VL-3B-Instruct-AWQ
bash scripts/serve_vllm.sh
```

脚本内部使用：

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

此时客户端访问的是 `8000`，而不是直接访问 vLLM 的 `8001`。

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

## 5. 参数怎么想

本地开发时，我倾向于保守参数，先保证服务稳定。远程 GPU 上再逐步放开：

- `--max-model-len`：上下文越长，服务越吃资源。先从 2048 或 4096 开始。
- `--gpu-memory-utilization`：不要一开始拉满，给系统和框架留一点空间。
- 并发数：先小并发验证，再做压测。

我的经验是，服务稳定比一次跑到极限更重要。项目初期要先能复现、能调试、能解释。

