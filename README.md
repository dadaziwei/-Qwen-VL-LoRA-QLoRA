# Edge Qwen QLoRA + AWQ + FastAPI

围绕 Qwen 系列模型整理的一套本地量化、轻量微调与流式 API 服务方案，重点放在普通开发机上的可复现工作流，以及后续迁移到远程 GPU 的一致接口设计。

这个仓库按一台 8GB 显卡的本地开发机来组织：本地侧负责 AWQ 推理、FastAPI/SSE 服务和 QLoRA 训练闭环，远程侧负责更完整的多模态训练和 vLLM 推理服务。这样拆开以后，开发、调试和部署的边界会清楚很多。

## 项目目标

这个项目主要解决四件事：

- 用 `AutoAWQ` 把模型压到更适合本地部署的形态。
- 用 `PEFT + QLoRA` 完成资源友好的微调流程。
- 用 `FastAPI + SSE` 封装统一的模型服务接口。
- 用 `vLLM` 承接远程 GPU 上的高效推理服务。

对应到工程里，仓库已经把这些链路拆成独立模块，方便单独调试，也方便组合部署。

## 当前实现

- 本地 `Transformers` 后端，支持 Qwen AWQ 模型推理。
- `/health` 健康检查接口。
- `/v1/chat/completions` 接口，兼容 OpenAI 风格请求格式。
- `stream=true` 的 SSE 流式输出。
- 基于 `trl + peft + bitsandbytes` 的 QLoRA SFT 训练脚本。
- 基于 `AutoAWQ` 的 INT4 权重量化脚本。
- 基于 `vLLM` 的远程 OpenAI 兼容推理接入。

## 适合的开发方式

这套项目结构比较适合下面这种节奏：

1. 在本地把模型调用、数据格式、SSE 返回和训练脚本调顺。
2. 保持 FastAPI 接口不变，把更重的推理和训练迁移到远程 GPU。
3. 后续再叠加前端、监控、压测和多 Adapter 管理。

这种方式的好处是，本地验证和远程部署不会互相打架，API 也始终保持一层稳定边界。

## 快速开始

### 1. 安装环境

Windows 本地开发：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Linux 远程 GPU：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt -r requirements-vllm.txt
pip install -e .
```

### 2. 启动本地 API

```powershell
$env:EDGE_QWEN_BACKEND="transformers"
$env:EDGE_QWEN_MODEL="Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
uvicorn edge_qwen.api:app --host 0.0.0.0 --port 8000
```

也可以直接使用：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_api.ps1
```

### 3. 验证流式请求

```powershell
curl.exe -N http://127.0.0.1:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d "{\"model\":\"edge-qwen\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"用三句话介绍边缘端模型量化。\"}]}"
```

### 4. 跑通 QLoRA 训练

```powershell
python scripts/train_qlora_sft.py `
  --config configs/local_8gb.yaml `
  --dataset data/sample_sft.jsonl `
  --output-dir outputs/qwen-lora-local
```

### 5. 量化自有模型

```bash
python scripts/quantize_awq.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir outputs/qwen2_5_1_5b_awq \
  --calib data/sample_sft.jsonl
```

## vLLM 接入

远程 Linux GPU 上先启动 vLLM：

```bash
export EDGE_QWEN_MODEL=Qwen/Qwen2.5-VL-3B-Instruct-AWQ
bash scripts/serve_vllm.sh
```

然后启动 FastAPI 代理：

```bash
export EDGE_QWEN_BACKEND=openai
export VLLM_BASE_URL=http://127.0.0.1:8001/v1
uvicorn edge_qwen.api:app --host 0.0.0.0 --port 8000
```

这样业务侧始终只需要访问 `/v1/chat/completions`，底层后端可以在本地 Transformers 和远程 vLLM 之间切换。

## 项目结构

```text
configs/              训练和服务配置
data/                 示例 SFT 数据
docs/                 远程 GPU 部署补充说明
scripts/              训练、量化、服务启动脚本
src/edge_qwen/        FastAPI 服务和推理后端
```

## 后续可继续扩展

- 增加多模态训练数据处理脚本。
- 增加前端聊天页，直接接入图片上传和流式展示。
- 增加多 LoRA Adapter 切换能力。
- 增加压测脚本，对比 Transformers 和 vLLM 的吞吐。
- 增加 Dockerfile、compose 和基础监控。

## 参考资料

- Qwen2.5-VL AWQ 模型卡：https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-AWQ
- vLLM OpenAI 兼容服务：https://docs.vllm.ai/en/latest/serving/openai_compatible_server/
- vLLM AWQ 量化说明：https://docs.vllm.ai/en/stable/features/quantization/auto_awq/
- PEFT 量化训练说明：https://huggingface.co/docs/peft/developer_guides/quantization
- Transformers bitsandbytes/QLoRA 说明：https://huggingface.co/docs/transformers/en/quantization/bitsandbytes
