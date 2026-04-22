# Edge Qwen QLoRA + AWQ + FastAPI

面向 8GB 显存边缘端的 Qwen 本地量化、轻量微调与 API 服务部署示例。

本项目把简历中的工作拆成可运行闭环：

- AutoAWQ：把 Qwen 系列模型做 INT4 权重量化，或直接使用官方/社区 AWQ 模型。
- PEFT + QLoRA：冻结基座模型，只训练 LoRA Adapter，降低显存和训练成本。
- vLLM：在 Linux GPU 环境中提供 OpenAI 兼容推理服务，改善 KV cache 管理和并发吞吐。
- FastAPI + SSE：封装统一 `/v1/chat/completions` 接口，支持流式返回，便于前后端集成。

## 硬件策略

你的本地显存只有 8GB，因此不要直接微调 FP16 的多模态 3B/7B 模型。

推荐分三档完成项目：

| 阶段 | 目标 | 推荐模型 | 运行位置 |
| --- | --- | --- | --- |
| 本地演示 | 多模态 AWQ 推理 + SSE API | `Qwen/Qwen2.5-VL-3B-Instruct-AWQ` | 本地 8GB GPU |
| 本地训练闭环 | QLoRA 文本微调流程 | `Qwen/Qwen2.5-0.5B-Instruct` 或 `Qwen/Qwen2.5-1.5B-Instruct` | 本地 8GB GPU |
| 完整项目 | Qwen-VL LoRA/QLoRA 微调 + vLLM 并发服务 | `Qwen/Qwen2.5-VL-3B-Instruct` | 租用 A10G/RTX 4090/A100 |

8GB 显存下的关键参数：

- `max_seq_length`: 512 或 768
- `batch_size`: 1
- `gradient_accumulation_steps`: 8 到 16
- LoRA rank `r`: 8 或 16
- 优先使用 4-bit NF4 训练小模型；多模态训练建议远程 GPU

## 快速开始

完整分步骤教程见：[tutorials/README.md](./tutorials/README.md)。建议按“环境准备 -> 本地推理 -> QLoRA 微调 -> AWQ 量化 -> vLLM 部署 -> 远程 GPU”的顺序执行。

### 1. 创建环境

Windows 本地推理/训练：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

Linux 远程 vLLM 服务：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -r requirements-vllm.txt
```

### 2. 本地启动 FastAPI 服务

Transformers 后端适合 Windows/本地 8GB 显存演示：

```powershell
$env:EDGE_QWEN_BACKEND="transformers"
$env:EDGE_QWEN_MODEL="Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
uvicorn edge_qwen.api:app --host 0.0.0.0 --port 8000
```

请求流式接口：

```powershell
Invoke-RestMethod http://localhost:8000/v1/chat/completions `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"model":"edge-qwen","stream":true,"messages":[{"role":"user","content":"用三句话介绍边缘端模型量化。"}]}'
```

### 3. 本地跑 QLoRA 微调闭环

```powershell
python scripts/train_qlora_sft.py `
  --config configs/local_8gb.yaml `
  --dataset data/sample_sft.jsonl `
  --output-dir outputs/qwen-lora-local
```

训练完成后只会保存 LoRA Adapter，不会复制完整基座模型。

### 4. 量化自有模型

8GB 显存不适合从 FP16 多模态模型现场量化大模型。建议在远程 GPU 上执行：

```bash
python scripts/quantize_awq.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir outputs/qwen2_5_1_5b_awq \
  --calib data/sample_sft.jsonl
```

多模态模型优先使用现成 AWQ 权重：

```text
Qwen/Qwen2.5-VL-3B-Instruct-AWQ
```

## vLLM 远程服务

在 Linux GPU 机器上启动 vLLM OpenAI 兼容服务：

```bash
vllm serve Qwen/Qwen2.5-VL-3B-Instruct-AWQ \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.88 \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --port 8001
```

然后启动本项目 FastAPI 代理：

```bash
export EDGE_QWEN_BACKEND=openai
export VLLM_BASE_URL=http://127.0.0.1:8001/v1
uvicorn edge_qwen.api:app --host 0.0.0.0 --port 8000
```

这样业务侧只调用本项目的 `/v1/chat/completions`，底层可以在 Transformers 和 vLLM 间切换。

## SSH 租用算力流程

```bash
ssh ubuntu@<remote-ip>
sudo apt update
sudo apt install -y python3-venv git
git clone <your-repo-url> edge-qwen
cd edge-qwen
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt -r requirements-vllm.txt
```

建议租用配置：

- 只做 API 推理：RTX 3060 12GB / T4 16GB / A10G 24GB
- 做 Qwen2.5-VL-3B LoRA 微调：RTX 4090 24GB / A10G 24GB
- 做 7B 多模态训练：A100 40GB 起步

## 项目结构

```text
configs/              训练和服务配置
data/                 示例 SFT 数据
scripts/              量化、训练、部署辅助脚本
src/edge_qwen/        FastAPI 服务和推理后端
```

## 常见限制

- Windows 上 vLLM 支持有限，推荐把 vLLM 放到 Linux 远程 GPU。
- 8GB 显存下，多模态 QLoRA 训练很容易 OOM；本地只建议跑小模型文本 SFT 闭环。
- AutoAWQ 已经能完成项目展示，但新项目也可以评估 GPTQModel、bitsandbytes 或厂商推理栈。

## 参考资料

- Qwen2.5-VL AWQ 模型卡：https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-AWQ
- vLLM OpenAI 兼容服务：https://docs.vllm.ai/en/latest/serving/openai_compatible_server/
- vLLM AWQ 量化说明：https://docs.vllm.ai/en/stable/features/quantization/auto_awq/
- PEFT 量化训练说明：https://huggingface.co/docs/peft/developer_guides/quantization
- Transformers bitsandbytes/QLoRA 说明：https://huggingface.co/docs/transformers/en/quantization/bitsandbytes
