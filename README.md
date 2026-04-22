# Edge Qwen QLoRA + AWQ + FastAPI

这是一个围绕 Qwen 系列模型做“本地量化、轻量微调、流式 API 服务”的学习型工程项目。

我选择把 8GB 显卡作为主要开发基准，不是因为它最强，而是因为它最常见：很多学生、个人开发者和刚开始接触大模型部署的人，手里就是一张 3060/4060/4060 Laptop 这类 8GB 显卡。如果一个项目只能在 A100 上跑，它当然很漂亮，但学习价值会离普通开发者远一些。这个项目更关注：在多数人能接触到的硬件上，怎样把大模型部署链路真正跑通。

## 我想解决的问题

大模型项目很容易停在“看起来能跑”的阶段：下载模型、写几行推理代码，然后就结束了。但真正做一个可讲、可改、可迁移的项目，至少要回答几个问题：

- 模型太大时，如何通过 AWQ 量化把推理成本降下来？
- 训练资源有限时，如何用 PEFT + QLoRA 只训练 LoRA Adapter？
- 本地开发和远程 GPU 部署之间，接口如何保持一致？
- 模型输出很慢时，如何用 SSE 做流式返回？
- 如果后面接入业务系统，怎样把推理后端藏在统一 API 后面？

这个仓库就是围绕这些问题搭出来的。

## 技术路线

- `AutoAWQ`：用于 INT4 权重量化，降低模型部署门槛。
- `PEFT + QLoRA`：冻结基座模型，只训练少量 Adapter 参数。
- `Transformers`：负责本地开发和小规模推理验证。
- `vLLM`：负责远程 GPU 上的高效推理服务。
- `FastAPI + SSE`：封装统一接口，并支持流式输出。

我把项目分成两条线：

| 阶段 | 目标 | 推荐模型 | 运行位置 |
| --- | --- | --- | --- |
| 本地学习线 | 跑通 AWQ 推理、SSE API、QLoRA 训练闭环 | `Qwen/Qwen2.5-VL-3B-Instruct-AWQ`、`Qwen/Qwen2.5-0.5B-Instruct` | 8GB 显卡 |
| 远程工程线 | 完整多模态 LoRA/QLoRA、vLLM 并发服务 | `Qwen/Qwen2.5-VL-3B-Instruct` | A10G/4090/A100 |

这也是我认为比较舒服的学习路径：先在本地把链路摸清楚，再把重计算部分迁移到租用算力。

## 快速开始

完整分步骤教程见：[tutorials/README.md](./tutorials/README.md)。教程不是简单命令堆砌，我把为什么这样选、哪些地方容易踩坑、以及后续怎么扩展都写进去了。

### 1. 创建环境

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

### 2. 本地启动 FastAPI 服务

本地开发默认使用 Transformers 后端：

```powershell
$env:EDGE_QWEN_BACKEND="transformers"
$env:EDGE_QWEN_MODEL="Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
uvicorn edge_qwen.api:app --host 0.0.0.0 --port 8000
```

也可以直接执行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_api.ps1
```

测试流式接口：

```powershell
curl.exe -N http://127.0.0.1:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d "{\"model\":\"edge-qwen\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"用三句话介绍边缘端模型量化。\"}]}"
```

### 3. 跑通 QLoRA 训练闭环

本地训练我选择小 Qwen 文本模型，因为这个阶段的重点不是追求最终效果，而是理解 QLoRA 的完整流程：4bit 加载、冻结基座、训练 Adapter、保存 LoRA。

```powershell
python scripts/train_qlora_sft.py `
  --config configs/local_8gb.yaml `
  --dataset data/sample_sft.jsonl `
  --output-dir outputs/qwen-lora-local
```

### 4. 量化自有模型

如果要量化自己的文本模型，可以执行：

```bash
python scripts/quantize_awq.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir outputs/qwen2_5_1_5b_awq \
  --calib data/sample_sft.jsonl
```

我的经验是：学习阶段可以直接使用官方 AWQ 模型，把时间花在服务封装和训练流程上；等链路跑通以后，再单独研究自定义量化。

## vLLM 服务部署

远程 Linux GPU 上启动 vLLM：

```bash
export EDGE_QWEN_MODEL=Qwen/Qwen2.5-VL-3B-Instruct-AWQ
bash scripts/serve_vllm.sh
```

再启动 FastAPI 代理：

```bash
export EDGE_QWEN_BACKEND=openai
export VLLM_BASE_URL=http://127.0.0.1:8001/v1
uvicorn edge_qwen.api:app --host 0.0.0.0 --port 8000
```

这样客户端始终访问本项目的 `/v1/chat/completions`，底层可以在本地 Transformers 和远程 vLLM 之间切换。这个抽象很重要，因为真实项目里经常会经历“先本地验证，再上远程 GPU，再接业务系统”的过程。

## 项目结构

```text
configs/              训练和服务配置
data/                 示例 SFT 数据
docs/                 远程 GPU 部署补充说明
scripts/              训练、量化、服务启动脚本
src/edge_qwen/        FastAPI 服务和推理后端
tutorials/            分章节中文教程
```

## 我的开发取舍

这个项目刻意没有一上来追求“最大模型、最高分数、最复杂训练”。我更希望它像一个真实开发者会做的学习项目：

- 先把最小可运行链路搭出来。
- 用 8GB 显卡作为多数人都能复现的基准。
- 把本地开发和远程部署分开设计。
- 把接口设计成稳定形态，后续换模型、换推理框架都不影响调用方。
- 把复杂问题拆成量化、微调、推理、服务四个模块逐个理解。

后续可以继续扩展：

- 增加多模态 LoRA 数据处理脚本。
- 增加简单 Web 前端，用于上传图片并流式显示回答。
- 增加压测脚本，对比 Transformers 和 vLLM 的吞吐差异。
- 增加 Dockerfile 和 systemd 服务文件。

## 参考资料

- Qwen2.5-VL AWQ 模型卡：https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-AWQ
- vLLM OpenAI 兼容服务：https://docs.vllm.ai/en/latest/serving/openai_compatible_server/
- vLLM AWQ 量化说明：https://docs.vllm.ai/en/stable/features/quantization/auto_awq/
- PEFT 量化训练说明：https://huggingface.co/docs/peft/developer_guides/quantization
- Transformers bitsandbytes/QLoRA 说明：https://huggingface.co/docs/transformers/en/quantization/bitsandbytes
