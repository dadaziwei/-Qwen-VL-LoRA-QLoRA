# Remote GPU SSH Runbook

本地 8GB 显存适合演示推理和小模型 QLoRA。完整的 Qwen-VL LoRA 微调建议放到远程 GPU。

## 选型

| 任务 | 最低建议 | 更稳选择 |
| --- | --- | --- |
| AWQ 推理服务 | T4 16GB | A10G 24GB |
| Qwen2.5-VL-3B LoRA | RTX 4090 24GB | A10G 24GB |
| Qwen2.5-VL-7B LoRA | A100 40GB | A100 80GB |

## SSH 初始化

```bash
ssh ubuntu@<remote-ip>
sudo apt update
sudo apt install -y git python3-venv
git clone <your-repo-url> edge-qwen
cd edge-qwen
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt -r requirements-vllm.txt
```

## 启动 vLLM

```bash
export EDGE_QWEN_MODEL=Qwen/Qwen2.5-VL-3B-Instruct-AWQ
bash scripts/serve_vllm.sh
```

## 启动 API 代理

另开一个 SSH 终端：

```bash
cd edge-qwen
source .venv/bin/activate
export EDGE_QWEN_BACKEND=openai
export VLLM_BASE_URL=http://127.0.0.1:8001/v1
uvicorn edge_qwen.api:app --host 0.0.0.0 --port 8000
```

## 本地端口转发

```bash
ssh -L 8000:127.0.0.1:8000 ubuntu@<remote-ip>
```

本地访问：

```bash
curl http://127.0.0.1:8000/health
```
