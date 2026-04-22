# Remote GPU SSH Runbook

这份文档是远程 GPU 的快速操作手册。我的思路是：本地 8GB 开发机负责把代码、接口和小实验跑顺；远程 GPU 负责更完整的多模态训练和 vLLM 服务。

## 什么时候上远程 GPU

我一般会等本地完成这些事再租机器：

- `/health` 接口能正常返回。
- `/v1/chat/completions` 能完成非流式和流式请求。
- `scripts/train_qlora_sft.py` 能跑通小模型 QLoRA。
- 数据格式和输出目录已经确认。

这样远程机器的时间会花在真正需要算力的地方，而不是基础排错。

## 选型

| 任务 | 推荐配置 |
| --- | --- |
| AWQ 推理服务 | T4 16GB / RTX 3060 12GB |
| Qwen2.5-VL-3B LoRA | RTX 4090 24GB / A10G 24GB |
| Qwen2.5-VL-7B LoRA | A100 40GB 起步 |

学习阶段优先考虑 24GB 档位，比较适合做完整实验。

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
pip install -e .
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

## 小心得

远程 GPU 最容易浪费时间的地方不是训练本身，而是环境、端口、缓存和进程管理。建议从第一天就用 `tmux`，并把模型缓存目录、项目目录和输出目录固定下来。这样下次登录服务器时，不需要重新猜文件在哪里。
