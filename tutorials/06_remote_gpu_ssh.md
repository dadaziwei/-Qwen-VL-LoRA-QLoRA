# 06 远程 GPU / SSH 实战

本章目标：从租用算力到启动服务，完整走一遍远程部署流程。

## 1. 选择机器

建议配置：

| 任务 | 推荐显卡 |
| --- | --- |
| AWQ 推理服务 | T4 16GB / RTX 3060 12GB |
| Qwen2.5-VL-3B LoRA | RTX 4090 24GB / A10G 24GB |
| 7B 多模态训练 | A100 40GB 起步 |

如果预算有限，优先租 24GB 显存机器，性价比较高。

## 2. SSH 登录

```bash
ssh ubuntu@<remote-ip>
```

如果平台提供密钥：

```bash
ssh -i your_key.pem ubuntu@<remote-ip>
```

## 3. 初始化环境

```bash
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

## 4. 后台运行服务

推荐使用 `tmux`：

```bash
sudo apt install -y tmux
tmux new -s qwen
```

启动 vLLM：

```bash
export EDGE_QWEN_MODEL=Qwen/Qwen2.5-VL-3B-Instruct-AWQ
bash scripts/serve_vllm.sh
```

按 `Ctrl+B`，再按 `D` 可以退出 tmux 会话，服务仍会继续运行。

恢复会话：

```bash
tmux attach -t qwen
```

## 5. 端口转发

如果远程安全组没有开放端口，可以从本地转发：

```bash
ssh -L 8000:127.0.0.1:8000 ubuntu@<remote-ip>
```

然后本地访问：

```bash
curl http://127.0.0.1:8000/health
```

## 6. 常见问题

`CUDA out of memory`：

- 降低 `MAX_MODEL_LEN`。
- 降低并发。
- 确认没有其他进程占用显存：`nvidia-smi`。

模型下载慢：

- 使用镜像源。
- 提前在服务器缓存模型。
- 使用平台提供的公共模型缓存。

服务断开：

- 使用 `tmux` 或 `systemd`。
- 不要直接在普通 SSH 前台长期运行生产服务。

