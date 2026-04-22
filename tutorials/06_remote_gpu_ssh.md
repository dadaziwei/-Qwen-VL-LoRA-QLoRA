# 06 远程 GPU / SSH 实战

这一章讲远程 GPU。我的看法是：本地 8GB 很适合学习和开发，远程 GPU 适合跑重任务。两者不是替代关系，而是分工关系。

## 1. 什么时候需要远程 GPU

当你已经在本地跑通：

- FastAPI 接口。
- SSE 流式输出。
- QLoRA 小模型训练。
- AWQ 模型推理。

再去租远程 GPU 会更划算。因为这时你知道自己要跑什么，不会把租来的时间花在改路径、改 JSON、调接口这种小问题上。

## 2. 机器选择

| 任务 | 推荐配置 |
| --- | --- |
| AWQ 推理服务 | T4 16GB / RTX 3060 12GB |
| Qwen2.5-VL-3B LoRA | RTX 4090 24GB / A10G 24GB |
| 7B 多模态训练 | A100 40GB 起步 |

如果只是学习，我会优先选择 24GB 档位。它既能跑比较完整的实验，价格也通常比 A100 友好。

## 3. SSH 登录

```bash
ssh ubuntu@<remote-ip>
```

如果平台提供密钥：

```bash
ssh -i your_key.pem ubuntu@<remote-ip>
```

## 4. 初始化环境

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

## 5. 用 tmux 跑服务

我不建议直接在普通 SSH 前台长期跑服务。SSH 断开后进程很容易一起没了。更稳妥的方式是 `tmux`：

```bash
sudo apt install -y tmux
tmux new -s qwen
```

启动 vLLM：

```bash
export EDGE_QWEN_MODEL=Qwen/Qwen2.5-VL-3B-Instruct-AWQ
bash scripts/serve_vllm.sh
```

按 `Ctrl+B`，再按 `D`，可以退出会话但保留服务。

恢复会话：

```bash
tmux attach -t qwen
```

## 6. 端口转发

如果服务器端口没有直接开放，可以用 SSH 转发：

```bash
ssh -L 8000:127.0.0.1:8000 ubuntu@<remote-ip>
```

然后本地访问：

```bash
curl http://127.0.0.1:8000/health
```

## 7. 我的排错顺序

服务起不来时，我一般这样看：

1. `nvidia-smi`：确认 GPU 和驱动正常。
2. `pip list | grep vllm`：确认依赖装在当前虚拟环境。
3. `curl http://127.0.0.1:8001/v1/models`：确认 vLLM 服务是否活着。
4. `curl http://127.0.0.1:8000/health`：确认 FastAPI 是否活着。
5. 再看端口、安全组和 SSH 转发。

按层排查，比盯着一大段报错更快。

