# 教程总览

这份教程是按“我真的要把这个项目做出来”的顺序写的，不是把命令简单排一下。每一章都会尽量说明：为什么要这么做、这个选择解决了什么问题、如果换到更强的机器上该怎么扩展。

我把 8GB 显卡作为教程基准，是因为它有代表性。很多刚开始做大模型部署的人，设备并不是服务器，而是自己电脑里一张 8GB 显卡。用这样的配置做项目，能逼着我们认真理解量化、Adapter、上下文长度、服务抽象这些真正重要的东西，而不是简单依赖硬件把问题盖过去。

## 阅读顺序

1. [环境准备](./01_environment.md)：从开发机检查、虚拟环境、依赖安装开始。
2. [本地 AWQ 推理服务](./02_local_inference.md)：先把 API 服务跑起来，确认请求和流式返回都通。
3. [QLoRA 轻量微调](./03_qlora_finetune.md)：用小模型跑通训练闭环，理解 LoRA Adapter 的价值。
4. [AutoAWQ INT4 量化](./04_awq_quantization.md)：理解量化流程，以及为什么学习阶段可以先用现成 AWQ 权重。
5. [vLLM + FastAPI 部署](./05_vllm_fastapi_deploy.md)：把推理引擎和业务 API 分层。
6. [远程 GPU / SSH 实战](./06_remote_gpu_ssh.md)：把重计算部分迁移到租用算力。
7. [项目汇报与简历描述](./07_report_and_resume.md)：把项目讲清楚，而不是只堆技术名词。

## 我推荐的执行路线

本地开发机先做这些：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
pip install -e .
powershell -ExecutionPolicy Bypass -File scripts/start_api.ps1
```

等本地接口、训练脚本和数据格式都理解后，再上远程 GPU：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt -r requirements-vllm.txt
pip install -e .
bash scripts/serve_vllm.sh
```

## 为什么这个项目有学习意义

很多大模型教程会直接假设你有一张很强的卡，然后把模型拉下来跑。这样确实快，但容易跳过几个关键问题：

- 模型为什么需要量化？
- 为什么微调时不直接更新全部参数？
- 为什么推理服务要和业务 API 解耦？
- 为什么流式输出比一次性返回更适合聊天场景？
- 为什么本地开发和远程部署要保持同一套接口？

这个项目的学习价值就在这里：用普通硬件把工程链路拆开，一块一块理解。

## 最终你应该能讲清楚

- 我为什么选择 AWQ 而不是直接 FP16 推理。
- 我为什么选择 QLoRA 而不是全量微调。
- 本地 Transformers 后端和远程 vLLM 后端分别适合什么阶段。
- FastAPI 在项目里不是摆设，而是稳定业务接口的边界。
- SSE 流式返回如何改善生成式接口的体验。
- 如果以后有 24GB 或更强 GPU，哪些部分可以继续升级。

