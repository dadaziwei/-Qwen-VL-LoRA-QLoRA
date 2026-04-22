# 项目教程总览

本目录是 `Edge Qwen QLoRA + AWQ + FastAPI` 的完整中文教程，目标是让你在 8GB 显存机器上先跑通可展示闭环，再把更重的训练和并发推理迁移到远程 GPU。

## 学习路线

建议按下面顺序阅读和执行：

1. [环境准备](./01_environment.md)：安装 Python 环境、依赖、CUDA/显卡检查和 Hugging Face 下载建议。
2. [本地 AWQ 推理服务](./02_local_inference.md)：使用 Transformers 后端启动 FastAPI，并通过 SSE 流式返回结果。
3. [QLoRA 轻量微调](./03_qlora_finetune.md)：在 8GB 显存下用小 Qwen 文本模型跑通 PEFT + QLoRA 训练闭环。
4. [AutoAWQ INT4 量化](./04_awq_quantization.md)：理解量化流程，并在远程 GPU 上量化自有模型。
5. [vLLM + FastAPI 部署](./05_vllm_fastapi_deploy.md)：用 vLLM 做底层推理引擎，再由 FastAPI 统一封装业务接口。
6. [远程 GPU / SSH 实战](./06_remote_gpu_ssh.md)：租用算力、SSH 登录、端口转发、后台运行和常见排错。
7. [项目汇报与简历描述](./07_report_and_resume.md)：整理项目亮点、技术路线、答辩话术和简历表达。

## 推荐执行路径

本地 8GB 显存机器：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
pip install -e .
powershell -ExecutionPolicy Bypass -File scripts/start_api.ps1
```

远程 Linux GPU 机器：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt -r requirements-vllm.txt
pip install -e .
bash scripts/serve_vllm.sh
```

## 8GB 显存边界

8GB 显存适合：

- 运行 `Qwen/Qwen2.5-VL-3B-Instruct-AWQ` 的轻量推理演示。
- 用 `Qwen/Qwen2.5-0.5B-Instruct` 或 `Qwen/Qwen2.5-1.5B-Instruct` 跑 QLoRA 文本微调闭环。
- 展示 FastAPI + SSE 服务封装和客户端接入。

8GB 显存不建议：

- 直接 FP16 加载 Qwen2.5-VL-3B/7B。
- 本地训练多模态 Qwen-VL LoRA。
- 大上下文、多图片、多并发压测。

## 项目最终验收标准

完成后你应该能展示：

- 一个可启动的 `/health` 接口。
- 一个兼容 OpenAI 风格的 `/v1/chat/completions` 接口。
- 一个支持 `stream=true` 的 SSE 流式响应。
- 一个 QLoRA 训练脚本和 LoRA Adapter 输出目录。
- 一个 AutoAWQ 量化脚本。
- 一套远程 vLLM 部署方案。
- 一份可以写进简历和项目报告的技术说明。

