# 01 环境准备

本章目标：把本地 Windows 机器准备成“能跑 API 服务、能做小模型 QLoRA 实验”的开发环境。

## 1. 确认显卡和驱动

在 PowerShell 中执行：

```powershell
nvidia-smi
```

重点看三项：

- `Driver Version`：驱动版本不能太旧。
- `CUDA Version`：只代表驱动支持的 CUDA 上限，不等于你安装的 PyTorch CUDA 版本。
- `Memory-Usage`：确认显存大约 8GB。

如果没有 NVIDIA 显卡，仍然可以阅读项目和启动部分 CPU 流程，但本地推理会很慢。

## 2. 创建虚拟环境

在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

看到命令行前面出现 `(.venv)`，说明环境已经激活。

## 3. 安装依赖

本地 Windows 推荐先只安装基础依赖：

```powershell
pip install -r requirements.txt
pip install -e .
```

`requirements-vllm.txt` 主要给 Linux GPU 机器使用。Windows 上 vLLM 支持并不理想，因此本地默认走 Transformers 后端。

## 4. 下载模型建议

国内网络下载 Hugging Face 模型可能较慢。你可以选择：

- 直接使用 Hugging Face：`Qwen/Qwen2.5-VL-3B-Instruct-AWQ`
- 使用镜像环境变量。
- 在远程服务器上提前下载模型缓存。

常用环境变量示例：

```powershell
$env:HF_HOME="E:\hf_cache"
$env:TRANSFORMERS_CACHE="E:\hf_cache\transformers"
```

## 5. 为什么本地不直接训练多模态模型

多模态模型除了语言模型权重，还包含视觉编码器、图像 token、较长上下文和更大的激活占用。8GB 显存在训练时很容易 OOM。

因此本项目采用两段式策略：

- 本地：AWQ 推理 + 小 Qwen 文本 QLoRA。
- 远程：Qwen-VL LoRA/QLoRA + vLLM 并发服务。

这个策略更符合真实边缘端项目：边缘端负责轻量推理和服务接入，训练和模型压缩可以在云端或租用算力上完成。

