# 01 环境准备

这一章先把开发环境搭好。我建议不要急着跑模型，先确认 Python、GPU、依赖、缓存目录这些基础东西。大模型项目里很多问题不是代码错，而是环境没整理好。

## 1. 看一下自己的开发机

在 PowerShell 中执行：

```powershell
nvidia-smi
```

我主要看三件事：

- 驱动是否正常识别显卡。
- 当前有多少程序在占用 GPU。
- 这台机器大概能承担什么阶段的任务。

我选择 8GB 作为教程基准，是因为它很常见，也足够完成“学习链路”：量化推理、API 服务、小模型 QLoRA。它不会让我们绕开工程问题，反而会促使我们认真做参数取舍。

## 2. 创建虚拟环境

在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

看到命令行前面出现 `(.venv)`，说明环境已经激活。

## 3. 安装依赖

本地开发先安装基础依赖：

```powershell
pip install -r requirements.txt
pip install -e .
```

`requirements-vllm.txt` 留给 Linux GPU 机器。我的做法是：本地先用 Transformers 后端把接口和数据流跑顺，远程机器再装 vLLM。这样排错更轻松。

## 4. 模型缓存目录

模型文件会比较大，建议单独放到空间充足的位置：

```powershell
$env:HF_HOME="E:\hf_cache"
$env:TRANSFORMERS_CACHE="E:\hf_cache\transformers"
```

如果下载慢，可以在远程服务器或镜像环境里提前缓存。学习阶段不要把时间全部耗在下载上，先保证项目结构和调用链路清楚。

## 5. 本地和远程的分工

我把项目分成两个环境：

- 本地开发机：写代码、调接口、跑 AWQ 推理、跑小模型 QLoRA。
- 远程 GPU：跑更完整的多模态训练、启动 vLLM、做并发服务。

这个分工很贴近真实开发。很多团队也是先在本地把接口和样例跑通，再把重任务交给远程机器。

