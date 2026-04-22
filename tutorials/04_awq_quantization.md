# 04 AutoAWQ INT4 量化

这一章讲 AWQ。我的理解是：如果 QLoRA 解决的是“怎样更便宜地训练”，那么 AWQ 解决的就是“怎样更轻地部署”。

## 1. 为什么需要 AWQ

直接用 FP16 模型做推理，体验往往不够友好：模型文件大、加载慢、运行成本高。学习大模型部署时，量化是绕不开的一步。

AWQ 的价值在于把权重压到 INT4，同时尽量保留模型能力。它非常适合做边缘端或个人开发机上的推理验证。

本项目本地推理默认选择：

```text
Qwen/Qwen2.5-VL-3B-Instruct-AWQ
```

这样可以把主要精力放在 API 服务、流式返回和后端抽象上，而不是一开始就陷入量化工具链排错。

## 2. 校准数据怎么准备

量化时需要校准数据。我的经验是：校准数据不一定要很多，但最好接近真实场景。

示例：

```text
data/sample_sft.jsonl
```

如果你的项目面向设备问答，就放设备问答；如果面向文档理解，就放文档类 prompt。校准数据越贴近任务，量化后越稳。

## 3. 执行量化

量化自有文本模型：

```bash
source .venv/bin/activate
python scripts/quantize_awq.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir outputs/qwen2_5_1_5b_awq \
  --calib data/sample_sft.jsonl \
  --calib-size 128
```

完成后会得到：

- 量化模型权重。
- tokenizer 文件。
- AWQ 量化配置。

## 4. 参数理解

- `w_bit: 4`：权重使用 4bit 表示。
- `q_group_size: 128`：常见分组大小，兼顾效果和部署。
- `version: GEMM`：比较通用的推理部署版本。
- `zero_point: true`：常见的量化配置。

这些参数不用一开始就调得很复杂。先用默认值跑通，再做对比实验。

## 5. 我的建议

学习阶段先用官方 AWQ 模型，原因很简单：它能让你更快进入服务化部署和接口设计。

等你已经能稳定启动 FastAPI、跑通 SSE、理解 QLoRA 后，再回头做自定义量化会更自然。否则很容易把时间耗在安装和算子兼容上，反而看不到完整项目。

