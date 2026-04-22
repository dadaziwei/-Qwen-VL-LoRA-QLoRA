# 04 AutoAWQ INT4 量化

本章目标：理解 AutoAWQ 的作用，并掌握如何把自有 Qwen 文本模型量化成 INT4 权重。

## 1. AWQ 解决什么问题

原生 FP16 模型显存占用高。以 3B 模型为例，纯权重就可能需要约 6GB，再加上 KV cache、激活、框架开销，很容易超过边缘端预算。

AWQ 会把大部分权重量化为 INT4，从而降低：

- 显存占用。
- 模型文件体积。
- 边缘端加载门槛。

本项目优先使用现成模型：

```text
Qwen/Qwen2.5-VL-3B-Instruct-AWQ
```

如果你需要量化自己的文本模型，可以使用 `scripts/quantize_awq.py`。

## 2. 准备校准数据

校准数据不需要很大，但要贴近真实使用场景。示例：

```text
data/sample_sft.jsonl
```

AutoAWQ 会用这些文本估计激活分布，从而减少量化后的精度损失。

## 3. 执行量化

建议在远程 GPU 上执行：

```bash
source .venv/bin/activate
python scripts/quantize_awq.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir outputs/qwen2_5_1_5b_awq \
  --calib data/sample_sft.jsonl \
  --calib-size 128
```

量化完成后会保存：

- 量化后的模型权重。
- tokenizer 文件。
- AWQ 量化配置。

## 4. 关键参数

- `w_bit: 4`：权重量化到 4 bit。
- `q_group_size: 128`：常见分组大小，兼顾精度和速度。
- `version: GEMM`：通用推理部署版本。
- `zero_point: true`：使用零点量化，通常更稳。

## 5. 本地 8GB 为什么不建议现场量化多模态大模型

量化过程需要加载原始模型，并执行校准计算。对于 Qwen-VL，多模态结构更复杂，现场量化可能比推理更吃内存。

更合理的方案是：

- 本地直接使用官方/社区 AWQ 模型。
- 远程 GPU 完成自有模型量化。
- 量化后把权重同步到边缘端。

