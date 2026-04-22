# 03 QLoRA 轻量微调

本章目标：用 PEFT + QLoRA 跑通低显存微调闭环，并保存 LoRA Adapter。

## 1. 为什么选择小文本模型做本地训练

本地 8GB 显存很难稳定训练 Qwen2.5-VL-3B。为了保证项目能演示“冻结基座模型，只训练 LoRA Adapter”的核心能力，本地训练默认使用：

```text
Qwen/Qwen2.5-0.5B-Instruct
```

这不会削弱项目逻辑，因为 QLoRA 的训练流程、数据格式、LoRA 保存方式和大模型是一致的。真正多模态训练可以复用配置迁移到远程 GPU。

## 2. 数据格式

示例文件位于：

```text
data/sample_sft.jsonl
```

每一行是一条 SFT 样本：

```json
{"messages":[{"role":"system","content":"你是一个边缘端大模型部署助手。"},{"role":"user","content":"什么是 INT4 权重量化？"},{"role":"assistant","content":"INT4 权重量化是把模型权重从 FP16 等高精度格式压缩到 4 bit 表示，从而显著降低显存占用和存储体积。"}]}
```

你可以替换成自己的业务数据，例如设备说明书问答、接口文档问答、边缘部署排错问答。

## 3. 启动训练

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/train_qlora_sft.py `
  --config configs/local_8gb.yaml `
  --dataset data/sample_sft.jsonl `
  --output-dir outputs/qwen-lora-local
```

输出目录只保存 LoRA Adapter 和 tokenizer 文件，不会保存完整基座模型。

## 4. 关键参数解释

`configs/local_8gb.yaml` 中最重要的是：

- `max_seq_length: 512`：限制训练上下文，降低显存占用。
- `per_device_train_batch_size: 1`：单卡小 batch。
- `gradient_accumulation_steps: 8`：用梯度累积模拟更大 batch。
- `lora.r: 8`：低 rank，适合小显存。
- `load_in_4bit: true`：使用 4bit 量化加载基座模型。
- `bnb_4bit_quant_type: nf4`：QLoRA 常用量化格式。

## 5. 常见 OOM 处理

如果显存不足，按顺序降低：

1. `max_seq_length` 从 512 降到 384 或 256。
2. `lora.r` 从 8 降到 4。
3. 使用更小模型，例如 `Qwen/Qwen2.5-0.5B-Instruct`。
4. 关闭浏览器、IDE、其他占用 GPU 的程序。

## 6. 迁移到远程多模态训练

远程配置在：

```text
configs/remote_a10g.yaml
```

建议在 24GB 显存以上机器上执行，并把数据换成多模态样本。多模态样本通常需要图片路径、文本 prompt 和 assistant 回答，训练脚本可在当前文本 SFT 脚本基础上扩展 processor 处理逻辑。

