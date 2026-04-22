# 03 QLoRA 轻量微调

这一章的目标不是训练出一个“最强模型”，而是把 QLoRA 的工程流程跑通。只要你理解了这个流程，后面换更大的模型、更多数据、远程 GPU，本质上都是扩展同一套方法。

## 1. 为什么本地先选小模型

我本地训练默认选择：

```text
Qwen/Qwen2.5-0.5B-Instruct
```

原因很朴素：学习阶段最重要的是反馈速度。小模型能让我们快速看到数据如何被 chat template 处理、4bit 如何加载、LoRA Adapter 如何保存。等这条链路顺了，再迁移到 Qwen-VL 多模态训练会踏实很多。

8GB 显卡在这里很有意义。它逼着我们使用 QLoRA，而不是全量微调；也逼着我们理解 `max_seq_length`、batch size、gradient accumulation、LoRA rank 这些参数到底在影响什么。

## 2. 数据格式

示例数据位于：

```text
data/sample_sft.jsonl
```

每一行是一条对话：

```json
{"messages":[{"role":"system","content":"你是一个边缘端大模型部署助手。"},{"role":"user","content":"什么是 INT4 权重量化？"},{"role":"assistant","content":"INT4 权重量化是把模型权重从 FP16 等高精度格式压缩到 4 bit 表示，从而降低模型部署成本。"}]}
```

你可以把它替换成自己的领域数据，比如：

- 设备说明书问答。
- API 文档问答。
- 部署排错记录。
- 图像识别任务的文本解释样本。

## 3. 启动训练

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/train_qlora_sft.py `
  --config configs/local_8gb.yaml `
  --dataset data/sample_sft.jsonl `
  --output-dir outputs/qwen-lora-local
```

训练完成后，输出目录保存的是 LoRA Adapter，而不是完整基座模型。这正是 PEFT 的好处：小文件、易迁移、易组合。

## 4. 我关注的几个参数

`configs/local_8gb.yaml` 里这些参数最值得看：

- `max_seq_length: 512`：先用较短上下文换来稳定反馈。
- `per_device_train_batch_size: 1`：单步小 batch，更适合普通开发机。
- `gradient_accumulation_steps: 8`：用累积梯度模拟更大的有效 batch。
- `lora.r: 8`：先用较小 rank 观察效果和训练速度。
- `load_in_4bit: true`：QLoRA 的核心之一，用 4bit 加载基座模型。
- `bnb_4bit_quant_type: nf4`：QLoRA 常用格式。

我的建议是一次只改一个参数。否则训练效果变化了，你很难判断原因。

## 5. 训练不稳时怎么调

如果训练过程不顺，我通常按这个顺序排查：

1. 先确认数据格式能被脚本正确读取。
2. 再把 `max_seq_length` 调小，确认不是上下文太长。
3. 再把 `lora.r` 调小，观察资源占用变化。
4. 最后再考虑换模型或迁移到远程 GPU。

这种排查方式比一上来换一堆参数更可靠。

## 6. 迁移到多模态训练

远程配置在：

```text
configs/remote_a10g.yaml
```

后续做 Qwen-VL LoRA 时，主要增加的是多模态数据处理：图片路径、文本 prompt、assistant 回答，以及 processor 对图片的编码。训练思想仍然是同一套：冻结基座，训练 Adapter。

