from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def load_config(path: Path) -> dict:
    """读取 YAML 训练配置。"""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dtype_from_name(name: str):
    """根据硬件能力选择训练计算精度。"""
    if name == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if name in {"float16", "bfloat16"}:
        return torch.float16
    return torch.float32


def format_messages(example: dict, tokenizer) -> str:
    """把 messages/text/prompt-response 三种常见数据格式统一成训练文本。"""
    if "messages" in example:
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    if "text" in example:
        return example["text"]
    prompt = example.get("prompt") or example.get("instruction") or ""
    answer = example.get("response") or example.get("output") or ""
    return tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )


def build_sft_config(config: dict, output_dir: Path, compute_dtype) -> SFTConfig:
    """兼容不同 TRL 版本的 SFTConfig 参数命名。"""
    parameters = inspect.signature(SFTConfig.__init__).parameters
    kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": config["per_device_train_batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "num_train_epochs": config["num_train_epochs"],
        "learning_rate": config["learning_rate"],
        "logging_steps": config["logging_steps"],
        "save_steps": config["save_steps"],
        "bf16": compute_dtype is torch.bfloat16,
        "fp16": compute_dtype is torch.float16,
        "gradient_checkpointing": True,
        "optim": "paged_adamw_8bit",
        "report_to": "none",
    }
    if "max_seq_length" in parameters:
        kwargs["max_seq_length"] = config["max_seq_length"]
    elif "max_length" in parameters:
        kwargs["max_length"] = config["max_seq_length"]
    if "dataset_text_field" in parameters:
        kwargs["dataset_text_field"] = "text"
    return SFTConfig(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="面向 Qwen 文本模型的 QLoRA SFT 训练脚本。")
    parser.add_argument("--config", type=Path, default=Path("configs/local_8gb.yaml"))
    parser.add_argument("--dataset", type=Path, default=Path("data/sample_sft.jsonl"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = args.output_dir or Path(config["output_dir"])
    model_name = config["model_name"]

    quant = config.get("quantization", {})
    compute_dtype = dtype_from_name(quant.get("bnb_4bit_compute_dtype", "bfloat16"))
    quant_config = BitsAndBytesConfig(
        load_in_4bit=quant.get("load_in_4bit", True),
        bnb_4bit_quant_type=quant.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant.get("bnb_4bit_use_double_quant", True),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora = config["lora"]
    peft_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        task_type="CAUSAL_LM",
    )

    dataset = load_dataset("json", data_files=str(args.dataset), split="train")
    dataset = dataset.map(lambda example: {"text": format_messages(example, tokenizer)})

    training_args = build_sft_config(config, output_dir, compute_dtype)

    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "peft_config": peft_config,
        "args": training_args,
    }
    try:
        # 新旧 TRL 版本分别使用 tokenizer 或 processing_class。
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved LoRA adapter to {output_dir}")


if __name__ == "__main__":
    main()
