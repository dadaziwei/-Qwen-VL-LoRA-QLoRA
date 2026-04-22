from __future__ import annotations

import argparse
from pathlib import Path

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


def load_calibration(path: Path, limit: int) -> list[str]:
    """从 jsonl 样本中提取 AutoAWQ 校准文本。"""
    import json

    samples: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(samples) >= limit:
                break
            row = json.loads(line)
            if "text" in row:
                samples.append(row["text"])
                continue
            if "messages" in row:
                text = "\n".join(f"{item['role']}: {item['content']}" for item in row["messages"])
                samples.append(text)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 AutoAWQ 对 Qwen 文本模型进行 INT4 权重量化。")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen-awq"))
    parser.add_argument("--calib", type=Path, default=Path("data/sample_sft.jsonl"))
    parser.add_argument("--calib-size", type=int, default=128)
    parser.add_argument("--w-bit", type=int, default=4)
    parser.add_argument("--q-group-size", type=int, default=128)
    args = parser.parse_args()

    quant_config = {
        # GEMM 版本适合通用推理部署，q_group_size=128 是 AWQ 常用设置。
        "zero_point": True,
        "q_group_size": args.q_group_size,
        "w_bit": args.w_bit,
        "version": "GEMM",
    }

    model = AutoAWQForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        safetensors=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    calibration = load_calibration(args.calib, args.calib_size)

    model.quantize(tokenizer, quant_config=quant_config, calib_data=calibration)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(args.output_dir), safetensors=True)
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Saved AWQ model to {args.output_dir}")


if __name__ == "__main__":
    main()
