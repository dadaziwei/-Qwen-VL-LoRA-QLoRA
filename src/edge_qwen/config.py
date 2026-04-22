from __future__ import annotations

import os
from dataclasses import dataclass


def _float_env(name: str, default: float) -> float:
    """读取浮点型环境变量，未设置时使用默认值。"""
    value = os.getenv(name)
    return default if value is None else float(value)


def _int_env(name: str, default: int) -> int:
    """读取整型环境变量，未设置时使用默认值。"""
    value = os.getenv(name)
    return default if value is None else int(value)


@dataclass(frozen=True)
class ServiceConfig:
    """服务运行配置，统一由环境变量驱动，方便本地和远程切换。"""

    backend: str = os.getenv("EDGE_QWEN_BACKEND", "transformers")
    model: str = os.getenv("EDGE_QWEN_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct-AWQ")
    vllm_base_url: str = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001/v1")
    api_key: str | None = os.getenv("VLLM_API_KEY")
    max_new_tokens: int = _int_env("EDGE_QWEN_MAX_NEW_TOKENS", 512)
    temperature: float = _float_env("EDGE_QWEN_TEMPERATURE", 0.7)
    top_p: float = _float_env("EDGE_QWEN_TOP_P", 0.9)
    device_map: str = os.getenv("EDGE_QWEN_DEVICE_MAP", "auto")


def get_config() -> ServiceConfig:
    return ServiceConfig()
