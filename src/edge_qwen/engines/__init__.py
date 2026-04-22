"""推理后端集合：本地 Transformers 与远程 vLLM 代理。"""

from .base import ChatEngine
from .openai_vllm import OpenAIVLLMEngine
from .transformers_vl import TransformersVLEngine

__all__ = ["ChatEngine", "OpenAIVLLMEngine", "TransformersVLEngine"]
