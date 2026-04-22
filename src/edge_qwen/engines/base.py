from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from edge_qwen.schemas import ChatCompletionRequest


class ChatEngine(ABC):
    @abstractmethod
    async def stream_chat(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        """按文本增量流式返回模型输出。"""

    @abstractmethod
    async def complete_chat(self, request: ChatCompletionRequest) -> str:
        """一次性返回完整模型输出。"""
