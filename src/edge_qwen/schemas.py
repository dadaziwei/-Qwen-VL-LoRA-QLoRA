from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """OpenAI 风格的单条对话消息。"""

    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str | list[dict[str, Any]]


class ChatCompletionRequest(BaseModel):
    """兼容 `/v1/chat/completions` 的最小请求结构。"""

    model: str = "edge-qwen"
    messages: list[ChatMessage]
    stream: bool = True
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)


class HealthResponse(BaseModel):
    """健康检查返回信息。"""

    status: str
    backend: str
    model: str
