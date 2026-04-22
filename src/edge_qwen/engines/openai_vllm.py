from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from edge_qwen.config import ServiceConfig
from edge_qwen.engines.base import ChatEngine
from edge_qwen.schemas import ChatCompletionRequest


class OpenAIVLLMEngine(ChatEngine):
    """把请求转发到 vLLM 的 OpenAI 兼容服务。"""

    def __init__(self, config: ServiceConfig):
        self.config = config
        self.endpoint = f"{config.vllm_base_url.rstrip('/')}/chat/completions"

    def _payload(self, request: ChatCompletionRequest, stream: bool) -> dict:
        """把本服务请求整理成 vLLM/OpenAI 兼容格式。"""
        payload = request.model_dump(exclude_none=True)
        payload["model"] = self.config.model
        payload["stream"] = stream
        payload.setdefault("temperature", self.config.temperature)
        payload.setdefault("top_p", self.config.top_p)
        if request.max_tokens is None:
            payload["max_tokens"] = self.config.max_new_tokens
        return payload

    def _headers(self) -> dict[str, str]:
        """按需附加远程 vLLM 服务的鉴权头。"""
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    async def stream_chat(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                self.endpoint,
                headers=self._headers(),
                json=self._payload(request, stream=True),
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line.removeprefix("data: ").strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content

    async def complete_chat(self, request: ChatCompletionRequest) -> str:
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                self.endpoint,
                headers=self._headers(),
                json=self._payload(request, stream=False),
            )
            response.raise_for_status()
            payload = response.json()
            return payload["choices"][0]["message"]["content"]
