from __future__ import annotations

import json
import time
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from edge_qwen.config import ServiceConfig, get_config
from edge_qwen.engines import ChatEngine, OpenAIVLLMEngine, TransformersVLEngine
from edge_qwen.schemas import ChatCompletionRequest, HealthResponse

app = FastAPI(title="Edge Qwen API", version="0.1.0")


@lru_cache(maxsize=1)
def get_engine() -> ChatEngine:
    """懒加载推理后端，避免启动时重复加载模型。"""
    config = get_config()
    if config.backend == "openai":
        return OpenAIVLLMEngine(config)
    if config.backend == "transformers":
        return TransformersVLEngine(config)
    raise RuntimeError(f"Unsupported EDGE_QWEN_BACKEND={config.backend!r}")


def _chunk(delta: str, model: str) -> str:
    """构造 OpenAI 风格的 SSE chunk。"""
    payload = {
        "id": f"chatcmpl-edge-{int(time.time())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    config: ServiceConfig = get_config()
    return HealthResponse(status="ok", backend=config.backend, model=config.model)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    engine = get_engine()

    if request.stream:
        async def event_stream():
            """逐段输出模型结果；异常也用 SSE 形式返回给客户端。"""
            try:
                async for delta in engine.stream_chat(request):
                    yield _chunk(delta, request.model)
                yield "data: [DONE]\n\n"
            except Exception as exc:
                error = {"error": {"message": str(exc), "type": exc.__class__.__name__}}
                yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    try:
        content = await engine.complete_chat(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "id": f"chatcmpl-edge-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }
