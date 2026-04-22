from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from threading import Thread

import torch
from transformers import AutoProcessor, TextIteratorStreamer

from edge_qwen.config import ServiceConfig
from edge_qwen.engines.base import ChatEngine
from edge_qwen.schemas import ChatCompletionRequest


class TransformersVLEngine(ChatEngine):
    """本地 Qwen-VL 推理后端，适合普通开发机上的 AWQ 模型验证。"""

    def __init__(self, config: ServiceConfig):
        self.config = config
        self.processor = AutoProcessor.from_pretrained(config.model, trust_remote_code=True)
        self.model = self._load_model()
        self.model.eval()

    def _load_model(self):
        """优先使用通用多模态入口，失败时回退到 Qwen2.5-VL 专用类。"""
        try:
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText.from_pretrained(
                self.config.model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=self.config.device_map,
                trust_remote_code=True,
            )
        except Exception:
            from transformers import Qwen2_5_VLForConditionalGeneration

            return Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.config.model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=self.config.device_map,
                trust_remote_code=True,
            )

    def _messages(self, request: ChatCompletionRequest) -> list[dict]:
        """把 Pydantic 对象还原成 Transformers chat template 接收的字典。"""
        return [message.model_dump() for message in request.messages]

    def _inputs(self, request: ChatCompletionRequest):
        """处理文本、图片和视频输入，并移动到模型所在设备。"""
        messages = self._messages(request)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        images = None
        videos = None
        try:
            from qwen_vl_utils import process_vision_info

            images, videos = process_vision_info(messages)
        except Exception:
            pass

        return self.processor(
            text=[text],
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

    def _generate_kwargs(self, request: ChatCompletionRequest) -> dict:
        """生成参数允许请求级覆盖，未传时使用服务默认值。"""
        temperature = self.config.temperature if request.temperature is None else request.temperature
        top_p = self.config.top_p if request.top_p is None else request.top_p
        return {
            "max_new_tokens": request.max_tokens or self.config.max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
        }

    async def stream_chat(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        inputs = self._inputs(request)
        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_kwargs = {
            **inputs,
            **self._generate_kwargs(request),
            "streamer": streamer,
        }
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs, daemon=True)
        thread.start()

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def pump() -> None:
            """跨线程读取 streamer，并安全投递回 asyncio 事件循环。"""
            for token in streamer:
                loop.call_soon_threadsafe(queue.put_nowait, token)
            loop.call_soon_threadsafe(queue.put_nowait, None)

        pump_thread = Thread(target=pump, daemon=True)
        pump_thread.start()

        while True:
            token = await queue.get()
            if token is None:
                break
            yield token

    async def complete_chat(self, request: ChatCompletionRequest) -> str:
        inputs = self._inputs(request)
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **self._generate_kwargs(request))

        prompt_len = inputs["input_ids"].shape[-1]
        generated_ids = generated_ids[:, prompt_len:]
        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
