# Windows 本地快速启动 FastAPI；默认走 Transformers 后端。
$env:EDGE_QWEN_BACKEND = if ($env:EDGE_QWEN_BACKEND) { $env:EDGE_QWEN_BACKEND } else { "transformers" }
$env:EDGE_QWEN_MODEL = if ($env:EDGE_QWEN_MODEL) { $env:EDGE_QWEN_MODEL } else { "Qwen/Qwen2.5-VL-3B-Instruct-AWQ" }

uvicorn edge_qwen.api:app --host 0.0.0.0 --port 8000
