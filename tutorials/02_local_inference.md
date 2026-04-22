# 02 本地 AWQ 推理服务

本章目标：在本地启动 FastAPI 服务，通过 Transformers 后端加载 AWQ 模型，并验证 SSE 流式返回。

## 1. 启动服务

在 PowerShell 中执行：

```powershell
.\.venv\Scripts\Activate.ps1
$env:EDGE_QWEN_BACKEND="transformers"
$env:EDGE_QWEN_MODEL="Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
uvicorn edge_qwen.api:app --host 0.0.0.0 --port 8000
```

也可以使用脚本：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_api.ps1
```

第一次启动会下载模型，耗时取决于网络。

## 2. 健康检查

打开新 PowerShell：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

正常会看到：

```json
{
  "status": "ok",
  "backend": "transformers",
  "model": "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
}
```

## 3. 非流式请求

```powershell
$body = @{
  model = "edge-qwen"
  stream = $false
  messages = @(
    @{ role = "user"; content = "用三句话说明为什么边缘端需要模型量化。" }
  )
} | ConvertTo-Json -Depth 10

Invoke-RestMethod http://127.0.0.1:8000/v1/chat/completions `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

## 4. 流式请求

PowerShell 对 SSE 展示不如浏览器直观，推荐使用 curl：

```powershell
curl.exe -N http://127.0.0.1:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d "{\"model\":\"edge-qwen\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"介绍 QLoRA 的核心思想。\"}]}"
```

你会看到多段 `data: ...` 输出，最后以：

```text
data: [DONE]
```

结束。

## 5. 多模态请求格式

Qwen-VL 支持图片输入。请求中的 `content` 可以是数组：

```json
{
  "model": "edge-qwen",
  "stream": true,
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image",
          "image": "file:///E:/images/demo.jpg"
        },
        {
          "type": "text",
          "text": "请描述图片内容。"
        }
      ]
    }
  ]
}
```

8GB 显存下图片分辨率不要太高，否则视觉 token 会明显增加显存占用。

