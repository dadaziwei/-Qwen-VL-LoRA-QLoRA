# 02 本地 AWQ 推理服务

这一章先把服务跑起来。我的习惯是先验证接口，再讨论训练。因为只要 API 形态定下来，后面不管底层换成 Transformers、vLLM 还是别的推理框架，调用方都不用大改。

## 1. 启动服务

在 PowerShell 中执行：

```powershell
.\.venv\Scripts\Activate.ps1
$env:EDGE_QWEN_BACKEND="transformers"
$env:EDGE_QWEN_MODEL="Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
uvicorn edge_qwen.api:app --host 0.0.0.0 --port 8000
```

也可以直接使用脚本：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_api.ps1
```

第一次启动会下载模型。这里我使用 AWQ 模型，是因为它更适合普通开发机做本地验证。

## 2. 健康检查

另开一个 PowerShell：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

正常会看到类似结果：

```json
{
  "status": "ok",
  "backend": "transformers",
  "model": "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
}
```

这个接口很简单，但很有用。后面部署到远程机器时，它可以快速判断服务是否活着、后端是否选对。

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

非流式请求适合调试，因为它一次返回完整 JSON，方便看结构。

## 4. 流式请求

```powershell
curl.exe -N http://127.0.0.1:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d "{\"model\":\"edge-qwen\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"介绍 QLoRA 的核心思想。\"}]}"
```

你会看到多段 `data: ...`，最后以：

```text
data: [DONE]
```

结束。聊天类产品里我更喜欢流式返回，因为用户能立刻看到模型开始工作，体感会好很多。

## 5. 多模态请求格式

Qwen-VL 的图片输入可以写成数组：

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

调试时我会先用较小图片。不是因为大图不能用，而是小图更容易定位问题：路径、格式、processor、模型输出，每一步都更清楚。

