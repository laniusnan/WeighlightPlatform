# 推理模块接入包：模型压缩（drop-in）

把「模型压缩」挂到你网站**推理模块**下的最小实现：一个 FastAPI 路由文件 + 一份配置，
不依赖压缩平台的 Python 代码，只走 HTTP。

```text
examples/inference_module/
├── api/compression_proxy.py    # 路由：方法目录 / 提交 / 轮询 / 推理 / 产物 / 部署包
├── test_harness.py             # 本地验证用（uvicorn test_harness:app）
├── requirements.txt            # 依赖（fastapi / httpx / pydantic）
└── README.md
```

## 1. 安装与挂载

```bash
pip install -r requirements.txt        # fastapi, httpx, pydantic

# 你网站的推理模块（例如 app/main.py）
from api.compression_proxy import router as compression_router
app.include_router(compression_router)   # 自带 /api/compression 前缀
```

## 2. 配置（环境变量）

| 变量 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `COMPRESSION_API_BASE` | 是 | `http://127.0.0.1:8001/api/v1` | 压缩平台 API 根地址（CPU 栈 8001 / GPU 栈 8002） |
| `COMPRESSION_SERVICE_TOKEN` | 二选一 | — | 平台服务账号 token（推荐生产使用） |
| `COMPRESSION_EMAIL` / `COMPRESSION_PASSWORD` | 二选一 | — | 没给 token 时用它自动换取（带缓存与 401 刷新） |
| `COMPRESSION_TIMEOUT` | 否 | `60` | 普通请求超时（秒）；产物下载自动放宽到 ≥300s |
| `COMPRESSION_VERIFY_SSL` | 否 | `true` | 内网自签证书可设 `false` |

```bash
export COMPRESSION_API_BASE=http://10.110.1.36:8001/api/v1
export COMPRESSION_EMAIL=admin@example.com
export COMPRESSION_PASSWORD=******
```

## 3. 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/compression/methods` | UI 方法目录（下拉框 + 表单字段 + 默认参数），前端据此渲染 |
| GET | `/api/compression/platform/models` | 平台模型（下拉框） |
| GET | `/api/compression/platform/datasets` | 平台数据集（含 `local` reference 数据集） |
| GET | `/api/compression/platform/local-datasets` | 扫描只读根 `/datasets-local`（类型/类别数/样本数） |
| POST | `/api/compression/tasks` | **提交压缩任务**（202 + `task_id`，异步） |
| GET | `/api/compression/tasks` | 任务列表（`?status=&limit=`） |
| GET | `/api/compression/tasks/{task_id}` | 任务状态（前端轮询；`progress`、`ready`、`metrics`） |
| POST | `/api/compression/tasks/{task_id}/retry` | 重试失败任务（202） |
| POST | `/api/compression/tasks/{task_id}/reevaluate` | 重跑已存配置的 eval/benchmark（`{"kind":"eval"}`） |
| DELETE | `/api/compression/tasks/{task_id}` | 删除任务（运行中先取消） |
| GET | `/api/compression/tasks/{task_id}/artifact` | 下载压缩产物 zip（**流式**转发） |
| GET | `/api/compression/tasks/{task_id}/deploy` | 下载部署包（Dockerfile + serve.py + model_card.json） |
| POST | `/api/compression/infer` | **用压缩产物推理**（CNN/ONNX/OV 走 predict；`chat=true`/带 `messages` 走 LLM chat） |

### 方法目录（`GET /methods` 的 key）

`int4` · `int8_openvino` · `onnx_int8` · `onnx_static_int8` · `gguf_q4km` ·
`prune_l2` · `prune_fisher` · `distill_logits` · `distill_features` · `lora` · `dpo`

> 平台新增方法时，只需在 `METHOD_CATALOG` 里加一条（或让前端直接用平台返回的字段），前端无需改渲染逻辑。

## 4. 前端流程（推荐）

```text
[模型详情 · 优化标签页]
  ① GET  /methods            → 渲染「压缩方法」下拉框 + 高级参数表单
  ② GET  /platform/models    → 渲染「选择模型」（或复用你已有的模型下拉）
  ③ （数据驱动方法）GET /platform/local-datasets → 渲染「选择数据集」
  ④ POST /tasks              → 提交，拿到 task_id，展示进度条
  ⑤ GET  /tasks/{id}         → 每 1~3s 轮询（或平台 SSE），直到 ready=true
  ⑥ 产物就绪：
       · 直接用   → POST /infer {task_id, ...}
       · 下载 zip → GET  /tasks/{id}/artifact（<a download> 即可）
       · 出部署包 → GET  /tasks/{id}/deploy
  ⑦ 失败 → 展示 error；可 POST /tasks/{id}/retry
```

`POST /tasks` 请求体：

```json
{
  "model_id": "CifarCNN10",              // 平台 UUID 或名称（唯一前缀即可）
  "method": "prune_l2",                  // /methods 返回的 key
  "params": {"rate": 0.2, "input_shape": [1, 3, 32, 32]},   // 覆盖默认值
  "eval": {"type": "classification", "dataset_id": "<uuid>", "image_size": 32}   // 可选
}
```

## 5. 端到端实测（本目录代码真机跑通）

```bash
# 1) 起服务（把 8001 换成你自己的平台地址）
export COMPRESSION_API_BASE=http://10.110.1.36:8001/api/v1
export COMPRESSION_EMAIL=admin@example.com COMPRESSION_PASSWORD=******
uvicorn test_harness:app --host 0.0.0.0 --port 8899

# 2) 提交压缩（异步）
curl -s localhost:8899/api/compression/tasks -H 'Content-Type: application/json' \
  -d '{"model_id":"CifarCNN10","method":"prune_l2","params":{"rate":0.2,"input_shape":[1,3,32,32]}}'
# {"task_id":"804aac30-…","status":"pending","platform_method":"prune","params":{...}}

# 3) 轮询
curl -s localhost:8899/api/compression/tasks/804aac30-…
# {…"status":"succeeded","progress":100.0,"ready":true,
#  "metrics":{"total_filters":224,"pruned_filters":46,"forward_check":"ok",
#             "head_layers_adjusted":1,"onnx_exported":true},
#  "artifact_url":"/api/compression/tasks/804aac30-…/artifact",
#  "deploy_url":"/api/compression/tasks/804aac30-…/deploy"}

# 4) 用压缩产物推理
curl -s localhost:8899/api/compression/infer -H 'Content-Type: application/json' \
  -d '{"task_id":"804aac30-…","inputs":[[[[0.5,...]]]]}'
# {"output_shape":[1,10],"output":[[-0.066, -0.015, …]],"latency_ms":…}

# 5) 产物 / 部署包
curl -sJO localhost:8899/api/compression/tasks/804aac30-…/artifact    # 449,585 B
curl -sJO localhost:8899/api/compression/tasks/804aac30-…/deploy      # 451,391 B
```

**实测结果**：15 项断言全绿（方法目录/资产盘点/提交/轮询/推理/下载/错误映射），
错误映射示例：未知方法 → 422、数据驱动方法缺数据集 → 422、模型不存在 → 404、任务不存在 → 404。

## 6. 实现要点（踩坑记录）

| 要点 | 说明 |
|---|---|
| **流式下载** | 转发产物时**不能**用 `async with httpx.AsyncClient(...)`：上下文退出即关闭连接，浏览器会看到 0 字节 / `IncompleteRead`。本实现把 client 的关闭放在响应生成器的 `finally` 里。 |
| **目录形态产物** | `prune`/`qat`/`distill` 的产物是**目录**；平台侧已支持按目录解析出具体模型文件（`model*.onnx` / `model*.pt` / OpenVINO IR），因此 `/infer` 直接传 `task_id` 即可。 |
| **异步 + 轮询** | 平台任务由独立 worker 执行，提交接口不阻塞；前端用 `progress` 渲染进度条，不要同步等待。 |
| **服务账号** | 生产环境建议用平台的服务账号 token（`COMPRESSION_SERVICE_TOKEN`），避免把用户密码放进前端或环境变量。 |
| **权限隔离** | 平台侧按任务 owner 校验；服务账号（superuser）可跨用户读取任务，注意在路由层限制可访问的 task_id。 |
