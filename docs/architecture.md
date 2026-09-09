# Architecture（节选：计算后端抽象层）

## 计算后端抽象层（2026-09 引入，GPU 迁移准备）

### 职责分层

```
API 层（backend/app/api）          —— 接口不变，不感知后端
    │  通过既有 jobs/路由调用
    ▼
任务层（backend/app/jobs）
    run_compression_task(task_id)   —— CPU 全量管线（原实现，未改）
    jobs.py::run_compression        —— 经 services 工厂路由
    ▼
服务层（backend/app/services）
    base.py      抽象基类 + 常量 + backend/device 解析 + configure_runtime
    factory.py   ServiceFactory（按 COMPUTE_BACKEND 返回缓存 ServiceBundle）
    *_cpu.py     现役实现：委托既有 CPU 逻辑（单一事实来源，无代码复制）
    *_gpu.py     骨架 + TODO（NotImplementedBackendError）
    ▼
计算库          torch / openvino(+GPU plugin) / onnxruntime(+CUDA/TensorRT)
```

### 设计约束
- **接口不变**：`/api/v1/*` 全部保持；服务层仅在 worker 执行路径接入，不影响路由。
- **CPU 优先**：`COMPUTE_BACKEND` 默认 `cpu`；gpu 分支未实现即抛
  `NotImplementedBackendError` 并把任务标记 failed（不会悬挂或静默）。
- **惰性导入**：CPU 进程不会 import GPU 相关包；gpu 分支内才 `from ..._gpu import ...`。
- **逐步使能**：每个方法单独从 `NOT_IMPLEMENTED_YET` 移除即可上线，无需整体切换。

### 典型调用链（Arq worker，CPU 现役）
`app.worker.WorkerSettings` → `jobs.run_compression(ctx, task_id)`
→ `get_service_bundle().runner.run(task_id)`（CpuCompressionRunner）
→ `jobs.compression.run_compression_task(task_id)`（原逻辑，含进度/评估/基准钩子）。
