/**
 * routes/lab/docs.tsx · 技术使用文档（双语，平台内内置）
 */
import { createFileRoute } from "@tanstack/react-router"
import { BookOpen, CheckCircle2, Lightbulb } from "lucide-react"

import { useI18n } from "@/components/lab/i18n"
import { Card } from "@/components/lab/ui"

export const Route = createFileRoute("/lab/docs")({
  component: DocsPage,
  head: () => ({ meta: [{ title: "Docs - AI Model Lightweight Platform" }] }),
})

interface DocSection {
  icon?: typeof BookOpen
  titleZh: string
  titleEn: string
  items: { zh: string; en: string }[]
}

const SECTIONS: DocSection[] = [
  {
    titleZh: "快速上手",
    titleEn: "Quick Start",
    items: [
      { zh: "登录平台后，先到「模型管理」上传/拉取/本地导入一个模型，状态变为 Ready 后才能压缩。", en: "Log in, then add a model (Upload / Pull / Import local) in Models until it is Ready." },
      { zh: "在「压缩任务」点右上角「新建压缩」，选择 Ready 模型与压缩方法并填参数，点「创建任务」。", en: "In Compressions click New Task, pick a Ready model + a method, then Create Task." },
      { zh: "任务由后台 Worker 执行；进度条/日志/产物在任务列表可见。", en: "Jobs run on the background Worker; progress and artifacts show in the task list." },
    ],
  },
  {
    titleZh: "模型来源",
    titleEn: "Model Sources",
    items: [
      { zh: "上传：.pt/.pth（PyTorch）、.onnx、.h5 等单文件或 zip。", en: "Upload: single .pt/.pth, .onnx, .h5 or a zip archive." },
      { zh: "URL 拉取：HuggingFace repo id 或直链；HF 不可达时自动走 hf-mirror/ModelScope。", en: "Pull: a HuggingFace repo id or direct URL, with hf-mirror/ModelScope fallbacks." },
      { zh: "本地导入：把 HF 目录放到服务器挂载目录（如 /home/zhangxuefei/Model → /models-local），填容器内路径即可（source_type=local）。", en: "Import local: put the HF dir under the mounted root (e.g. /models-local), then submit its container path." },
    ],
  },
  {
    titleZh: "LLM 量化（engine=nncf）",
    titleEn: "LLM Quantization (engine=nncf)",
    items: [
      { zh: "适用：本地 HF 目录的 decoder（Qwen/Llama 等）。方法 llm_quantize，参数 engine=nncf。", en: "Use llm_quantize with engine=nncf on a local HF decoder (Qwen/Llama…)." },
      { zh: "推荐：bits=4、group_size=128（隐藏层可整除即可 64/128/256）、sym=true。", en: "Recommended: bits=4, group_size=128 (64/128/256 work when the hidden dim divides), sym=true." },
      { zh: "产物 = OpenVINO IR(model.xml) + tokenizer + engine_nncf.txt 标记，可直接挂到 /api/v1/inference/chat/completions。", en: "Output is an OpenVINO IR + tokenizer (+ engine_nncf.txt) that serves /api/v1/inference/chat/completions." },
    ],
  },
  {
    titleZh: "评估与部署",
    titleEn: "Evaluate & Deploy",
    items: [
      { zh: "「评估对比」支持分类/YOLO/Perplexity，可附带 dataset_id 或 data_url 数据集。", en: "Evaluate supports classification / YOLO / perplexity with a platform dataset or URL." },
      { zh: "「部署导出」提供推理下载与 chat 端点信息；模型产物可用 /inference/predict 或 chat/completions 调起。", en: "Deploy shows downloads and the chat endpoint; artifacts run via /inference/predict or chat/completions." },
    ],
  },
  {
    titleZh: "环境与 GPU",
    titleEn: "Environment & GPU",
    items: [
      { zh: "默认 CPU 栈：Python3.11 + torch2.9.1(cpu) + transformers4.49 + OpenVINO/NNCF（已验证 GPT/Qwen 解码器）。", en: "Default CPU stack: Python 3.11, torch 2.9.1 (cpu), transformers 4.49, OpenVINO/NNCF." },
      { zh: "GPU 迁移处于规划阶段：COMPUTE_BACKEND=cpu|gpu；inference_gpu 已支持 CUDA fp16 HF 对话，压缩类 GPU 实现为骨架。", en: "GPU is planned: COMPUTE_BACKEND=cpu|gpu; inference.gpu already supports CUDA fp16 chat, compression GPU runners are skeletons." },
      { zh: "本地排查：python backend/scripts/check_backend.py（打印当前后端与服务实例）。", en: "Debug locally: python backend/scripts/check_backend.py." },
    ],
  },
  {
    titleZh: "压缩方法使用指南（全量）",
    titleEn: "Usage guide for every method",
    items: [
      { zh: "export_onnx（转 ONNX）：适用 CNN/YOLO/PointNet .pt/.pth；参数 model_kind/cnn|yolo|pointnet、input_shape 如 1,3,224,224、opset 12；产物 model.onnx（含 onnxsim 简化），可下载并在 /inference/predict 测试。", en: "export_onnx: CNN/YOLO/PointNet PyTorch -> simplified ONNX." },
      { zh: "quantize_onnx（ONNX 动态 INT8）：输入为 .onnx；权重量化无需数据，产出动态 int8 版；适合快速压体积（体积降但速度提升有限）。", en: "quantize_onnx: weight-only dynamic INT8, no data needed." },
      { zh: "quantize_onnx_static（ONNX 静态 INT8）：需要校准数据（zip 图片或 data_url，可选平台数据集），calibration_samples 控制样本量；精度更高、CPU 推理更快。", en: "quantize_onnx_static: calibration-based static INT8, needs images." },
      { zh: "openvino_int8（OV INT8，NNCF）：任意可用模型（torch/onnx），precision=int8/fp16/fp32，model_kind 与 input_shape；产出 OpenVINO IR（model.xml），适合 CPU/GPU(Intel) 部署。", en: "openvino_int8: NNCF quantized OpenVINO IR." },
      { zh: "prune（结构化剪枝）：PyTorch + arch.py；algorithm=l1|l2|bnslim|group_fisher|autoslim|dmcp（L1/L2 按范数；bnslim/autoslim 按 BN γ，其中 autoslim 先做 L1 稀疏训练再按 γ 剪；group_fisher 用梯度平方；dmcp 用一阶 Taylor 近似——后三者需按类别分文件夹的数据集），rate=0~1；产出更小 .pt。", en: "prune: structured pruning L1/L2/BN-Slim with ratio." },
      { zh: "sparsify（非结构化稀疏）：strategy=uniform|global 与 rate；产出稀疏 .pt（适合后续再压缩）。", en: "sparsify: unstructured uniform/global sparsity." },
      { zh: "distill（知识蒸馏）：教师=上传 PyTorch checkpoint+arch.py，需按类别分文件夹的数据集(zip)。FitNets 步骤：① 模型管理上传含 arch.py 的 CNN 教师，并上传按类别分文件夹的 zip；② 压缩任务→知识蒸馏→蒸馏方式选 FitNets 特征层对齐，width_mult=0.5 等；③ 完成后 metrics 含 method: features，产物 student .pt/.onnx。method=logits（Hinton 软标签，任意分类骨干）或 features（FitNets，需 CNN 特征图）。", en: "distill: logits KD or FitNets; teacher + class-folder dataset needed." },
      { zh: "qat（量化感知训练）：PyTorch + arch.py + 数据集；scheme=fx(默认)｜lsq(可学习步长，含 lsq_bits)，另有 weight_mode/weight_symmetric/lr_schedule/epochs/lr/patience；fx 产出 FX-QDQ ONNX；lsq 用数据集校准激活并产出全 INT8 QDQ：Conv/Linear 权重 = 学习步长 INT8，激活 = 校准 per-tensor INT8，产物 model_qat.onnx（另有 model_qat_wq.onnx 权重级、model_qat_float.onnx 浮点对照）。", en: "qat: FX quant-aware training (best INT8) or LSQ learnable-step weight training." },
      { zh: "export_gguf（llama.cpp GGUF）：HuggingFace 本地/远程 decoder；outtype=f16|bf16|q8_0|q4_0；用于 llama.cpp/ollama 等 GGUF 生态；转换需 llama.cpp 工具——离线/受限网络可预置目录并设置环境变量 LLAMA_CPP_PATH（本平台 GPU worker 即如此挂载）；f16 无需编译二进制，q4_0/q8_0 需 llama-quantize。", en: "export_gguf: HF LLM -> llama.cpp GGUF." },
      { zh: "llm_quantize（OpenVINO LLM INT4/INT8）：本地 HF 目录（可在模型页本地导入）；engine=auto/nncf/optimum（本地目录推荐 nncf）、bits=4|8、group_size=32..256（推荐 128）、sym；产物 OpenVINO IR + tokenizer，可直接在任务行💬对话或 /chat/completions 使用。", en: "llm_quantize: nncf INT4/INT8 OpenVINO IR for chat." },
      { zh: "④ 精度评估：classification 需要按类别 zip；yolo 需要含 data.yaml 的 zip；perplexity 可给 text_url/text_file，否则用内置提示词（LLM 轻量自检）。", en: "④ eval: classification/yolo/perplexity datasets." },
      { zh: "⑤ 性能基准：CPU 延迟/吞吐/峰值内存；LLM 任务自动走“文本生成感知”基准，评估页右侧紫色雷达对比原/压缩。", en: "⑤ benchmark incl. LLM-aware text generation; radar view." },
    ],
  },
  {
    titleZh: "✅ 已完成功能（总结）",
    titleEn: "Implemented (summary)",
    items: [
      { zh: "压缩方法：export_onnx / quantize_onnx(INT8 动态) / quantize_onnx_static / openvino_int8(NNCF) / prune(L1·L2·BN-Slim·GroupFisher) / sparsify(均匀·全局) / distill(软标签 logits KD) / QAT(FX 量化感知训练) / export_gguf(f16/bf16/q8_0/q4_0)", en: "Methods: onnx export, dynamic/static INT8, NNCF OpenVINO INT8, prune L1/L2/BN-Slim, sparsify, KD(logits), QAT(FX), GGUF." },
      { zh: "LLM 量化：llm_quantize engine=nncf（本地 HF 目录，INT4 group-wise，Qwen2.5-0.5B 实测 18.6s/999MB→339MB）", en: "llm_quantize engine=nncf verified (Qwen2.5-0.5B: 18.6s, 999->339MB INT4)." },
      { zh: "评估与基准：④ 分类/YOLO/Perplexity(内置提示词轻量回退)，⑤ CPU 基准含 LLM 文本生成感知路径，右栏紫色雷达", en: "Eval: classification/YOLO/perplexity (built-in prompts fallback); CPU benchmark incl. LLM-aware path; radar chart." },
      { zh: "任务中心：实时进度、状态筛选（排队/运行/完成/失败/取消）+ 关键字搜索（模型/方法/任务ID）；成功任务可查看「产物说明」（文件清单、生成参数、评估/基准摘要）并打包下载。", en: "Task center: live progress, status filter + keyword search; per-task artifact details (files, params, eval/benchmark summary) with zip download." },
      { zh: "模型/数据管理：上传、URL 拉取(HF→hf-mirror→ModelScope)、服务器本地目录导入(下拉+手填)；数据集上传支持 zip/tar/tgz 及 txt/csv/json/jsonl/xlsx/xls 纯文件", en: "Models: upload / pull / local import; Datasets: archives or plain txt/csv/json/jsonl/xlsx/xls." },
      { zh: "LLM 对话：chat/completions（CPU）与任务行💬对话测试（top_p/min_p/max tokens）", en: "LLM chat: CPU API + in-app dialog with top_p/min_p." },
      { zh: "GPU：inference.gpu fp16 CUDA 对话实测(RTX4060)；ONNX→CUDA EP；OV GPU 自动回落；COMPUTE_BACKEND 配置与 services 工厂", en: "GPU: fp16 CUDA chat verified; ONNX CUDA EP; service factory & COMPUTE_BACKEND ready." },
    ],
  },
  {
    titleZh: "📋 待完成 / 规划（含详细清单）",
    titleEn: "Planned (detailed)",
    items: [
      { zh: "MMRazor · Neural Architecture Search(NAS)：DARTS(ICLR2019) / DetNAS(NeurIPS2019) / SPOS(ECCV2020)。可加，但 NAS 需要 GPU 大规模搜索，当前 CPU-only 主机不可在线运行。", en: "MMRazor NAS (DARTS / DetNAS / SPOS): feasible only with GPU search budget." },
      { zh: "MMRazor · Pruning：Group Fisher / AutoSlim（稀疏训练+γ）/ DMCP（Taylor 一阶）均已实现为 prune 子算法（需数据集）；L1-norm 已等同现有 L1 选项。", en: "MMRazor pruning (AutoSlim / Group Fisher / DMCP) as new sub-algorithms." },
      { zh: "MMRazor · Knowledge Distillation（除 KD/logits 已实现外全部待加）：CWD(ICCV2021) / WSLD(ICLR2021) / ABLoss / BYOT / DAFL / DFAD / DKD / Factor Transfer / FitNets / KD(已实现) / OFD / RKD / ZSKT / FBKD。建议优先 FitNets 特征层。", en: "MMRazor KD methods: all listed except plain KD pending; FitNets first." },
      { zh: "Quantization 扩展：PTQ/QAT 已有基础（PyTorch FX）。LSQ 是否加：✅ 可加但需自实现——LSQ 需要可学习的步长(scale)与直通梯度(STE)自定义 module，属训练式量化，放入 QAT 的 weight/activation 模式选项（规划项）。", en: "LSQ: planned; needs learnable scale + STE training modules." },
      { zh: "PyTorch Supported observers（计划暴露为 QAT/PTQ 观测器选项，逐项）：FixedQParamsObserver · HistogramObserver · MinMaxObserver · MovingAverageMinMaxObserver · MovingAveragePerChannelMinMaxObserver · NoopObserver · ObserverBase · PerChannelMinMaxObserver · PlaceholderObserver · RecordingObserver · ReuseInputObserver · UniformQuantizationObserverBase", en: "Plan to expose these PyTorch observers as options (full list)." },
      { zh: "PyTorch Supported fakequants（计划暴露）：FakeQuantize · FakeQuantizeBase · FixedQParamsFakeQuantize · FusedMovingAvgObsFakeQuantize", en: "Plan to expose PyTorch fakequant modules (full list)." },
      { zh: "XTuner（LoRA/QLoRA 大模型微调）：llm_finetune CPU 微型原型已上线（内置字符 LM + 冻结基座训练 LoRA；数据可直接文本框 问题=>回答 或选平台数据集自动解析 .csv/.tsv/.xlsx 前两列，并支持 .jsonl/.json（messages 多轮对话、prompt+chosen 偏好样本按正例进入 SFT），产出 adapter/merged，并导出 preference_pairs.jsonl 供 DPO 预留）；7B 级需 GPU Worker 二期 (XTuner/DeepSpeed)。", en: "XTuner fine-tuning feasible on GPU workers only." },
      { zh: "LMDeploy（TurboMind 推理 · W4A16/AWQ · KV cache）：⚠️ 有条件——服务化可在 GPU profile 接入；W4A16/AWQ 需 NVIDIA CUDA kernel+校准数据。", en: "LMDeploy service wrapper on GPU; AWQ/GPTQ quantization needs CUDA + calibration." },
      { zh: "Deci 对标：AutoNAC/NAS 自动搜索⚠️不能在线化(多 GPU 预算)；INT4 部署✅已有等价(nncf INT4+chat)；Infery/ONNX/TensorRT 优化可在 GPU 上接 LMDeploy/TensorRT；按架构自动推荐参数✅可加。", en: "Deci: NAS online not feasible; INT4 serving already covered; TensorRT/LMDeploy upgrades on GPU; auto-presets feasible." },
      { zh: "其他：GPU 压缩 runner(compression_gpu 骨架)逐方法实现 · GPU SSE 流式/KV stateful · optimum stateful 默认化 · 多轮历史 · 其余数据集类型（多轮/偏好格式） · 多轮历史 · 文档渲染 docs/*.md", en: "Others: GPU compression runners, GPU SSE/KV, stateful optimum default, Excel->instruction sets for tuning, multi-turn history, docs/*.md rendering." },
    ],
  },
]

function Section({ section, index }: { section: DocSection; index: number }) {
  const { t } = useI18n()
  const Icon = section.icon ?? BookOpen
  return (
    <Card className="p-5">
      <div className="mb-3 flex items-center gap-2">
        <span className="flex size-7 items-center justify-center rounded-md bg-[var(--lb-primary-soft)] text-[var(--lb-primary)]">
          <Icon size={15} />
        </span>
        <h2 className="text-[15px] font-semibold text-[var(--lb-text)]">
          {t(`${index + 1}. ${section.titleZh}`, `${index + 1}. ${section.titleEn}`)}
        </h2>
      </div>
      <ul className="grid gap-2.5">
        {section.items.map((item, i) => (
          <li key={i} className="flex gap-2 text-[13.5px] leading-relaxed text-[var(--lb-text-2)]">
            <CheckCircle2 size={15} className="mt-0.5 shrink-0 text-[var(--lb-success)]" />
            <span>{t(item.zh, item.en)}</span>
          </li>
        ))}
      </ul>
    </Card>
  )
}

function DocsPage() {
  const { t } = useI18n()
  return (
    <div className="space-y-6">
      <div>
        <p className="lb-section">
          {t("平台内技术使用文档", "Platform technical docs")}
        </p>
        <h1 className="mt-1 flex items-center gap-2 text-2xl font-bold tracking-tight text-[var(--lb-text)]">
          <BookOpen size={22} className="text-[var(--lb-primary)]" />
          {t("技术使用文档", "Technical Docs")}
        </h1>
        <p className="lb-muted mt-1 text-sm">
          {t(
            "本页内置使用指南；仓库级规划文档见 docs/gpu-migration-plan.md 等。",
            "Built-in usage guide; repo-level plans live in docs/*.md.",
          )}
        </p>
      </div>

      <div className="rounded-md border border-[var(--lb-warn)]/40 bg-[var(--lb-warn-soft)] px-4 py-3 text-sm text-[var(--lb-warn)]">
        <div className="flex items-start gap-2">
          <Lightbulb size={16} className="mt-0.5 shrink-0" />
          <span>
            {t(
              "提示：若「新建压缩」点击无反应，通常是没有 Ready 模型；先到「模型管理」导入/拉取模型。",
              "Tip: if New Task does nothing, you likely have no Ready model — add one in Models first.",
            )}
          </span>
        </div>
      </div>

      {SECTIONS.map((section, index) => (
        <Section key={section.titleZh} section={section} index={index} />
      ))}
    </div>
  )
}
