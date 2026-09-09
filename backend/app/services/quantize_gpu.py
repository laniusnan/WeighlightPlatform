"""GPU quantization/conversion service - SKELETON.

TODO(gpu): implement each method and remove it from the not-implemented set:
  * llm_quantize   -> NNCF/optimum-intel on OpenVINO GPU plugin, or torch CUDA
                      calibration before weight compression.
  * openvino_int8  -> NNCF calibration on device "GPU".
  * quantize_onnx / quantize_onnx_static -> onnxruntime CUDAExecutionProvider
                      dynamic/static quantization.
  * qat            -> FX QAT fine-tuning on CUDA.
  * export_onnx / export_gguf -> keep on CPU for now (IO bound) unless fp16
                      CUDA tracing is needed for very large models.
"""
from __future__ import annotations

import logging
from typing import Any

from app.services.base import QUANTIZE_METHODS, NotImplementedBackendError
from app.services.compression_gpu import GpuCompressionRunner

logger = logging.getLogger("app.services.quantize_gpu")


class QuantizeGpuService(GpuCompressionRunner):
    name = "quantize.gpu"
    supported_methods = QUANTIZE_METHODS

    def quantize(self, task_id: Any) -> None:
        raise NotImplementedBackendError(
            "GPU quantization service is a skeleton; set COMPUTE_BACKEND=cpu."
        )
