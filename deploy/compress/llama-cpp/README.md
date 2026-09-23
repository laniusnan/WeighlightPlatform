# llama.cpp 离线转换工具（可选）

GGUF 导出（`--method=quantize --algo=gguf` / `--method=export --algo=gguf`）由平台
调用 llama.cpp 的 `convert_hf_to_gguf.py` 完成。默认行为是**联网自动获取**（GitHub →
ghfast/gh-proxy/gitclone 镜像 → codeload tarball → jsDelivr CDN），失败会报错。

若目标机器**不能访问外网**，把 llama.cpp 仓库的转换脚本放进本目录即可：

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp
cp -r /tmp/llama.cpp/convert_hf_to_gguf.py /tmp/llama.cpp/gguf-py deploy/compress/llama-cpp/
# 然后重新构建镜像
```

镜像构建时会把本目录整体复制到 `/opt/llama.cpp` 并设置：

```
ENV LLAMA_CPP_PATH=/opt/llama.cpp
```

`_fetch_llama_cpp_tools()` 会优先使用 `LLAMA_CPP_PATH`（只要其中存在
`convert_hf_to_gguf.py`），从而**完全离线**完成 GGUF 转换。

> 目录为空 / 只放本 README 时不影响镜像构建，GGUF 导出将退回联网获取逻辑；
> 其它压缩方法（量化、剪枝、蒸馏、微调、ONNX/OpenVINO 导出）与网络无关。
