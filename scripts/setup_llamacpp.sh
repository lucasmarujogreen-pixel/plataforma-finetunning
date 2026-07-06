#!/usr/bin/env bash
# Clone and build llama.cpp for GGUF conversion and quantization.
set -euo pipefail

LLAMACPP_DIR="third_party/llama.cpp"

if [ ! -d "$LLAMACPP_DIR" ]; then
    echo ">> Cloning llama.cpp"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMACPP_DIR"
else
    echo ">> llama.cpp already cloned"
fi

if command -v cmake >/dev/null 2>&1; then
    CMAKE="cmake"
else
    echo ">> cmake not found, using PyPI wheel via uv"
    CMAKE="uv run --with cmake cmake"
fi

echo ">> Building llama-quantize (CPU build)"
$CMAKE -S "$LLAMACPP_DIR" -B "$LLAMACPP_DIR/build" -DGGML_CUDA=OFF -DLLAMA_CURL=OFF \
    -DCMAKE_BUILD_TYPE=Release >/dev/null
$CMAKE --build "$LLAMACPP_DIR/build" --target llama-quantize -j "$(nproc)" >/dev/null

echo ">> Done:"
echo "   convert: $LLAMACPP_DIR/convert_hf_to_gguf.py"
echo "   quantize: $LLAMACPP_DIR/build/bin/llama-quantize"
