#!/bin/bash
# make_dit_artifact.sh <out.json> <width> <height> "<prompt>" [-- extra sd-cli args]
# Captures stable-diffusion.cpp's Qwen-Image 2.1 transformer graph for this prompt length and image size as a cgraph
# artifact the GGML plugin can run (see openvino/src/plugins/ggml/src/dit_runner.cpp).
#
# One-time setup:
#   git clone https://github.com/leejet/stable-diffusion.cpp && cd stable-diffusion.cpp && git submodule update --init
#   git apply <this dir>/capture-cgraph.patch
#   cmake -S . -B build -DSD_VULKAN=OFF -DCMAKE_CXX_FLAGS="-I<ggml-cgraph-generator dir>" && cmake --build build -j
# Needs SD_CLI (build/bin/sd-cli), DIT (Qwen-Image-2.1 .gguf), VAE (.safetensors) and LLM (Qwen3-VL-8B .gguf) in the environment.
set -e
out=$1; w=$2; h=$3; prompt=$4; shift 4
OV_CGRAPH_OUT=$(realpath -m "$out") "${SD_CLI:?}" --diffusion-model "${DIT:?}" --vae "${VAE:?}" --llm "${LLM:?}" \
  -p "$prompt" --cfg-scale 1.0 --sampling-method euler --steps 1 -W "$w" -H "$h" -o /dev/null "$@" 2>&1 | grep "ov-cgraph"
# The pipeline needs one artifact per (text tokens, image tokens): the text length is the prompt's token count after
# the system prefix. A mismatch is reported by the runner with the lengths it needs.
