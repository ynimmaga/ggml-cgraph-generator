#!/bin/bash
# Builds both dumpers against a llama.cpp checkout.
#
# Set LLAMA_CPP_DIR to a llama.cpp source tree that has been built (so its ggml/llama/mtmd
# libraries exist under LLAMA_CPP_DIR/build/bin). Both dumpers link llama.cpp ONLY -- this tool
# is the one place in the broader project that llama.cpp appears at all; everything downstream
# of its JSON output runs without it.
set -e
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-../llama.cpp}"

INC="-I. -I$LLAMA_CPP_DIR/include -I$LLAMA_CPP_DIR/ggml/include -I$LLAMA_CPP_DIR/tools/mtmd"
LIB="-L$LLAMA_CPP_DIR/build/bin -lllama -lggml -lggml-base"
RPATH="-Wl,-rpath,$LLAMA_CPP_DIR/build/bin"

g++ -O2 -std=c++17 $INC dump_cgraph.cpp -o dump_cgraph $LIB $RPATH
echo "built dump_cgraph"

# dump_vision additionally needs libmtmd (llama.cpp's multimodal support library,
# built when llama.cpp is configured with LLAMA_BUILD_TOOLS=ON, its default).
g++ -O2 -std=c++17 $INC dump_vision.cpp -o dump_vision $LIB -lmtmd $RPATH
echo "built dump_vision"
