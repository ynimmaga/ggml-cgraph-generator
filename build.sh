#!/bin/bash
# Thin wrapper around the CMake build; see README.md "Build" for full CMake usage
# (including Windows). LLAMA_CPP_DIR / WHISPER_CPP_DIR are passed through as CMake
# cache variables; both fall back to CMake-side defaults if unset.
set -e

echo "start build......"

cmake -S "$(dirname "$0")" -B "$(dirname "$0")/build" \
    -DLLAMA_CPP_DIR="${LLAMA_CPP_DIR:-../llama.cpp}" \
    ${WHISPER_CPP_DIR:+-DWHISPER_CPP_DIR="${WHISPER_CPP_DIR}"}
cmake --build "$(dirname "$0")/build" --config Release -j
