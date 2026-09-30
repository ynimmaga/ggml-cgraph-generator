#!/bin/bash
# Reproduce the model runs.
#
# Runtime note: libggml.so lists libggml-cpu / libggml-vulkan as NEEDED, but the ggml install
# does not put its own lib dir on their RUNPATH, so LD_LIBRARY_PATH is required. Backends are
# then picked up via ggml_backend_load_all() and selected by device type.
#
# Environment switches understood by the harnesses:
#   USE_CPU=1            force the CPU backend (default: GPU -> IGPU -> CPU)
#   STUB_FA=1            bypass flash attention (diagnostic; output becomes meaningless)
#   STUB_KV=1            feed q as k and v (diagnostic)
#   ROPE_MODE=N          force a ggml RoPE mode instead of deriving it from op_case
#   OV_GGUF_ANY_ARCH=1   skip the arch_registry accept-list (test hook, OpenVINO side)
set -e
M=/home/ynimmaga/models
export LD_LIBRARY_PATH=/home/ynimmaga/ggml-install/lib

PROMPT="The capital of France is"
N=14
NKV=128

echo "=== in-tree OpenVINO component (libopenvino_ggml_emitter), Vulkan ==="
for m in Llama-3.2-1B-Instruct-Q4_0 Qwen3-0.6B-Q4_0 granite-4.0-1b-Q4_0; do
  printf "%-30s " "$m"
  ./use_ov_ggml "$M/$m.gguf" "$PROMPT" $N $NKV 2>/dev/null | grep "^output:" | cut -c9-
done

echo
echo "=== standalone harness, CPU vs Vulkan (numerical agreement check) ==="
for m in Llama-3.2-1B-Instruct-Q4_0 Qwen3-0.6B-Q4_0; do
  for be in "" "USE_CPU=1"; do
    printf "%-30s %-7s " "$m" "${be:+CPU}${be:-Vulkan}"
    env $be ./genai_ggml "$M/$m.gguf" "$PROMPT" $N $NKV 2>/dev/null | grep "^output:" | cut -c9-
  done
done

echo
echo "=== standalone ggml 0.23 (no llama.cpp anywhere in the process) ==="
./genai_ggml_standalone "$M/Llama-3.2-1B-Instruct-Q4_0.gguf" "$PROMPT" $N $NKV 2>/dev/null \
  | grep -E "^(backend|output):"

echo
echo "=== architecture sweep (synthetic models, no downloads) ==="
cd archprobe
PYTHONPATH=/home/ynimmaga/vk-base/gguf-py /home/ynimmaga/archvenv/bin/python \
  gen_synthetic.py synth >/dev/null
: > results.tsv
for f in synth/*.gguf; do
  a=$(basename "$f" .gguf)
  o=$(OV_GGUF_ANY_ARCH=1 timeout 60 ./probe_arch "$a" "$f" 2>/dev/null) || true
  [ -z "$o" ] && o=$(printf "CRASH\t%s\t-" "$a")
  echo "$o" >> results.tsv
done
join -t$'\t' -1 2 -2 1 <(sort -k2,2 results.tsv) <(cut -f1,2 synth/skipped.tsv | sort -k1,1) \
  > joined.tsv
echo "faithful & building : $(awk -F'\t' '$4==0 && $2=="OK"' joined.tsv | wc -l)"
echo "inconclusive        : $(awk -F'\t' '$4>0  && $2=="OK"' joined.tsv | wc -l)"
echo "failed              : $(awk -F'\t' '$2!="OK"' joined.tsv | wc -l)"
