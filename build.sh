#!/bin/bash
# Build everything in this directory.
#
# Two ggml sources are in play deliberately:
#   GGML_LLAMA  vk-base/build/bin   - ggml 0.19, as vendored by llama.cpp. Used by the original
#                                     harnesses so results stay comparable with llama.cpp runs.
#   GGML_STANDALONE  ggml-install   - ggml 0.23 from github.com/ggml-org/ggml, built and
#                                     installed on its own. Used to prove llama.cpp is not
#                                     required, and by the in-tree OpenVINO component.
#
# Prerequisites:
#   ov-master       built (ninja -C build)
#   genai-src       built against ov-master  (-DOpenVINO_DIR=<ov-master>/build)
#   ggml-install    cmake --install of ggml-org/ggml with -DGGML_VULKAN=ON
#   for use_ov_ggml: ov-master configured with -DENABLE_GGML_EMITTER=ON
set -e

OV=/home/ynimmaga/ov-master
GENAI=/home/ynimmaga/genai-src
GGML_LLAMA=/home/ynimmaga/vk-base
GGML_STANDALONE=/home/ynimmaga/ggml-install

OVLIB=$OV/bin/intel64/Release
GENAILIB=$GENAI/build/openvino_genai

# The emitter subclasses GraphEmitter and calls build_ggml_graph_from_gguf; both live in the
# frontend's PRIVATE source tree, hence the -I into src/ rather than include/.
OVINC="-I$OV/src/frontends/gguf/src -I$OV/src/frontends/gguf/include -I$OV/src/core/include \
       -I$OV/src/core/dev_api -I$OV/src/inference/include -I$OV/src/common/util/include \
       -I$OV/src/frontends/common/include"

# ---------------------------------------------------------------------------------------------
# 1. Standalone harnesses, against llama.cpp's ggml 0.19
# ---------------------------------------------------------------------------------------------
INC="$OVINC -I$GGML_LLAMA/ggml/include"
LIB="-L$OVLIB -lopenvino -lopenvino_gguf_frontend -L$GGML_LLAMA/build/bin -lggml -lggml-base"
RPATH="-Wl,-rpath,$OVLIB -Wl,-rpath,$GGML_LLAMA/build/bin"

for t in dump_graph emit_ggml run_ov; do
  [ -f $t.cpp ] && g++ -O2 -std=c++17 $INC $t.cpp -o $t $LIB $RPATH && echo "built $t"
done

# genai_ggml adds the GenAI tokenizer (encode/detokenize straight from the .gguf).
g++ -O2 -std=c++17 $INC -I$GENAI/src/cpp/include genai_ggml.cpp -o genai_ggml \
    $LIB -L$GENAILIB -lopenvino_genai $RPATH -Wl,-rpath,$GENAILIB
echo "built genai_ggml"

# ---------------------------------------------------------------------------------------------
# 2. Same source, against standalone ggml 0.23 -- proves no llama.cpp is needed
# ---------------------------------------------------------------------------------------------
if [ -d "$GGML_STANDALONE" ]; then
  g++ -O2 -std=c++17 $OVINC -I$GGML_STANDALONE/include -I$GENAI/src/cpp/include \
      genai_ggml.cpp -o genai_ggml_standalone \
      -L$OVLIB -lopenvino -lopenvino_gguf_frontend \
      -L$GGML_STANDALONE/lib -lggml -lggml-base -L$GENAILIB -lopenvino_genai \
      -Wl,-rpath,$OVLIB -Wl,-rpath,$GGML_STANDALONE/lib -Wl,-rpath,$GENAILIB
  echo "built genai_ggml_standalone"
fi

# ---------------------------------------------------------------------------------------------
# 3. Consumer of the IN-TREE OpenVINO component (needs -DENABLE_GGML_EMITTER=ON)
#    Note the include path: ONLY the public header. No ggml headers, no frontend internals.
# ---------------------------------------------------------------------------------------------
if [ -f "$OVLIB/libopenvino_ggml_emitter.so" ]; then
  g++ -O2 -std=c++17 \
      -I$OV/src/ggml_emitter/include -I$OV/src/core/include -I$OV/src/inference/include \
      -I$GENAI/src/cpp/include \
      use_ov_ggml.cpp -o use_ov_ggml \
      -L$OVLIB -lopenvino -lopenvino_ggml_emitter -L$GENAILIB -lopenvino_genai \
      -Wl,-rpath,$OVLIB -Wl,-rpath,$GENAILIB -Wl,-rpath,$GGML_STANDALONE/lib
  echo "built use_ov_ggml"
else
  echo "skip use_ov_ggml (configure ov-master with -DENABLE_GGML_EMITTER=ON)"
fi

# ---------------------------------------------------------------------------------------------
# 4. Architecture probes
# ---------------------------------------------------------------------------------------------
if [ -d archprobe ]; then
  ( cd archprobe
    g++ -O2 -std=c++17 $OVINC probe_arch.cpp -o probe_arch \
        -L$OVLIB -lopenvino -lopenvino_gguf_frontend -Wl,-rpath,$OVLIB
    echo "built archprobe/probe_arch"
    g++ -O2 -std=c++17 $OVINC -I$GGML_STANDALONE/include probe_emit.cpp -o probe_emit \
        -L$OVLIB -lopenvino -lopenvino_gguf_frontend \
        -L$GGML_STANDALONE/lib -lggml -lggml-base \
        -Wl,-rpath,$OVLIB -Wl,-rpath,$GGML_STANDALONE/lib
    echo "built archprobe/probe_emit" )
fi
