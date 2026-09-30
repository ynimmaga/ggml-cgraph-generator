# Architecture coverage probe

Answers "which of llama.cpp's architectures would OpenVINO's generic GGUF decoder builder
handle if we just added the name to `arch_registry.cpp`?" — **without downloading a single
model** and **without any llama.cpp C++ in the build**.

## How it works

`gguf-py` (published to PyPI as `gguf`, MIT) carries, for every architecture, the exact set of
tensors it has (`MODEL_TENSORS`) and each tensor's name template (`TENSOR_NAMES`). That is
precisely the input OV's `DecoderConfig` auto-detection consumes — it decides QK-norm,
fused-QKV, MoE routing, biases and post-norms from which layer-0 weights exist.

1. `gen_synthetic.py` expands those tables into one tiny (2-layer, 64-dim) `.gguf` per arch.
2. `probe_arch.cpp` calls `build_ggml_graph_from_gguf` on each and reports build/fail.
3. The driver buckets results by *fidelity* (below).

Only the generated data crosses the boundary; nothing from llama.cpp ships.

## Running

```sh
python3 -m venv venv && ./venv/bin/pip install numpy gguf
PYTHONPATH=<llama.cpp>/gguf-py ./venv/bin/python gen_synthetic.py synth
for f in synth/*.gguf; do
  OV_GGUF_ANY_ARCH=1 ./probe_arch "$(basename "$f" .gguf)" "$f"
done
```

`OV_GGUF_ANY_ARCH=1` is a test hook in `gguf_builder.cpp` that skips the accept-list, so an
unregistered architecture can be probed. Never set it in production.

## Reading the results — three buckets, not two

`shape_for()` cannot model every tensor (SSM state, RWKV time-mix, vision towers). Omitting
those **silently turns a hybrid model into a plain transformer**, and the builder then
"succeeds" on a topology the architecture does not have. `synth/skipped.tsv` records every
omission, and any arch with a non-empty list is **INCONCLUSIVE, not supported**.

`BENIGN_UNMODELLED` exempts tensors that don't change decoder topology (`attn_rot_embd`,
`nextn.*`). That set is calibrated against ground truth: `llama`, `minicpm`, `smollm3` and
`mistral3` are verified-working and all carry `attn_rot_embd`.

## Known limits — read before trusting a number

- **Building is necessary, not sufficient.** The probe proves a graph assembles. It cannot
  detect numerically wrong output. `granite` passed the probe and then *crashed* on a real
  model (missing `GGML_OP_SCALE` in the ggml emitter); other archs may pass and produce
  silently wrong text. Reference validation per architecture is still required.
- **`shape_for()` matches on substrings**, so exotic names can get a plausible-but-wrong
  shape instead of being flagged — e.g. `attn_kv_a_mqa` (deepseek2 MLA) matches the `attn_k`
  rule. Cross-check novel tensor vocabularies before trusting a candidate.
- **The synthetic model emits the union of optional tensors**, so it probes one variant of an
  architecture, typically the most-featured one.
- Architectures needing metadata the probe doesn't set (M-RoPE sections, chunked-attention
  parameters) may build and still be wrong.
