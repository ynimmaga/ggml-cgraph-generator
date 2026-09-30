# ov2ggml

Tooling built alongside two feature branches:

- [`ynimmaga/openvino@gguf-pluggable-emitter`](https://github.com/ynimmaga/openvino/tree/gguf-pluggable-emitter)
  — a pluggable `GraphEmitter` in OpenVINO's GGUF frontend, an experimental ggml/Vulkan execution
  backend (`openvino_ggml_emitter`), and a loader that replays a dumped llama.cpp `ggml_cgraph` as
  an OpenVINO-executed graph with no llama.cpp at runtime.
- [`ynimmaga/openvino.genai@ggmlpipeline`](https://github.com/ynimmaga/openvino.genai/tree/ggmlpipeline)
  — `GgmlPipeline` / `GgmlVLMPipeline`: GenAI's tokenizer, chat templates, sampler and streaming
  driving that ggml-executed graph instead of `ov::InferRequest`.

Everything here is the harness/dumper/proof-of-concept layer connecting the two: no production
code, no test suite, no CI. It is a lab notebook, not a library.

## What's in here

| Path | What |
|---|---|
| `cgraphdump/` | Offline dumpers. `dump_cgraph.cpp` captures a llama.cpp decoder graph via the public `llama_context_params::cb_eval` hook into a JSON artifact (`"ov-cgraph-v1"`) OpenVINO's `cgraph_loader.cpp` replays. `dump_vision.cpp` does the same for a VLM's vision tower via `mtmd_context_params::cb_eval`. `cgraph_capture.hpp` is the shared capture/serialisation logic both use. |
| `archprobe/` | `gen_synthetic.py` builds one tiny synthetic `.gguf` per llama.cpp architecture from `gguf-py`'s own tensor-name tables (no model downloads); `probe_arch`/`probe_emit` check whether OpenVINO's native GGUF builder and the ggml emitter, respectively, can handle each one. `results.tsv`/`joined.tsv`/`*_candidates.txt` are the last sweep's output. |
| `ggml_emitter.hpp` | An early, standalone copy of the `GraphEmitter` subclass that now lives properly in the OpenVINO branch (`src/ggml_emitter/src/emitter.cpp`) — kept for history, superseded there. |
| `genai_ggml.cpp`, `emit_ggml.cpp`, `use_ov_ggml.cpp`, `use_ov_cgraph.cpp` | Progressively cleaner harnesses for driving the emitter/loader graphs directly, without GenAI. |
| `vlm_e2e.cpp`, `vlm_pipeline_demo.cpp` | Raw VLM wiring test, then the same thing through the real `GgmlVLMPipeline`. |
| `bucketed_prefill.cpp`, `batched_prefill_test.cpp`, `partial_batch_test.cpp`, `prefill_benchmark.cpp`, `bucketed_pipeline_demo*.cpp` | Batched-prefill correctness and performance investigation: padding-to-scratch-slot technique, bucket selection, isolated vs. pipeline-level timing. |
| `continuous_batch_test.cpp` | Proves the two properties that define continuous batching (as opposed to static batching) on a ggml graph with no paging support: independent, different-length sequences co-batched in one forward pass with zero cross-contamination, and a finished sequence's slot reused by a brand-new one mid-stream while other slots keep running. |
| `mem_check.cpp` | Direct RSS measurement used to test (and refute) a memory-duplication hypothesis for an unresolved decode-time performance regression. |
| `ov_fallback.cpp`, `ov2ggml.cpp`, `dump_graph.cpp`, `run_ov.cpp` | Earlier exploratory tools from before the emitter/loader design settled (whole-model Vulkan fallback, a from-scratch GGUF→ggml translator, op-set dumping via the public frontend API). Kept for the trail, not recommended as a starting point. |
| `build.sh` | Compiles the harnesses. Hardcodes `-I`/`-L` paths into `ov-master`, `genai-src`/`genai-ggml`, and a standalone `ggml-install` tree — edit the paths at the top before use elsewhere. |

## Known limits, honestly

- Every dumped cgraph artifact is shape-static: valid for exactly the `(n_tokens, n_kv)` it was
  dumped at.
- Dumping at `N_TOKENS >= 16` currently produces backend-tagged weight references
  (`Vulkan0#...#0`) the loader can't resolve — a real bug, not yet fixed.
- Batched prefill is faster in isolation and at a single decode step, but an undiagnosed
  per-decode-step cost in the multi-bucket pipeline configuration erases that win somewhere
  between 4 and 12 generated tokens on the hardware this was built against.
- `continuous_batch_test.cpp` proves the core mechanism only: no real scheduler, no sampler
  integration, no prefill-on-admit, no pipeline API. See its own header comment.
