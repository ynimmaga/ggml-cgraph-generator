# ggml-cgraph-generator

Dumps a llama.cpp model's computation graph (a `ggml_cgraph`) to a portable,
declarative JSON artifact, offline. The resulting file describes the graph's ops,
shapes, and data flow well enough that another runtime can rebuild and execute it
**without linking llama.cpp at all** — llama.cpp is only ever used here, at dump time,
as a build-time tool.

This exists because a `ggml_cgraph` is not something you can serialize off the shelf:
it's a graph of `ggml_tensor*` with raw `src[]`/`data` pointers, and ggml itself
dropped `ggml_graph_export`. This tool captures the graph through llama.cpp's public
`cb_eval` callback (the same mechanism llama.cpp itself uses for tensor-level
debugging) and writes out everything a consumer needs to rebuild it: op type, every
input's identity, shape (`ne`)/stride (`nb`), element type, and the raw per-op
parameter blob (`op_params`).

One example consumer — an OpenVINO component that replays this format and executes it
on a ggml backend (Vulkan/CPU) with no llama.cpp present at runtime — is at
[`ynimmaga/openvino@ggml_cgraph_loader`](https://github.com/ynimmaga/openvino/tree/ggml_cgraph_loader)
(see `src/ggml_cgraph_loader/src/cgraph_loader.cpp`), driven from OpenVINO GenAI by
[`ynimmaga/openvino.genai@ggml_cgraph_loader`](https://github.com/ynimmaga/openvino.genai/tree/ggml_cgraph_loader).
This repo doesn't depend on either; the artifact format is the only contract between them.
See [END_TO_END.md](END_TO_END.md) for the whole flow: dump → build → run.

## Build

Needs a built llama.cpp checkout (with `LLAMA_BUILD_TOOLS=ON`, llama.cpp's default, if
you want `dump_vision` too — that needs `libmtmd`).

```sh
LLAMA_CPP_DIR=/path/to/llama.cpp ./build.sh
```

Produces `dump_cgraph` and `dump_vision`.

## `dump_cgraph` — text decoder graphs

```sh
./dump_cgraph <model.gguf> [out.json] [n_kv]
```

- `model.gguf` — any llama.cpp-compatible GGUF model.
- `out.json` — output path (default `cgraph.json`).
- `n_kv` — KV cache size to build the graph for (default 256). Every tensor shape
  touching the cache is baked in at this size — see **Shape-static, read before
  relying on this** below.

Four environment variables switch which graph gets dumped:

| Variable | Effect |
|---|---|
| `N_TOKENS=<n>` | Dump a graph accepting `n` tokens in one forward pass instead of 1 (default). Every node's shape is baked in for exactly `n` — useful for batched prefill, or for building a fixed-size multi-sequence decode graph (see `ALL_LOGITS` below). |
| `ALL_LOGITS=1` | By default only the *last* token's logits are computed (llama.cpp's own default when decoding a batch). This forces every row's logits to be computed, so with `N_TOKENS=n` set too, the graph produces independent next-token predictions for all `n` rows — e.g. for `n` concurrently-decoding, unrelated sequences sharing one forward pass, not just `n` prefill positions of one sequence. |
| `DUMP_EMBEDDINGS=1` | Dump an embedding/encoder graph instead of a causal decoder: non-causal (bidirectional) attention and a pooling head, as llama.cpp builds when the context is created in embedding mode. For BERT-family models (embedding, reranking). |
| `DUMP_VLM_DECODER=1` | Dump a decoder that takes embeddings (`llama_batch.embd`) as input instead of token ids — i.e. `GET_ROWS(token_embd, ids)` is replaced by a graph input. This is what a vision encoder's projected output, or any other externally-supplied embedding, plugs into. |

Examples:

```sh
# Ordinary single-token decoder, the common case
./dump_cgraph llama.gguf decoder.json 256

# Batched prefill: one forward pass for a 4-token prompt chunk
N_TOKENS=4 ./dump_cgraph llama.gguf prefill4.json 256

# 4-way concurrent decode: 4 independent sequences, each producing its own next-token
# logits from one shared forward pass
N_TOKENS=4 ALL_LOGITS=1 ./dump_cgraph llama.gguf decode_batch4.json 256

# BERT-family embedding model
DUMP_EMBEDDINGS=1 ./dump_cgraph bge-small.gguf embed.json 512

# VLM text decoder, embedding-input
DUMP_VLM_DECODER=1 ./dump_cgraph llava.gguf vlm_decoder.json 256
```

`DUMP_EMBEDDINGS` and `DUMP_VLM_DECODER` are mutually exclusive with each other and
with `N_TOKENS`/`ALL_LOGITS` (both of those apply to the plain causal-decoder path).

Console output on success is a topology summary: node count, weight-leaf count, graph
input count, KV cache tensor count, and an op histogram — useful for sanity-checking
that a new architecture or a new flag combination produced the shape you expected
before trusting the file.

## `dump_vision` — VLM vision tower graphs

```sh
./dump_vision <mmproj.gguf> <text-model.gguf> [out.json] [px]
```

Dumps the vision encoder + projector graph from a VLM's mmproj file, via
`mtmd_context_params::cb_eval` — the same public hook mechanism, just on llama.cpp's
multimodal path instead of the plain text path. `px` is the square image resolution to
build the graph for (default 512) — this determines the patch count and therefore
every shape in the dump, same caveat as `n_kv` above.

The image fed through at dump time is a synthetic mid-grey placeholder — only the
resolution matters for capturing topology, not pixel content. A real consumer needs to
run its own image preprocessing (resize + per-channel normalize) matching the
values in the mmproj's own `clip.vision.image_size`/`image_mean`/`image_std` GGUF
metadata; this tool does not do that for you, it only captures the compute graph that
runs *after* preprocessing.

```sh
./dump_vision mmproj-model.gguf text-model.gguf vision.json 512
```

## Output format (`"ov-cgraph-v1"`)

Top-level JSON object:

```jsonc
{
  "format": "ov-cgraph-v1",
  "source": "llama.cpp cb_eval",
  "model": "...", "architecture": "...",
  "n_tokens": 4, "n_kv": 256,           // baked-in shape parameters, see below
  "weights": { "<id>": {"type": "...", "gguf_name": "..."} },
  "leaves":  [ {"id", "name", "type", "ne", "nb", "is_input", "has_buffer"} ],
  "nodes":   [ {"op", "id", "name", "type", "ne", "nb",
                "inputs": ["<id>", ...], "input_ne", "input_type",
                "op_params": [...], "view_src", "view_offs"} ]
}
```

Two design choices a consumer needs to know about:

- **Node identity is by `id`, not `name`.** ggml reuses tensor *names* across a
  transform chain — the `mul_mat` producing `Qcur-0` and the `rope` consuming it are
  *both* named `Qcur-0` in the live graph. A name-keyed rebuild silently drops nodes.
  `id` is a synthetic, always-unique identifier this tool assigns by interning tensor
  *pointers*, not names; `name` is kept alongside purely for human readability.
- **Graph inputs are identified by which op consumes them, not by name.** llama.cpp
  leaves most inputs auto-named (`leaf_5`, `leaf_11`, ...). A consumer recovers each
  input's *role* from context: the tensor feeding `GGML_OP_ROPE`'s second input is the
  position stream; the one feeding `GGML_OP_SET_ROWS`'s second input is a KV
  cache row index; the one feeding `GGML_OP_FLASH_ATTN_EXT`'s fourth input is the
  attention mask; and so on. `leaves[].is_input` marks which leaves are genuine
  runtime inputs (as opposed to weights or KV caches).
- `weights` maps a node `id` to the literal tensor name inside the original `.gguf` —
  a consumer resolves the weight by re-reading that file, rather than this tool
  embedding weight *values* into the JSON (which would make every dump as large as the
  model itself). The artifact captures topology only.

## Shape-static — read before relying on this

Every dump is valid for exactly the `(n_tokens, n_kv)` — or, for `dump_vision`, `px` —
it was built with. There is no symbolic/dynamic shape support: a consumer that needs a
different prompt length, context size, or image resolution needs a fresh dump at that
size, not a reshape of an existing one. This is a fundamental property of a ggml graph
(shapes are concrete at construction time), not a limitation of this tool specifically.

## Known bug

Dumping the plain decoder with `N_TOKENS >= 16` currently produces weight references
in the output like `Vulkan0#blk.2.attn_norm.weight#0`, which don't correspond to any
real tensor in the source `.gguf`. This happens because ggml's backend scheduler
inserts a repacked-weight copy node once Vulkan becomes a candidate backend at larger
batch sizes, and this tool's weight-vs-input classification (in `cgraph_capture.hpp`)
currently misclassifies that copy as an ordinary `.gguf` weight leaf. `N_TOKENS` values
of 1, 4, and 8 are unaffected and have been used extensively; treat 16+ as unreliable
until this is fixed.

## How it hooks into llama.cpp

Entirely through public API — `llama_context_params::cb_eval` /
`mtmd_context_params::cb_eval`. The scheduler calls the callback once per node with
`ask=true` before computing it, handing over the tensor with full metadata; this tool
records it and returns `false`, so no node is split out for isolated evaluation and the
dump costs about as much as one ordinary decode. No llama.cpp source changes, no custom
ggml backend.
