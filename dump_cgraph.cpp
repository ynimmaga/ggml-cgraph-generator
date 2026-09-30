// Offline tool: dump llama.cpp's cgraph for a model to a declarative JSON artifact that
// OpenVINO can replay, so the OV side needs neither llama.cpp nor its own architecture builder.
//
// WHY THIS SHAPE
// OpenVINO's GGUF frontend already consumes a flat, topologically-ordered list of nodes in the
// GGML op vocabulary (GgufOp: op_type, name, input_names, shapes, strides, typed attributes).
// That structure is pointer-free, so it serialises directly. A ggml_cgraph is NOT -- it is
// ggml_tensor* with src[] and data pointers -- and ggml removed ggml_graph_export, so there is
// nothing off the shelf. This tool writes the GgufOp-shaped form.
//
// HOW IT HOOKS IN
// Entirely through llama.cpp's PUBLIC API: llama_context_params::cb_eval. The scheduler calls it
// once per node with ask=true before computing, handing over the tensor with full metadata (op,
// srcs, ne, nb, type, op_params). We record and return false, so no node is split out for
// isolated evaluation and the run stays fast. No llama.cpp patch, no custom ggml backend.
//
// SCOPE
// Captures TOPOLOGY, not weights -- weights stay in the .gguf and are resolved by name at load
// time. The artifact is therefore small. It is also SHAPE-STATIC: the graph is dumped for one
// concrete (n_tokens, n_kv), which is baked into every node's dimensions. That is the central
// trade of this approach versus OpenVINO's own builder.
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <algorithm>

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include "cgraph_capture.hpp"

using namespace cgraph;

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: dump_cgraph <model.gguf> [out.json] [n_kv]\n");
        return 1;
    }
    const std::string path = argv[1];
    const std::string out = argc > 2 ? argv[2] : "cgraph.json";
    const int n_kv = argc > 3 ? atoi(argv[3]) : 256;
    // Encoder models (BERT-family embedding and reranking) build a different graph: non-causal
    // attention and a pooling head instead of an LM head. llama.cpp only emits that graph when
    // the context is created in embedding mode, so it has to be requested up front.
    const bool embd_mode = getenv("DUMP_EMBEDDINGS") != nullptr;
    // VLM decoders: a batch built with llama_batch_init(n, n_embd, ...) feeds embeddings
    // (llama_batch.embd) in place of token ids, so the graph gets an "inp_embd" input instead
    // of GET_ROWS(token_embd, inp_tokens). This is what the vision projector's output plugs
    // into; dumped separately because it is a structurally different graph, not a flag on the
    // normal one.
    const bool vlm_embd_input = getenv("DUMP_VLM_DECODER") != nullptr;
    // Batched-prefill probe: dump the plain token-id decoder for N tokens in one forward pass
    // instead of 1, so a caller can feed a whole prompt in one graph invocation rather than
    // stepping it. Every node's ne[]/nb[] is then baked in for exactly N, same as n_kv.
    const int n_tokens = getenv("N_TOKENS") ? atoi(getenv("N_TOKENS")) : 1;
    // Continuous-batching decode: llama_batch_get_one only requests logits for the LAST row,
    // so the dumped graph's lm_head only ever produces one row. ALL_LOGITS requests every row's
    // logits, so N independent sequences occupying the N rows can each sample their own next
    // token from one forward pass.
    const bool all_logits = getenv("ALL_LOGITS") != nullptr;

    llama_backend_init();
    ggml_backend_load_all();

    Capture cap;

    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;  // CPU: we want the topology, not performance
    llama_model* model = llama_model_load_from_file(path.c_str(), mp);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", path.c_str());
        return 2;
    }

    auto cp = llama_context_default_params();
    cp.n_ctx = n_kv;
    cp.n_batch = n_kv;
    cp.cb_eval = eval_cb;
    cp.cb_eval_user_data = &cap;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;  // match the OV builder's topology
    if (embd_mode) {
        cp.embeddings = true;
        cp.pooling_type = LLAMA_POOLING_TYPE_MEAN;
        // Encoders attend bidirectionally, so flash attention with a causal mask is wrong here.
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 3;
    }

    // One decode is enough: it builds and schedules the full graph, which is all cb_eval needs.
    const llama_vocab* vocab = llama_model_get_vocab(model);
    llama_token bos = llama_vocab_bos(vocab);
    if (bos < 0) {
        bos = 1;
    }
    // A VLM decoder graph needs SOME embedding values to shape n_embd correctly; the actual
    // values do not matter, we are capturing topology, not running inference. llama_batch_init
    // already allocates b.embd -- fill it in place, do not repoint it at another buffer, or
    // llama_batch_free() will free memory it never allocated (this crashed with "double free or
    // corruption" the first time, from freeing a std::vector's internal buffer via C free()).
    llama_batch b;
    if (vlm_embd_input) {
        const int n_embd = llama_model_n_embd(model);
        b = llama_batch_init(1, n_embd, 1);
        b.n_tokens = 1;
        std::fill(b.embd, b.embd + n_embd, 0.0f);
        b.pos[0] = 0;
        b.n_seq_id[0] = 1;
        b.seq_id[0][0] = 0;
        b.logits[0] = 1;
    } else if (all_logits) {
        // Manual batch: llama_batch_get_one only flags the last row for logits, so build one
        // by hand and set every row's flag. Token/position values don't matter for topology.
        b = llama_batch_init(n_tokens, 0, 1);
        b.n_tokens = n_tokens;
        for (int i = 0; i < n_tokens; i++) {
            b.token[i] = bos;
            b.pos[i] = i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = 1;
        }
    } else {
        // Token VALUES don't matter for topology, only the count -- llama_batch_get_one lays
        // n_tokens sequential tokens into one sequence starting at whatever position the
        // context is currently at (0, here).
        static std::vector<llama_token> toks(n_tokens, bos);
        b = llama_batch_get_one(toks.data(), n_tokens);
    }
    if (embd_mode) {
        // An encoder has no autoregressive step: llama_encode builds and runs the whole graph.
        if (llama_encode(ctx, b) != 0) {
            fprintf(stderr, "encode failed\n");
            return 4;
        }
    } else if (llama_decode(ctx, b) != 0) {
        fprintf(stderr, "decode failed\n");
        return 4;
    }

    char arch_buf[128] = {0};
    llama_model_meta_val_str(model, "general.architecture", arch_buf, sizeof(arch_buf));

    write_json(cap, out, path, arch_buf, vlm_embd_input ? 1 : n_tokens, n_kv);
    if (vlm_embd_input || all_logits) {
        llama_batch_free(b);
    }

    std::map<std::string, int> hist;
    for (const auto& n : cap.nodes) {
        hist[n.op]++;
    }
    printf("mode         : %s\n", embd_mode ? "encoder (embeddings)" :
                                  vlm_embd_input ? "decoder (embedding-input, for VLM)" : "decoder");
    printf("architecture : %s\n", arch_buf);
    printf("nodes        : %zu\n", cap.nodes.size());
    printf("weight leaves: %zu\n", cap.weights.size());
    size_t n_in = 0, n_cache = 0;
    for (const auto& kv : cap.leaves) {
        if (kv.second.is_input) n_in++;
        else if (kv.second.has_buffer && !cap.weights.count(kv.first)) n_cache++;
    }
    printf("graph inputs : %zu\n", n_in);
    printf("kv caches    : %zu\n", n_cache);
    printf("total leaves : %zu\n", cap.leaves.size());
    printf("op types     : %zu\n", hist.size());
    for (const auto& h : hist) {
        printf("  %-28s %4d\n", h.first.c_str(), h.second);
    }
    printf("wrote %s\n", out.c_str());

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
