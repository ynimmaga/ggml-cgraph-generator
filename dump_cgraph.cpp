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

namespace {

// Keep only the token embedding lookup and the leaves it reads (DUMP_TOKEN_EMBD).
bool keep_token_embd(Capture& cap) {
    for (size_t i = 0; i < cap.nodes.size(); i++) {
        const Node& n = cap.nodes[i];
        auto w = n.inputs.empty() ? cap.weight_names.end() : cap.weight_names.find(n.inputs[0]);
        if (n.op != "GGML_OP_GET_ROWS" || w == cap.weight_names.end() || w->second != "token_embd.weight") {
            continue;
        }
        Node lookup = n;
        cap.nodes = {lookup};
        std::map<std::string, Capture::Leaf> leaves;
        std::map<std::string, std::string> weights, weight_names;
        for (const auto& id : lookup.inputs) {
            if (auto l = cap.leaves.find(id); l != cap.leaves.end()) leaves.insert(*l);
            if (auto t = cap.weights.find(id); t != cap.weights.end()) weights.insert(*t);
            if (auto g = cap.weight_names.find(id); g != cap.weight_names.end()) weight_names.insert(*g);
        }
        cap.leaves = std::move(leaves);
        cap.weights = std::move(weights);
        cap.weight_names = std::move(weight_names);
        return true;
    }
    return false;
}

}  // namespace

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
    // Token embedding lookup only: GET_ROWS(token_embd.weight, inp_tokens) for one token, cut out
    // of the decoder graph. A VLM runtime needs it as a model of its own (text embeddings) to
    // embed prompt tokens next to image features; dequantization stays in ggml, so it works for
    // any weight type.
    const bool token_embd_only = getenv("DUMP_TOKEN_EMBD") != nullptr;
    // Quantized KV cache: KV_TYPE=q8_0 / q4_0 / ... (any ggml type name; default f16). The cache
    // leaves are then recorded with that type, SET_ROWS quantizes on store and FLASH_ATTN_EXT
    // reads the quantized K/V -- nothing changes on the replay side.
    ggml_type kv_type = GGML_TYPE_F16;
    if (const char* kt = getenv("KV_TYPE")) {
        int i = 0;
        for (; i < GGML_TYPE_COUNT; i++) {
            const char* name = ggml_type_name(static_cast<ggml_type>(i));
            if (name && strcmp(name, kt) == 0) break;
        }
        if (i == GGML_TYPE_COUNT) {
            fprintf(stderr, "unknown KV_TYPE '%s' (expected a ggml type name, e.g. q8_0)\n", kt);
            return 1;
        }
        kv_type = static_cast<ggml_type>(i);
    }

    llama_backend_init();
    ggml_backend_load_all();

    Capture cap;
    // llama.cpp fills these rotation matrices once (quantized-KV-cache Hadamard rotation of Q/K/V);
    // they are constants of the model + cache type, not per-step inputs.
    cap.is_constant_input = [](const ggml_tensor* t) {
        const std::string n = ggml_get_name(t);
        return n.rfind("attn_inp_", 0) == 0 && n.size() > 4 && n.compare(n.size() - 4, 4, "_rot") == 0;
    };

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
    // REF_OUT=<file>: also write llama.cpp's value of every node (see write_ref). Use NO_FILL=1 with it:
    // the captured decode is then the first token at position 0 of an empty cache, which a replay
    // can reproduce exactly.
    const char* ref_out = getenv("REF_OUT");
    const bool no_fill = getenv("NO_FILL") != nullptr;
    cp.cb_eval = ref_out ? eval_cb_values : eval_cb;
    cp.cb_eval_user_data = &cap;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;  // match the OV builder's topology
    // A quantized V cache requires flash attention, which is on for every decoder dump.
    cp.type_k = kv_type;
    cp.type_v = kv_type;
    if (embd_mode) {
        cp.embeddings = true;
        // DUMP_EMBEDDINGS=none dumps per-token hidden states (with ALL_LOGITS=1) for hosts that pool themselves,
        // e.g. OpenVINO GenAI's TextEmbeddingPipeline; any other value keeps the in-graph mean pooling.
        const char* pool = getenv("DUMP_EMBEDDINGS");
        cp.pooling_type = !strcmp(pool, "none") ? LLAMA_POOLING_TYPE_NONE : LLAMA_POOLING_TYPE_MEAN;
        // Encoders attend bidirectionally, so flash attention with a causal mask is wrong here.
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }
    llama_context* ctx = llama_init_from_model(model, cp);
    // NON_CAUSAL=1: full bidirectional attention over the batch, as diffusion language models use.
    if (ctx && getenv("NON_CAUSAL")) {
        llama_set_causal_attn(ctx, false);
    }
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
    // llama.cpp sizes the attention window to the used KV cells (padded to 256), not the whole
    // cache: decoding into an empty cache would bake in a 256-slot window whatever n_kv is. Fill
    // the cells the captured batch does not occupy first, so the graph spans all n_kv of them.
    int32_t base_pos = 0;  // position of the captured batch's first row
    if (!embd_mode && !token_embd_only && !no_fill && n_kv > n_tokens) {
        static std::vector<llama_token> fill(n_kv - n_tokens, bos);
        base_pos = static_cast<int32_t>(fill.size());
        if (llama_decode(ctx, llama_batch_get_one(fill.data(), static_cast<int32_t>(fill.size()))) != 0) {
            fprintf(stderr, "KV fill decode failed\n");
            return 4;
        }
        cap.reset();
    }
    // A VLM decoder graph needs SOME embedding values to shape n_embd correctly; the actual
    // values do not matter, we are capturing topology, not running inference. llama_batch_init
    // already allocates b.embd -- fill it in place, do not repoint it at another buffer, or
    // llama_batch_free() will free memory it never allocated (this crashed with "double free or
    // corruption" the first time, from freeing a std::vector's internal buffer via C free()).
    llama_batch b;
    if (vlm_embd_input) {
        const int n_embd = llama_model_n_embd(model);
        // N_TOKENS rows give batched-prefill buckets for the embedding-input decoder too; only
        // the last row produces logits, as in the token-id path.
        b = llama_batch_init(n_tokens, n_embd, 1);
        b.n_tokens = n_tokens;
        std::fill(b.embd, b.embd + static_cast<size_t>(n_tokens) * n_embd, 0.0f);
        for (int i = 0; i < n_tokens; i++) {
            b.pos[i] = base_pos + i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = i == n_tokens - 1;
        }
    } else if (all_logits && getenv("PROMPT_IDS")) {
        // PROMPT_IDS="1 2 3 ...": real token ids instead of BOS repeated (needs N_TOKENS to match)
        std::vector<llama_token> ids;
        { char* e = strdup(getenv("PROMPT_IDS")); for (char* p = strtok(e, " ,"); p; p = strtok(nullptr, " ,")) ids.push_back(atoi(p)); free(e); }
        if (static_cast<int>(ids.size()) != n_tokens) { fprintf(stderr, "PROMPT_IDS has %zu ids, N_TOKENS is %d\n", ids.size(), n_tokens); return 1; }
        b = llama_batch_init(n_tokens, 0, 1);
        b.n_tokens = n_tokens;
        for (int i = 0; i < n_tokens; i++) {
            b.token[i] = ids[i];
            b.pos[i] = base_pos + i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = 1;
        }
    } else if (all_logits) {
        // Manual batch: llama_batch_get_one only flags the last row for logits, so build one
        // by hand and set every row's flag. Token/position values don't matter for topology.
        b = llama_batch_init(n_tokens, 0, 1);
        b.n_tokens = n_tokens;
        for (int i = 0; i < n_tokens; i++) {
            b.token[i] = bos;
            b.pos[i] = base_pos + i;
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

    if (token_embd_only && !keep_token_embd(cap)) {
        fprintf(stderr, "no GET_ROWS(token_embd.weight, ...) in the graph\n");
        return 5;
    }
    write_json(cap, out, path, arch_buf, n_tokens, n_kv);
    if (ref_out) {
        write_ref(cap, ref_out);
        write_ref_inputs(std::string(ref_out) + ".inputs");
    }
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
    printf("kv caches    : %zu (%s)\n", n_cache, ggml_type_name(kv_type));
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
