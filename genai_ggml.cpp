// OpenVINO GenAI driving the ggml-emitter path.
//
// The model graph comes from OpenVINO's native GGUF builder via a caller-supplied GraphEmitter
// that creates ggml tensors instead of ov::Nodes (ggml_emitter.hpp), and is executed by a ggml
// backend (Vulkan / CPU). GenAI supplies the tokenizer: ov::genai::Tokenizer accepts a .gguf path
// directly and builds tokenizer + detokenizer ov::Models from the GGUF metadata, so the same file
// feeds both halves.
//
// What this is NOT: it does not go through ov::genai::LLMPipeline. That pipeline is built around
// ov::InferRequest and cannot drive a ggml backend. Making it do so means a sibling pipeline
// implementation (KV slot management, sampler wiring, streaming); this harness is the execution
// core such a pipeline would sit on top of.
//
// The builder's token length T is a compile-time 1 in the frontend, so the graph is single-token.
// Prompt prefill is therefore done one token at a time -- correct, just not batched.
#include "ggml_emitter.hpp"

#include "gguf.h"
#include "builder/gguf_builder.hpp"

#include "openvino/genai/tokenizer.hpp"

using namespace ov::frontend::gguf;
using namespace ov2ggml;

namespace {

// One forward pass over the single-token graph at absolute position `pos`, writing K/V into
// cache slot `pos`. Returns the logits row.
void set_step_inputs(const std::map<std::string, ggml_tensor *> & ext, int32_t tok, int32_t pos,
                     int n_kv) {
    const int32_t oid = 0;
    const int64_t slot = pos;
    if (ext.count("inp_tokens"))  ggml_backend_tensor_set(ext.at("inp_tokens"), &tok, 0, 4);
    if (ext.count("inp_pos"))     ggml_backend_tensor_set(ext.at("inp_pos"), &pos, 0, 4);
    if (ext.count("inp_out_ids")) ggml_backend_tensor_set(ext.at("inp_out_ids"), &oid, 0, 4);
    if (ext.count("inp_kv_idx"))  ggml_backend_tensor_set(ext.at("inp_kv_idx"), &slot, 0, 8);

    // Causal mask for the one query token: it may attend to slots 0..pos, nothing beyond.
    // Rows past the first are padding (llama.cpp pads the mask to 64) and stay masked out.
    for (const auto & kv : ext) {
        if (kv.first.rfind("self_kq_mask", 0) != 0) continue;
        std::vector<ggml_fp16_t> m(ggml_nelements(kv.second), ggml_fp32_to_fp16(-INFINITY));
        for (int j = 0; j <= pos && j < n_kv; j++) {
            m[j] = ggml_fp32_to_fp16(0.0f);
        }
        ggml_backend_tensor_set(kv.second, m.data(), 0, m.size() * sizeof(ggml_fp16_t));
    }
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: genai_ggml <model.gguf> [prompt] [n_predict] [n_kv]\n");
        return 1;
    }
    const std::string path = argv[1];
    const std::string prompt = argc > 2 ? argv[2] : "The capital of France is";
    const int n_predict = argc > 3 ? atoi(argv[3]) : 16;
    const int n_kv = argc > 4 ? atoi(argv[4]) : 256;
    const int n_tokens = 1;  // the frontend builds a single-token graph (T == 1)

    // ---- GenAI tokenizer, straight from the .gguf ----------------------------------------
    ov::genai::Tokenizer tokenizer(path);
    auto enc = tokenizer.encode(prompt);
    const ov::Tensor ids = enc.input_ids;
    std::vector<int64_t> prompt_ids(ids.data<int64_t>(), ids.data<int64_t>() + ids.get_size());
    printf("prompt: %s\n", prompt.c_str());
    printf("prompt tokens (%zu):", prompt_ids.size());
    for (auto t : prompt_ids) printf(" %lld", (long long) t);
    printf("\n");

    if ((int) prompt_ids.size() + n_predict > n_kv) {
        fprintf(stderr, "prompt + n_predict exceeds n_kv (%d); pass a larger n_kv\n", n_kv);
        return 1;
    }

    // ---- ggml backend --------------------------------------------------------------------
    ggml_backend_load_all();
    const bool force_cpu = getenv("USE_CPU") != nullptr;
    ggml_backend_dev_t dev = force_cpu ? nullptr : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev && !force_cpu) dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    if (!dev) dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    printf("backend: %s\n", ggml_backend_name(backend));

    // ---- weights, packed, via ggml's gguf reader -------------------------------------------
    ggml_context * wctx = nullptr;
    gguf_init_params gp = {true, &wctx};
    gguf_context * gg = gguf_init_from_file(path.c_str(), gp);
    ggml_backend_alloc_ctx_tensors(wctx, backend);
    std::map<std::string, ggml_tensor *> wmap;
    FILE * f = fopen(path.c_str(), "rb");
    const size_t doff = gguf_get_data_offset(gg);
    std::vector<uint8_t> tmp;
    for (ggml_tensor * t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t)) {
        const int i = gguf_find_tensor(gg, ggml_get_name(t));
        if (i < 0) continue;
        tmp.resize(ggml_nbytes(t));
        fseek(f, (long) (doff + gguf_get_tensor_offset(gg, i)), SEEK_SET);
        if (fread(tmp.data(), 1, tmp.size(), f) != tmp.size()) return 2;
        ggml_backend_tensor_set(t, tmp.data(), 0, tmp.size());
        wmap[ggml_get_name(t)] = t;
    }
    fclose(f);
    printf("weights: %zu\n", wmap.size());

    // ---- build the graph once, through the OpenVINO GGUF builder ---------------------------
    ggml_context * ctx_ext = ggml_init({ggml_tensor_overhead() * 256, nullptr, true});
    ggml_context * ctx_g = ggml_init(
        {ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead_custom(MAX_NODES, false), nullptr, true});

    // -1 = derive the rope mode from op_case; ROPE_MODE=N forces a ggml mode for A/B testing.
    const int rope_override = getenv("ROPE_MODE") ? atoi(getenv("ROPE_MODE")) : -1;
    Out out;
    auto factory = [&](std::unordered_map<std::string, ov::Tensor> & w,
                       std::unordered_map<std::string, GgufTensorType> & q,
                       const std::string & arch) -> std::unique_ptr<GraphEmitter> {
        auto e = std::make_unique<GgmlEmitter>(w, q, arch, ctx_g, &wmap, n_tokens, n_kv, rope_override, &out);
        e->set_ext_ctx(ctx_ext);
        return e;
    };
    build_ggml_graph_from_gguf(path, factory);

    if (!out.unsupported.empty()) {
        printf("unsupported: %zu\n", out.unsupported.size());
        for (const auto & u : out.unsupported) printf("  %s\n", u.c_str());
    }
    if (!out.last) {
        fprintf(stderr, "no output tensor produced\n");
        return 3;
    }

    ggml_backend_alloc_ctx_tensors(ctx_ext, backend);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx_g, MAX_NODES, false);
    ggml_build_forward_expand(gf, out.last);
    printf("cgraph: %d nodes\n", ggml_graph_n_nodes(gf));

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) return 4;

    // KV caches live in ctx_ext, so they are allocated once and persist across compute calls;
    // ggml_set_rows writes through a view into that same buffer. Zero them before the first step.
    for (const auto & kv : out.externals) {
        if (kv.first.rfind("cache_", 0) != 0) continue;
        std::vector<char> z(ggml_nbytes(kv.second), 0);
        ggml_backend_tensor_set(kv.second, z.data(), 0, z.size());
    }

    // ---- generate ---------------------------------------------------------------------------
    if (getenv("STUB_FA")) {
        printf("NOTE: STUB_FA=1 -- attention is bypassed, so the text is meaningless.\n"
               "      This validates the integration mechanically, not the model.\n");
    }
    ggml_tensor * lg = out.last;
    const size_t n_vocab = (size_t) lg->ne[0];
    std::vector<float> logits(n_vocab);

    auto step = [&](int32_t tok, int32_t pos) -> int32_t {
        set_step_inputs(out.externals, tok, pos, n_kv);
        const ggml_status st = ggml_backend_graph_compute(backend, gf);
        if (st != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "compute failed at pos %d: %s\n", pos, ggml_status_to_string(st));
            exit(5);
        }
        ggml_backend_tensor_get(lg, logits.data(), 0, n_vocab * sizeof(float));
        size_t best = 0;
        for (size_t i = 1; i < n_vocab; i++) {
            if (logits[i] > logits[best]) best = i;
        }
        return (int32_t) best;
    };

    int pos = 0;
    int32_t next = 0;
    for (size_t i = 0; i < prompt_ids.size(); i++) {  // prefill, one token per pass
        next = step((int32_t) prompt_ids[i], pos++);
    }

    std::vector<int64_t> generated;
    std::string shown;
    printf("\noutput: %s", prompt.c_str());
    fflush(stdout);
    for (int i = 0; i < n_predict && pos < n_kv; i++) {
        generated.push_back(next);
        // Decode the whole run each step and print the delta, so multi-byte pieces that span
        // several tokens come out intact.
        const std::string full = tokenizer.decode(generated);
        if (full.size() > shown.size()) {
            fwrite(full.data() + shown.size(), 1, full.size() - shown.size(), stdout);
            fflush(stdout);
            shown = full;
        }
        next = step(next, pos++);
    }
    printf("\n\ngenerated %zu tokens over %d KV slots\n", generated.size(), pos);
    return 0;
}
