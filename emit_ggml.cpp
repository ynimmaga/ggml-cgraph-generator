// Emitter route: OpenVINO's GGUF builder drives a caller-supplied emitter that creates ggml
// tensors directly. No GgufGraph, no decoder, no second frontend -- the builder's add_op() calls
// land straight on ggml. OpenVINO links no ggml; it only calls virtual methods on GraphEmitter.
//
// Single-step diagnostic harness: one token at position 0. The emitter itself lives in
// ggml_emitter.hpp; genai_ggml.cpp drives the same emitter with a real generate loop.
#include "ggml_emitter.hpp"

#include "gguf.h"
#include "builder/gguf_builder.hpp"

using namespace ov::frontend::gguf;
using namespace ov2ggml;

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: emit_ggml <model.gguf> [n_kv]\n");
        return 1;
    }
    const std::string path = argv[1];
    const int n_kv = argc > 2 ? atoi(argv[2]) : 64;
    const int n_tokens = 1;

    ggml_backend_load_all();
    const bool force_cpu = getenv("USE_CPU") != nullptr;
    ggml_backend_dev_t dev = force_cpu ? nullptr : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev && !force_cpu) dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    if (!dev) dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    printf("backend: %s\n", ggml_backend_name(backend));

    // weights, packed, via ggml's gguf reader
    ggml_context * wctx = nullptr;
    gguf_init_params gp = {true, &wctx};
    gguf_context * gg = gguf_init_from_file(path.c_str(), gp);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
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

    ggml_context * ctx_ext = ggml_init({ggml_tensor_overhead() * 256, nullptr, true});
    ggml_context * ctx_g = ggml_init(
        {ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead_custom(MAX_NODES, false), nullptr, true});

    // The builder needs its weight/qtype tables; the emitter only reads them through the base.
    // The builder parses the file, then calls this factory with its own weight/qtype tables --
    // which blocks/ query for weight shapes -- and drives the returned emitter.
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


    printf("unsupported: %zu\n", out.unsupported.size());
    for (const auto & u : out.unsupported) printf("  %s\n", u.c_str());
    if (!out.last) {
        fprintf(stderr, "no output tensor produced\n");
        return 3;
    }

    ggml_backend_buffer_t ebuf = ggml_backend_alloc_ctx_tensors(ctx_ext, backend);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx_g, MAX_NODES, false);
    ggml_build_forward_expand(gf, out.last);
    printf("cgraph: %d nodes\n", ggml_graph_n_nodes(gf));

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) return 4;

    const auto & ext = out.externals;
    const int32_t tok = 1, pos = 0, oid = 0;
    const int64_t slot = 0;
    if (ext.count("inp_tokens")) ggml_backend_tensor_set(ext.at("inp_tokens"), &tok, 0, 4);
    if (ext.count("inp_pos")) ggml_backend_tensor_set(ext.at("inp_pos"), &pos, 0, 4);
    if (ext.count("inp_out_ids")) ggml_backend_tensor_set(ext.at("inp_out_ids"), &oid, 0, 4);
    if (ext.count("inp_kv_idx")) ggml_backend_tensor_set(ext.at("inp_kv_idx"), &slot, 0, 8);
    for (const auto & kv : ext) {
        if (kv.first.rfind("self_kq_mask", 0) == 0) {
            std::vector<ggml_fp16_t> m(ggml_nelements(kv.second), ggml_fp32_to_fp16(-INFINITY));
            m[0] = ggml_fp32_to_fp16(0.0f);
            ggml_backend_tensor_set(kv.second, m.data(), 0, m.size() * sizeof(ggml_fp16_t));
        } else if (kv.first.rfind("cache_", 0) == 0) {
            std::vector<char> z(ggml_nbytes(kv.second), 0);
            ggml_backend_tensor_set(kv.second, z.data(), 0, z.size());
        }
    }

    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    printf("compute: %s\n", ggml_status_to_string(st));
    if (st != GGML_STATUS_SUCCESS) return 5;

    ggml_tensor * lg = out.last;
    std::vector<float> lo(lg->ne[0]);
    ggml_backend_tensor_get(lg, lo.data(), 0, lo.size() * sizeof(float));
    int best = 0;
    for (size_t i = 1; i < lo.size(); i++)
        if (lo[i] > lo[best]) best = (int) i;
    printf("logits ne0=%lld argmax=%d logit=%.4f (first 5: %.3f %.3f %.3f %.3f %.3f)\n",
           (long long) lg->ne[0], best, lo[best], lo[0], lo[1], lo[2], lo[3], lo[4]);
    return 0;
}
