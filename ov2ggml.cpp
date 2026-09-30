// ov2ggml: run a GGUF model on a ggml backend (Vulkan/CPU) using OpenVINO's GGUF frontend
// purely as the graph builder.
//
// Split of responsibilities:
//   * OpenVINO GGUF frontend  -> TOPOLOGY. FrontEnd::load(.gguf) gives a gguf::InputModel whose
//     visit_subgraph() walks the decoder-family graph the native builder assembled, in the GGML
//     op vocabulary, with typed attributes. All of the per-architecture knowledge comes from here.
//   * ggml's own gguf reader   -> WEIGHTS, left in their packed quant format (Q4_K etc.) so the
//     backend's quantized matmul kernels run at full speed. Node input names are the GGUF tensor
//     names, so binding is by name.
//   * ggml backend             -> COMPUTE.
//
// No llama.cpp: only ggml (tensor library + backend), not llama_model / llama_context.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#ifdef OV2GGML_VULKAN
#    include "ggml-vulkan.h"
#endif

#include "input_model.hpp"
#include "openvino/frontend/gguf/decoder.hpp"
#include "openvino/frontend/gguf/frontend.hpp"

using namespace ov::frontend::gguf;

namespace {

constexpr int MAX_NODES = 8192;

// OpenVINO carries shapes as [ne3, ne2, ne1, ne0]; ggml's ne[] is the reverse.
std::vector<int64_t> to_ggml_ne(const ov::PartialShape & ps, int64_t dyn_fill) {
    std::vector<int64_t> ne(4, 1);
    const size_t rank = ps.size();
    for (size_t i = 0; i < rank && i < 4; i++) {
        const auto & d = ps[rank - 1 - i];
        ne[i] = d.is_static() ? d.get_length() : dyn_fill;
    }
    return ne;
}

struct Weights {
    gguf_context * gguf = nullptr;
    ggml_context * ctx = nullptr;  // holds weight tensor metadata
    ggml_backend_buffer_t buf = nullptr;
    std::map<std::string, ggml_tensor *> by_name;

    bool load(const std::string & path, ggml_backend_t backend) {
        gguf_init_params p = {/*no_alloc*/ true, /*ctx*/ &ctx};
        gguf = gguf_init_from_file(path.c_str(), p);
        if (!gguf) {
            fprintf(stderr, "gguf_init_from_file failed\n");
            return false;
        }
        // Upload every weight to the backend, keeping its on-disk quant type.
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) {
            fprintf(stderr, "failed to allocate weight buffer\n");
            return false;
        }
        FILE * f = fopen(path.c_str(), "rb");
        if (!f) {
            return false;
        }
        const size_t data_off = gguf_get_data_offset(gguf);
        std::vector<uint8_t> tmp;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
            const int idx = gguf_find_tensor(gguf, ggml_get_name(t));
            if (idx < 0) {
                continue;
            }
            const size_t off = data_off + gguf_get_tensor_offset(gguf, idx);
            const size_t nb = ggml_nbytes(t);
            tmp.resize(nb);
            if (fseek(f, (long) off, SEEK_SET) != 0 || fread(tmp.data(), 1, nb, f) != nb) {
                fprintf(stderr, "short read for %s\n", ggml_get_name(t));
                fclose(f);
                return false;
            }
            ggml_backend_tensor_set(t, tmp.data(), 0, nb);
            by_name[ggml_get_name(t)] = t;
        }
        fclose(f);
        return true;
    }
};

// Builds a ggml cgraph from the OpenVINO gguf InputModel.
class GgmlBuilder {
public:
    GgmlBuilder(const std::shared_ptr<InputModel> & im, Weights & w, int rope_mode)
        : m_im(im), m_w(w), m_rope_mode(rope_mode) {}

    // n_tokens: tokens in this step; n_kv: live cache length (cache tensors are external).
    ggml_cgraph * build(ggml_context * ctx,
                        int n_tokens,
                        int n_kv,
                        const std::map<std::string, ggml_tensor *> & externals) {
        m_ctx = ctx;
        m_map.clear();
        m_n_tokens = n_tokens;
        m_n_kv = n_kv;
        for (const auto & kv : externals) {
            m_map[kv.first] = kv.second;
        }

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, MAX_NODES, false);
        m_im->visit_subgraph([&](std::shared_ptr<GgufDecoder> nd) {
            ggml_tensor * out = translate(nd);
            if (out) {
                const std::string name = nd->get_op_name();
                ggml_set_name(out, name.c_str());
                m_map[name] = out;
                m_last = out;
            }
        });
        // The final MUL_MAT against output.weight produces the logits.
        ggml_build_forward_expand(gf, m_last);
        for (const auto & name : m_im->get_model_output_names()) {
            auto it = m_map.find(name);
            if (it != m_map.end() && it->second != m_last) {
                ggml_build_forward_expand(gf, it->second);
            }
        }
        return gf;
    }

    const std::set<std::string> & unsupported() const { return m_unsupported; }
    ggml_tensor * logits() const { return m_last; }

private:
    ggml_tensor * in(const std::shared_ptr<GgufDecoder> & nd, size_t i) {
        const auto names = nd->get_input_names();
        if (i >= names.size()) {
            return nullptr;
        }
        auto it = m_map.find(names[i]);
        if (it != m_map.end()) {
            return it->second;
        }
        auto wit = m_w.by_name.find(names[i]);
        return wit == m_w.by_name.end() ? nullptr : wit->second;
    }

    ggml_tensor * translate(const std::shared_ptr<GgufDecoder> & nd) {
        const std::string op = nd->get_op_type();
        const auto oc_any = nd->get_attribute("op_case");
        const int oc = oc_any.empty() ? 0 : oc_any.as<int>();

        if (op == "GGML_OP_NONE") {
            // Weight leaf: bind the packed tensor ggml's gguf reader already loaded.
            auto it = m_w.by_name.find(nd->get_op_name());
            if (it == m_w.by_name.end()) {
                m_unsupported.insert("weight:" + nd->get_op_name());
            }
            return it == m_w.by_name.end() ? nullptr : it->second;
        }
        if (op == "GGML_OP_GET_ROWS") {
            return ggml_get_rows(m_ctx, in(nd, 0), in(nd, 1));
        }
        if (op == "GGML_OP_RMS_NORM") {
            return ggml_rms_norm(m_ctx, in(nd, 0), nd->get_attribute("eps").as<float>());
        }
        if (op == "GGML_OP_MUL") {
            return ggml_mul(m_ctx, in(nd, 0), in(nd, 1));
        }
        if (op == "GGML_OP_ADD") {
            return ggml_add(m_ctx, in(nd, 0), in(nd, 1));
        }
        if (op == "GGML_OP_MUL_MAT") {
            return ggml_mul_mat(m_ctx, in(nd, 0), in(nd, 1));
        }
        if (op == "GGML_OP_RESHAPE") {
            auto ne = to_ggml_ne(nd->get_output_shape(), m_n_tokens);
            return ggml_reshape_4d(m_ctx, in(nd, 0), ne[0], ne[1], ne[2], ne[3]);
        }
        if (op == "GGML_GLU_OP_SWIGLU") {
            ggml_tensor * a = in(nd, 0);
            ggml_tensor * b = in(nd, 1);
            const auto sw = nd->get_attribute("swapped");
            if (!sw.empty() && sw.as<bool>()) {
                std::swap(a, b);
            }
            return ggml_swiglu_split(m_ctx, a, b);
        }
        if (op == "GGML_OP_ROPE") {
            const auto rc = nd->get_attribute("rope_config").as<RopeConfig>();
            return ggml_rope_ext(m_ctx, in(nd, 0), in(nd, 1), in(nd, 2), rc.n_dims, m_rope_mode,
                                 rc.n_ctx_orig, rc.freq_base, rc.freq_scale, rc.ext_factor,
                                 rc.attn_factor, rc.beta_fast, rc.beta_slow);
        }
        if (op == "GGML_OP_SET_ROWS") {
            // SET_ROWS(data, idx, cache): write this step's rows into the external cache.
            ggml_tensor * data = in(nd, 0);
            ggml_tensor * idx = in(nd, 1);
            ggml_tensor * cache = in(nd, 2);
            // ggml_set_rows wants i64 indices and a 2D-collapsed source view.
            // A cache row is one token's (n_head_kv * head_size) values, which are contiguous in
            // the head-major layout, so collapsing ne0*ne1 gives ggml_set_rows the [row, token]
            // view it wants. Reshape the result back to 4D: downstream FLASH_ATTN_EXT needs the
            // head-major cache shape, and keeping the set_rows result as the producer preserves
            // the ordering edge so the write happens before the read.
            ggml_tensor * src = ggml_reshape_4d(m_ctx, data, data->ne[0] * data->ne[1],
                                                data->ne[2], data->ne[3], 1);
            ggml_tensor * dst = ggml_reshape_4d(m_ctx, cache, cache->ne[0] * cache->ne[1],
                                                cache->ne[2], cache->ne[3], 1);
            ggml_tensor * r = ggml_set_rows(m_ctx, dst, src, idx);
            return ggml_reshape_4d(m_ctx, r, cache->ne[0], cache->ne[1], cache->ne[2], cache->ne[3]);
        }
        if (op == "GGML_OP_FLASH_ATTN_EXT") {
            // op_case 100: operands are ggml-natural [B, tokens, heads, head_size] in OV order,
            // i.e. ne = [head_size, heads, tokens, B]. ggml_flash_attn_ext wants
            // ne = [head_size, tokens, heads, B], so permute the head/token axes on all three.
            ggml_tensor * q = ggml_permute(m_ctx, in(nd, 0), 0, 2, 1, 3);
            ggml_tensor * k = ggml_permute(m_ctx, in(nd, 1), 0, 2, 1, 3);
            ggml_tensor * v = ggml_permute(m_ctx, in(nd, 2), 0, 2, 1, 3);
            ggml_tensor * mask = in(nd, 3);
            const auto sc = nd->get_attribute("scale");
            const auto cap = nd->get_attribute("kq_soft_cap");
            ggml_tensor * r = ggml_flash_attn_ext(m_ctx, ggml_cont(m_ctx, q), ggml_cont(m_ctx, k),
                                                  ggml_cont(m_ctx, v), mask,
                                                  sc.empty() ? 1.0f : sc.as<float>(), 0.0f,
                                                  cap.empty() ? 0.0f : cap.as<float>());
            // ggml FA returns [head_size, heads, tokens, B]; the graph expects the builder's
            // [B, tokens, heads, head_size] OV order, which is the same ne layout. No permute.
            return r;
        }
        m_unsupported.insert(op + " (case " + std::to_string(oc) + ")");
        return nullptr;
    }

    std::shared_ptr<InputModel> m_im;
    Weights & m_w;
    int m_rope_mode;
    ggml_context * m_ctx = nullptr;
    std::map<std::string, ggml_tensor *> m_map;
    std::set<std::string> m_unsupported;
    ggml_tensor * m_last = nullptr;
    int m_n_tokens = 1;
    int m_n_kv = 1;
};

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: ov2ggml <model.gguf> [n_kv] [--cpu]\n");
        return 1;
    }
    const std::string path = argv[1];
    const int n_kv = argc > 2 ? atoi(argv[2]) : 32;
    bool use_cpu = false;
    for (int i = 2; i < argc; i++) {
        if (std::string(argv[i]) == "--cpu") {
            use_cpu = true;
        }
    }

    // Backends ship as separate shared objects and register themselves; go through the registry
    // rather than a backend-specific init, so this binary is not tied to Vulkan at link time.
    ggml_backend_load_all();
    printf("ggml devices (%zu):", ggml_backend_dev_count());
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        printf(" [%s type=%d]", ggml_backend_dev_name(d), (int) ggml_backend_dev_type(d));
    }
    printf("\n");
    ggml_backend_dev_t dev = nullptr;
    if (!use_cpu) {
        // An Intel iGPU registers as IGPU, not GPU, so accept either.
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!dev) {
            dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
        }
    }
    if (!dev) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    if (!dev) {
        fprintf(stderr, "no ggml backend device found\n");
        return 8;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) {
        fprintf(stderr, "ggml_backend_dev_init failed\n");
        return 9;
    }
    printf("backend: %s (device %s)\n", ggml_backend_name(backend), ggml_backend_dev_name(dev));

    // ---- weights, packed, via ggml's own gguf reader ----
    Weights w;
    if (!w.load(path, backend)) {
        return 2;
    }
    printf("weights: %zu tensors\n", w.by_name.size());

    // ---- topology, via OpenVINO's GGUF frontend ----
    FrontEnd fe;
    auto im = std::dynamic_pointer_cast<InputModel>(fe.load(path));
    if (!im) {
        fprintf(stderr, "not a gguf::InputModel\n");
        return 3;
    }
    const auto rope = im->get_rope_config();
    printf("topology: %zu inputs, %zu outputs, rope n_dims=%d freq_base=%.0f\n",
           im->get_model_inputs().size(), im->get_model_output_names().size(), rope.n_dims,
           rope.freq_base);

    // ---- external tensors: graph inputs and the KV cache ----
    const int n_tokens = 1;
    ggml_init_params ip = {ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead_custom(MAX_NODES, false),
                           nullptr, true};
    ggml_context * ctx_in = ggml_init({ggml_tensor_overhead() * 256, nullptr, true});

    std::map<std::string, ggml_tensor *> externals;
    for (const auto & kv : im->get_model_inputs()) {
        const auto & name = kv.first;
        auto ne = to_ggml_ne(kv.second->get_output_partial_shape(0),
                             name.rfind("cache_", 0) == 0 ? n_kv : n_tokens);
        ggml_type type = GGML_TYPE_F32;
        if (name.rfind("cache_", 0) == 0) {
            type = GGML_TYPE_F16;
        } else if (name == "inp_tokens" || name == "inp_pos" || name == "inp_out_ids") {
            type = GGML_TYPE_I32;
        } else if (name == "inp_kv_idx") {
            type = GGML_TYPE_I64;
        } else if (name.rfind("self_kq_mask", 0) == 0) {
            // ggml_flash_attn_ext requires an F16 contiguous mask, and its kernels expect the
            // token axis padded to GGML_KQ_MASK_PAD (llama.cpp pads the same way).
            type = GGML_TYPE_F16;
            ne[0] = n_kv;
            ne[1] = GGML_PAD(n_tokens, 64);  // llama.cpp's GGML_KQ_MASK_PAD
        }
        ggml_tensor * t = ggml_new_tensor_4d(ctx_in, type, ne[0], ne[1], ne[2], ne[3]);
        ggml_set_name(t, name.c_str());
        // KV caches are written by SET_ROWS and read back by FLASH_ATTN_EXT in the same graph.
        // Marking them as graph inputs makes that an alias gallocr does not guarantee, and the
        // backends then disagree. Keep them as plain persistent tensors, like llama.cpp's KV
        // cache, and only flag the true per-step inputs.
        if (name.rfind("cache_", 0) != 0) {
            ggml_set_input(t);
        }
        externals[name] = t;
    }
    ggml_backend_buffer_t buf_in = ggml_backend_alloc_ctx_tensors(ctx_in, backend);
    if (!buf_in) {
        fprintf(stderr, "failed to allocate input buffer\n");
        return 4;
    }

    // ---- build the ggml graph from OV's topology ----
    ggml_context * ctx_g = ggml_init(ip);
    // llama arch uses NORM rope (mode 0); NEOX architectures would need mode 2.
    GgmlBuilder gb(im, w, /*rope_mode*/ 0);
    ggml_cgraph * gf = gb.build(ctx_g, n_tokens, n_kv, externals);
    printf("cgraph: %d nodes\n", ggml_graph_n_nodes(gf));
    if (!gb.unsupported().empty()) {
        printf("UNSUPPORTED (%zu):\n", gb.unsupported().size());
        for (const auto & u : gb.unsupported()) {
            printf("  %s\n", u.c_str());
        }
    }
    if (ggml_graph_n_nodes(gf) == 0) {
        fprintf(stderr, "empty graph\n");
        return 5;
    }

    // ---- allocate and run ----
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        fprintf(stderr, "gallocr_alloc_graph failed\n");
        return 6;
    }

    // Minimal single-step inputs: one token at position 0, empty causal mask.
    const int32_t tok = 1, pos = 0, out_id = 0;
    const int64_t kv_slot = 0;
    ggml_backend_tensor_set(externals["inp_tokens"], &tok, 0, sizeof(tok));
    ggml_backend_tensor_set(externals["inp_pos"], &pos, 0, sizeof(pos));
    if (externals.count("inp_out_ids")) {
        ggml_backend_tensor_set(externals["inp_out_ids"], &out_id, 0, sizeof(out_id));
    }
    ggml_backend_tensor_set(externals["inp_kv_idx"], &kv_slot, 0, sizeof(kv_slot));
    if (externals.count("self_kq_mask")) {
        ggml_tensor * m = externals["self_kq_mask"];
        std::vector<ggml_fp16_t> mask(ggml_nelements(m), ggml_fp32_to_fp16(-INFINITY));
        mask[0] = ggml_fp32_to_fp16(0.0f);  // the single token attends only to KV slot 0
        ggml_backend_tensor_set(m, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }
    for (const auto & kv : externals) {
        if (kv.first.rfind("cache_", 0) == 0) {
            std::vector<char> z(ggml_nbytes(kv.second), 0);
            ggml_backend_tensor_set(kv.second, z.data(), 0, z.size());
        }
    }

    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    printf("compute: %s\n", ggml_status_to_string(st));
    if (st != GGML_STATUS_SUCCESS) {
        return 7;
    }

    ggml_tensor * logits = gb.logits();
    printf("logits tensor: %s ne=[%lld,%lld,%lld,%lld] type=%s\n", ggml_get_name(logits),
           (long long) logits->ne[0], (long long) logits->ne[1], (long long) logits->ne[2],
           (long long) logits->ne[3], ggml_type_name(logits->type));

    const int64_t n_vocab = logits->ne[0];
    std::vector<float> out(n_vocab);
    ggml_backend_tensor_get(logits, out.data(), 0, n_vocab * sizeof(float));
    int best = 0;
    for (int64_t i = 1; i < n_vocab; i++) {
        if (out[i] > out[best]) {
            best = (int) i;
        }
    }
    printf("argmax token=%d logit=%.4f  (first 5: %.3f %.3f %.3f %.3f %.3f)\n", best, out[best],
           out[0], out[1], out[2], out[3], out[4]);

    ggml_gallocr_free(alloc);
    ggml_free(ctx_g);
    ggml_backend_buffer_free(buf_in);
    ggml_free(ctx_in);
    ggml_backend_buffer_free(w.buf);
    ggml_free(w.ctx);
    gguf_free(w.gguf);
    ggml_backend_free(backend);
    return 0;
}
