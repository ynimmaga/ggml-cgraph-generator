// The GraphEmitter subclass that turns OpenVINO's GGUF builder callbacks into ggml tensors.
// Shared by emit_ggml.cpp (single-step diagnostic harness) and genai_ggml.cpp (generate loop).
//
// OpenVINO links no ggml: the builder only calls virtual methods on GraphEmitter, and this
// subclass is the only place ggml appears.
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include "builder/graph_emitter.hpp"

namespace ov2ggml {

using namespace ov::frontend::gguf;

constexpr int MAX_NODES = 8192;

inline std::vector<int64_t> to_ne(const ov::PartialShape & ps, int64_t dyn) {
    std::vector<int64_t> ne(4, 1);
    const size_t r = ps.size();
    for (size_t i = 0; i < r && i < 4; i++) {
        const auto & d = ps[r - 1 - i];
        ne[i] = d.is_static() ? d.get_length() : dyn;
    }
    return ne;
}

// Output state, owned by the caller: the emitter itself is destroyed together with the
// DecoderBuilder when build_ggml_graph_from_gguf returns.
struct Out {
    std::map<std::string, ggml_tensor *> map, externals;
    std::set<std::string> unsupported;
    ggml_tensor * last = nullptr;
};

// Every GGML op this emitter can translate. Kept beside build() deliberately: archprobe diffs
// an architecture's op inventory against this set to report missing mappings by name, which is
// how a newly added llama.cpp architecture announces what it needs instead of segfaulting.
// ADD A CASE TO build() AND A NAME HERE TOGETHER.
inline const std::set<std::string> & handled_ops() {
    static const std::set<std::string> ops = {
        "GGML_OP_NONE",     "GGML_OP_GET_ROWS",  "GGML_OP_MUL",      "GGML_OP_ADD",
        "GGML_OP_MUL_MAT",  "GGML_OP_RMS_NORM",  "GGML_OP_SCALE",    "GGML_OP_RESHAPE",
        "GGML_GLU_OP_SWIGLU", "GGML_GLU_OP_GEGLU", "GGML_OP_ROPE",    "GGML_OP_SET_ROWS", "GGML_OP_FLASH_ATTN_EXT",
    };
    return ops;
}

// Emits ggml tensors as the builder assembles the decoder.
class GgmlEmitter : public GraphEmitter {
public:
    GgmlEmitter(std::unordered_map<std::string, ov::Tensor> & weights,
                std::unordered_map<std::string, GgufTensorType> & qtypes,
                std::string arch,
                ggml_context * ctx,
                std::map<std::string, ggml_tensor *> * weights_by_name,
                int n_tokens,
                int n_kv,
                int rope_mode,
                Out * out)
        : GraphEmitter(weights, qtypes, std::move(arch)),
          m_ctx(ctx),
          m_w(weights_by_name),
          m_n_tokens(n_tokens),
          m_n_kv(n_kv),
          m_rope_mode(rope_mode),
          m_out(out) {}

    std::string add_op(const std::string & op_type,
                       const std::string & name,
                       const std::vector<std::string> & inputs,
                       const ov::PartialShape & out_shape,
                       ov::element::Type out_type,
                       int op_case,
                       std::map<std::string, ov::Any> attrs) override {
        // Keep the base's shape/type bookkeeping alive: blocks/ query shape_of_tensor().
        record_tensor_meta(name, out_shape, out_type);

        ggml_tensor * t = build(op_type, name, inputs, out_shape, op_case, attrs);
        if (t) {
            ggml_set_name(t, name.c_str());
            m_out->map[name] = t;
            m_out->last = t;
        } else {
            m_out->unsupported.insert(op_type + " (case " + std::to_string(op_case) + ")");
        }
        return name;
    }

    std::shared_ptr<ov::op::v0::Parameter> add_input(const std::string & name,
                                                     ov::element::Type type,
                                                     const ov::PartialShape & shape) override {
        auto p = GraphEmitter::add_input(name, type, shape);  // keeps model_inputs bookkeeping
        const bool cache = name.rfind("cache_", 0) == 0;
        auto ne = to_ne(shape, cache ? m_n_kv : m_n_tokens);
        ggml_type gt = GGML_TYPE_F32;
        if (cache) {
            gt = GGML_TYPE_F16;
        } else if (name == "inp_tokens" || name == "inp_pos" || name == "inp_out_ids") {
            gt = GGML_TYPE_I32;
        } else if (name == "inp_kv_idx") {
            gt = GGML_TYPE_I64;
        } else if (name.rfind("self_kq_mask", 0) == 0) {
            gt = GGML_TYPE_F16;
            ne[0] = m_n_kv;
            ne[1] = GGML_PAD(m_n_tokens, 64);
        }
        ggml_tensor * t = ggml_new_tensor_4d(m_ext_ctx, gt, ne[0], ne[1], ne[2], ne[3]);
        ggml_set_name(t, name.c_str());
        if (!cache) {
            ggml_set_input(t);
        }
        m_out->map[name] = t;
        m_out->externals[name] = t;
        return p;
    }

    void set_ext_ctx(ggml_context * c) { m_ext_ctx = c; }

private:
    ggml_tensor * get(const std::vector<std::string> & in, size_t i) {
        if (i >= in.size()) {
            return nullptr;
        }
        auto it = m_out->map.find(in[i]);
        if (it != m_out->map.end()) {
            return it->second;
        }
        auto w = m_w->find(in[i]);
        return w == m_w->end() ? nullptr : w->second;
    }

    ggml_tensor * build(const std::string & op,
                        const std::string & name,
                        const std::vector<std::string> & in,
                        const ov::PartialShape & out_shape,
                        int op_case,
                        std::map<std::string, ov::Any> & a) {
        auto attr = [&](const char * k) -> ov::Any {
            auto it = a.find(k);
            return it == a.end() ? ov::Any() : it->second;
        };
        if (op == "GGML_OP_NONE") {
            // A weight leaf carries the GGUF tensor name as the NODE name, not as an input.
            auto w = m_w->find(name);
            return w == m_w->end() ? nullptr : w->second;
        }
        if (op == "GGML_OP_GET_ROWS") return ggml_get_rows(m_ctx, get(in, 0), get(in, 1));
        if (op == "GGML_OP_MUL")      return ggml_mul(m_ctx, get(in, 0), get(in, 1));
        if (op == "GGML_OP_ADD")      return ggml_add(m_ctx, get(in, 0), get(in, 1));
        if (op == "GGML_OP_MUL_MAT")  return ggml_mul_mat(m_ctx, get(in, 0), get(in, 1));
        if (op == "GGML_OP_RMS_NORM") return ggml_rms_norm(m_ctx, get(in, 0), attr("eps").as<float>());
        if (op == "GGML_OP_SCALE") {
            // Granite-style scalar multipliers (embedding/attention/residual/logits) and any
            // other architecture that folds a constant scale+bias into the graph.
            auto s = attr("scale");
            auto b = attr("bias");
            return ggml_scale_bias(m_ctx, get(in, 0), s.empty() ? 1.0f : s.as<float>(),
                                   b.empty() ? 0.0f : b.as<float>());
        }
        if (op == "GGML_OP_RESHAPE") {
            auto ne = to_ne(out_shape, m_n_tokens);
            return ggml_reshape_4d(m_ctx, get(in, 0), ne[0], ne[1], ne[2], ne[3]);
        }
        if (op == "GGML_GLU_OP_SWIGLU" || op == "GGML_GLU_OP_GEGLU") {
            ggml_tensor * x = get(in, 0);
            ggml_tensor * y = get(in, 1);
            auto sw = attr("swapped");
            if (!sw.empty() && sw.as<bool>()) std::swap(x, y);
            return op == "GGML_GLU_OP_GEGLU" ? ggml_geglu_split(m_ctx, x, y)
                                             : ggml_swiglu_split(m_ctx, x, y);
        }
        if (op == "GGML_OP_ROPE") {
            const auto rc = attr("rope_config").as<RopeConfig>();
            // The rope variant is carried in the high 16 bits of op_case (arch_registry.hpp:
            // ROPE_OP_CASE_NORMAL/NEOX/IMROPE), which is the frontend's mirror of
            // llama_model_rope_type. Translate it to ggml's mode bits. Getting this wrong is
            // silent: a NEOX arch run as NORMAL still produces fluent-looking but degraded text.
            int mode = GGML_ROPE_TYPE_NORMAL;
            switch (op_case >> 16) {
                case 1:  mode = GGML_ROPE_TYPE_NEOX;   break;
                case 2:  mode = GGML_ROPE_TYPE_IMROPE; break;
                default: mode = GGML_ROPE_TYPE_NORMAL; break;
            }
            if (m_rope_mode >= 0) {
                mode = m_rope_mode;  // explicit override, for A/B testing the mapping
            }
            return ggml_rope_ext(m_ctx, get(in, 0), get(in, 1), get(in, 2), rc.n_dims, mode,
                                 rc.n_ctx_orig, rc.freq_base, rc.freq_scale, rc.ext_factor,
                                 rc.attn_factor, rc.beta_fast, rc.beta_slow);
        }
        if (op == "GGML_OP_SET_ROWS") {
            ggml_tensor * d = get(in, 0);
            ggml_tensor * idx = get(in, 1);
            ggml_tensor * c = get(in, 2);
            ggml_tensor * src = ggml_reshape_4d(m_ctx, d, d->ne[0] * d->ne[1], d->ne[2], d->ne[3], 1);
            ggml_tensor * dst = ggml_reshape_4d(m_ctx, c, c->ne[0] * c->ne[1], c->ne[2], c->ne[3], 1);
            ggml_tensor * r = ggml_set_rows(m_ctx, dst, src, idx);
            return ggml_reshape_4d(m_ctx, r, c->ne[0], c->ne[1], c->ne[2], c->ne[3]);
        }
        if (op == "GGML_OP_FLASH_ATTN_EXT") {
            // Diagnostic: bypass FA entirely. q already has FA's output shape
            // (ne = [head_size, n_head, n_tokens]), so the graph stays well-formed.
            if (getenv("STUB_FA")) {
                return get(in, 0);
            }
            // Diagnostic: feed q as k and v too. If the backends agree under this, the
            // permute and mask are fine and the fault is in the K/V path out of SET_ROWS.
            const bool qkv = getenv("STUB_KV") != nullptr;
            auto q = ggml_cont(m_ctx, ggml_permute(m_ctx, get(in, 0), 0, 2, 1, 3));
            auto k = ggml_cont(m_ctx, ggml_permute(m_ctx, get(in, qkv ? 0 : 1), 0, 2, 1, 3));
            auto v = ggml_cont(m_ctx, ggml_permute(m_ctx, get(in, qkv ? 0 : 2), 0, 2, 1, 3));
            auto sc = attr("scale");
            auto cap = attr("kq_soft_cap");
            return ggml_flash_attn_ext(m_ctx, q, k, v, get(in, 3), sc.empty() ? 1.0f : sc.as<float>(),
                                       0.0f, cap.empty() ? 0.0f : cap.as<float>());
        }
        return nullptr;
    }

    ggml_context * m_ctx = nullptr;
    ggml_context * m_ext_ctx = nullptr;
    std::map<std::string, ggml_tensor *> * m_w;
    Out * m_out;
    int m_n_tokens, m_n_kv, m_rope_mode;
};

}  // namespace ov2ggml
