// Shared cgraph capture: turns a live ggml graph into the declarative "ov-cgraph-v1" artifact.
// Used by dump_cgraph (llama_context::cb_eval) and dump_vision (mtmd_context_params::cb_eval),
// both of which hand over each node once, with full metadata, before it is computed.
#pragma once

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "ggml.h"

namespace cgraph {

struct Node {
    std::string op;                        // "GGML_OP_MUL_MAT" / "GGML_GLU_OP_SWIGLU"
    std::string id;                        // unique; ggml names are NOT (see Capture::intern)
    std::string name;                      // the raw ggml name, for readability only
    std::vector<std::string> inputs;
    int64_t ne[4] = {1, 1, 1, 1};
    size_t nb[4] = {0, 0, 0, 0};
    std::string type;
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)] = {0};
    std::string view_src;
    size_t view_offs = 0;
    bool is_input = false;
    // Per-input shape/type, which the OV translators query and cannot re-derive once the
    // producing tensor is gone.
    std::vector<std::vector<int64_t>> input_ne;
    std::vector<std::string> input_type;
};

struct Capture {
    std::vector<Node> nodes;
    // A ggml graph identifies nodes by POINTER, not by name: llama.cpp calls cb() again after
    // each transform, so e.g. the mul_mat producing Qcur-0 and the rope consuming it are BOTH
    // named "Qcur-0". Keying on names silently drops nodes -- it cost us every ROPE node and
    // half the norms before this was caught. So intern pointers to unique ids and make
    // input references point at those.
    std::map<const ggml_tensor*, std::string> ids;
    std::map<std::string, std::string> weights;  // id -> ggml type
    std::map<std::string, std::string> weight_names;  // id -> gguf tensor name to resolve
    size_t counter = 0;

    // Model init routinely runs its own graph passes before we get control back --
    // clip_init's flash-attention auto-probe (reserve_compute_meta) and, if requested, an
    // explicit warmup encode both drive the SAME cb_eval. Their tensors then get freed, and
    // ggml's arena allocator hands the same addresses to the NEXT (real) pass, which corrupts
    // a pointer-keyed identity map: a later node can appear to reference itself, or two
    // distinct tensors can collide onto one id. Call this right before the pass you actually
    // want to capture, discarding anything seen so far.
    void reset() {
        nodes.clear();
        ids.clear();
        weights.clear();
        weight_names.clear();
        leaves.clear();
        counter = 0;
    }

    // Leaves are NOT nodes, so cb_eval never hands them over -- they surface only as src[] of
    // something else. Without recording them the artifact has dangling references: the KV
    // caches and the real graph inputs (tokens, positions, mask, kv index) would be named but
    // never described, and ggml's auto-named ones ("leaf_5") would be indistinguishable.
    struct Leaf {
        std::string id, name, type;
        int64_t ne[4] = {1, 1, 1, 1};
        size_t nb[4] = {0, 0, 0, 0};
        bool is_input = false;   // GGML_TENSOR_FLAG_INPUT: caller writes it each step
        bool has_buffer = false; // preallocated by llama.cpp: a weight or a KV cache
    };
    std::map<std::string, Leaf> leaves;

    void record_leaf(const ggml_tensor* t, const std::string& id) {
        if (leaves.count(id)) {
            return;
        }
        Leaf l;
        l.id = id;
        l.name = ggml_get_name(t);
        l.type = ggml_type_name(t->type);
        for (int i = 0; i < 4; i++) {
            l.ne[i] = t->ne[i];
            l.nb[i] = t->nb[i];
        }
        l.is_input = (t->flags & GGML_TENSOR_FLAG_INPUT) != 0;
        l.has_buffer = t->buffer != nullptr;
        leaves.emplace(id, std::move(l));
    }

    const std::string& intern(const ggml_tensor* t) {
        auto it = ids.find(t);
        if (it != ids.end()) {
            return it->second;
        }
        std::string nm = ggml_get_name(t);
        // Keep the ggml name in the id where it helps a human read the artifact, but always
        // suffix it so uniqueness never depends on llama.cpp's naming.
        for (char& c : nm) {
            if (c == ' ' || c == '(' || c == ')') c = '_';
        }
        return ids.emplace(t, "n" + std::to_string(counter++) + (nm.empty() ? "" : "_" + nm))
            .first->second;
    }
};

// The GGUF frontend spells ops "GGML_OP_<NAME>"; ggml_op_name gives "<NAME>". GLU is a family:
// the concrete function lives in op_params[0], and the frontend names those "GGML_GLU_OP_<NAME>".
inline std::string op_string(const ggml_tensor* t) {
    if (t->op == GGML_OP_GLU) {
        return std::string("GGML_GLU_OP_") + ggml_glu_op_name(static_cast<ggml_glu_op>(t->op_params[0]));
    }
    if (t->op == GGML_OP_UNARY) {
        return std::string("GGML_UNARY_OP_") + ggml_unary_op_name(static_cast<ggml_unary_op>(t->op_params[0]));
    }
    return std::string("GGML_OP_") + ggml_op_name(t->op);
}

inline bool eval_cb(ggml_tensor* t, bool ask, void* user_data) {
    auto* cap = static_cast<Capture*>(user_data);
    if (!ask) {
        return true;  // we never request isolated evaluation, so this branch is unused
    }
    // IM2COL is always the first compute op of an image encode (the patch embedding
    // convolution). Vision models tile one image into several sub-images and re-run the SAME
    // graph shape per tile, freeing and rebuilding the scratch ggml_context each time -- so
    // pointers from an earlier tile get reused by the next one, corrupting a pointer-keyed
    // identity map (a later node can appear to reference itself). Seeing IM2COL again after we
    // already have nodes means a new pass started; keep only the most recent one.
    if (t->op == GGML_OP_IM2COL && !cap->nodes.empty()) {
        cap->reset();
    }
    // The scheduler can revisit a node across splits; intern() makes the first sighting win.
    if (cap->ids.count(t)) {
        return false;
    }
    Node n;
    n.id = cap->intern(t);
    n.op = op_string(t);
    n.name = ggml_get_name(t);
    n.type = ggml_type_name(t->type);
    for (int i = 0; i < 4; i++) {
        n.ne[i] = t->ne[i];
        n.nb[i] = t->nb[i];
    }
    std::memcpy(n.op_params, t->op_params, sizeof(n.op_params));
    n.is_input = (t->flags & GGML_TENSOR_FLAG_INPUT) != 0;
    if (t->view_src) {
        const bool vs_new = !cap->ids.count(t->view_src);
        n.view_src = cap->intern(t->view_src);
        n.view_offs = t->view_offs;
        if (vs_new && t->view_src->op == GGML_OP_NONE) {
            cap->record_leaf(t->view_src, n.view_src);  // e.g. cache_k_l0 behind its view
        }
    }
    for (int i = 0; i < GGML_MAX_SRC && t->src[i]; i++) {
        ggml_tensor* s = t->src[i];
        const bool is_leaf = !cap->ids.count(s) && s->op == GGML_OP_NONE;
        const std::string sid = cap->intern(s);
        n.inputs.emplace_back(sid);
        n.input_ne.push_back({s->ne[0], s->ne[1], s->ne[2], s->ne[3]});
        n.input_type.emplace_back(ggml_type_name(s->type));
        // A source with no producing op and a backing buffer is a weight leaf; the loader
        // resolves it from the .gguf by its ORIGINAL ggml name, so record that mapping.
        if (is_leaf) {
            cap->record_leaf(s, sid);
            // A leaf with a buffer, no op and no INPUT flag is either a .gguf weight or a KV
            // cache. Weights are resolvable by name in the file; caches are not, so the
            // distinction is left to the loader, which has the file open.
            if (s->buffer && !(s->flags & GGML_TENSOR_FLAG_INPUT)) {
                cap->weights[sid] = ggml_type_name(s->type);
                cap->weight_names[sid] = ggml_get_name(s);
            }
        }
    }
    cap->nodes.push_back(std::move(n));
    return false;  // do not ask the scheduler to break this node out
}

inline void json_escape(FILE* f, const std::string& s) {
    for (char c : s) {
        if (c == '"' || c == '\\') {
            fprintf(f, "\\%c", c);
        } else {
            fputc(c, f);
        }
    }
}

inline void write_json(const Capture& cap, const std::string& path, const std::string& model,
                const std::string& arch, int n_tokens, int n_kv) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", path.c_str());
        return;
    }
    fprintf(f, "{\n  \"format\": \"ov-cgraph-v1\",\n");
    fprintf(f, "  \"source\": \"llama.cpp cb_eval\",\n");
    fprintf(f, "  \"model\": \"");
    json_escape(f, model);
    fprintf(f, "\",\n  \"architecture\": \"%s\",\n", arch.c_str());
    // Shapes are baked in, so the artifact is only valid for this configuration. Record it so a
    // loader can refuse a mismatch rather than produce silent nonsense.
    fprintf(f, "  \"n_tokens\": %d,\n  \"n_kv\": %d,\n", n_tokens, n_kv);
    fprintf(f, "  \"weights\": {\n");
    bool first = true;
    for (const auto& w : cap.weights) {
        fprintf(f, "%s    \"", first ? "" : ",\n");
        json_escape(f, w.first);
        fprintf(f, "\": {\"type\": \"%s\", \"gguf_name\": \"", w.second.c_str());
        json_escape(f, cap.weight_names.at(w.first));
        fprintf(f, "\"}");
        first = false;
    }
    fprintf(f, "\n  },\n  \"leaves\": [\n");
    first = true;
    for (const auto& kv : cap.leaves) {
        const auto& l = kv.second;
        fprintf(f, "%s    {\"id\": \"", first ? "" : ",\n");
        json_escape(f, l.id);
        fprintf(f, "\", \"name\": \"");
        json_escape(f, l.name);
        fprintf(f, "\", \"type\": \"%s\", \"ne\": [%" PRId64 ", %" PRId64 ", %" PRId64
                   ", %" PRId64 "], \"nb\": [%zu, %zu, %zu, %zu], \"is_input\": %s, "
                   "\"has_buffer\": %s}",
                l.type.c_str(), l.ne[0], l.ne[1], l.ne[2], l.ne[3], l.nb[0], l.nb[1], l.nb[2],
                l.nb[3], l.is_input ? "true" : "false", l.has_buffer ? "true" : "false");
        first = false;
    }
    fprintf(f, "\n  ],\n  \"nodes\": [\n");

    for (size_t i = 0; i < cap.nodes.size(); i++) {
        const Node& n = cap.nodes[i];
        fprintf(f, "    {\"op\": \"%s\", \"id\": \"", n.op.c_str());
        json_escape(f, n.id);
        fprintf(f, "\", \"name\": \"");
        json_escape(f, n.name);
        fprintf(f, "\", \"type\": \"%s\"", n.type.c_str());
        fprintf(f, ", \"ne\": [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]", n.ne[0],
                n.ne[1], n.ne[2], n.ne[3]);
        fprintf(f, ", \"nb\": [%zu, %zu, %zu, %zu]", n.nb[0], n.nb[1], n.nb[2], n.nb[3]);
        if (n.is_input) {
            fprintf(f, ", \"is_input\": true");
        }
        if (!n.view_src.empty()) {
            fprintf(f, ", \"view_src\": \"");
            json_escape(f, n.view_src);
            fprintf(f, "\", \"view_offs\": %zu", n.view_offs);
        }
        fprintf(f, ", \"inputs\": [");
        for (size_t j = 0; j < n.inputs.size(); j++) {
            fprintf(f, "%s\"", j ? ", " : "");
            json_escape(f, n.inputs[j]);
            fputc('"', f);
        }
        fprintf(f, "]");
        fprintf(f, ", \"input_ne\": [");
        for (size_t j = 0; j < n.input_ne.size(); j++) {
            fprintf(f, "%s[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]", j ? ", " : "",
                    n.input_ne[j][0], n.input_ne[j][1], n.input_ne[j][2], n.input_ne[j][3]);
        }
        fprintf(f, "]");
        fprintf(f, ", \"input_type\": [");
        for (size_t j = 0; j < n.input_type.size(); j++) {
            fprintf(f, "%s\"%s\"", j ? ", " : "", n.input_type[j].c_str());
        }
        fprintf(f, "]");
        // op_params is the raw ggml blob. Kept verbatim rather than decoded per op: the loader
        // decodes it the same way ggml-decoder.cpp::compute_op_case already does, so there is
        // exactly one place that knows the per-op layout.
        int last = -1;
        for (int k = 0; k < (int) (GGML_MAX_OP_PARAMS / sizeof(int32_t)); k++) {
            if (n.op_params[k] != 0) last = k;
        }
        if (last >= 0) {
            fprintf(f, ", \"op_params\": [");
            for (int k = 0; k <= last; k++) {
                fprintf(f, "%s%d", k ? ", " : "", n.op_params[k]);
            }
            fprintf(f, "]");
        }
        fprintf(f, "}%s\n", i + 1 == cap.nodes.size() ? "" : ",");
    }
    fprintf(f, "  ]\n}\n");
    fclose(f);
}
// One-line summary for the console: op histogram plus the leaf breakdown.
inline void report(const Capture& cap, const char* what) {
    std::map<std::string, int> hist;
    for (const auto& n : cap.nodes) {
        hist[n.op]++;
    }
    size_t n_in = 0;
    for (const auto& kv : cap.leaves) {
        if (kv.second.is_input) n_in++;
    }
    printf("captured     : %s\n", what);
    printf("nodes        : %zu\n", cap.nodes.size());
    printf("weight leaves: %zu\n", cap.weights.size());
    printf("graph inputs : %zu\n", n_in);
    printf("total leaves : %zu\n", cap.leaves.size());
    printf("op types     : %zu\n", hist.size());
    for (const auto& h : hist) {
        printf("  %-28s %4d\n", h.first.c_str(), h.second);
    }
}

}  // namespace cgraph
