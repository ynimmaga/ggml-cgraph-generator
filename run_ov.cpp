// Run the SAME GGUF graph through OpenVINO: FrontEnd::load -> convert() -> ov::Model ->
// compile_model(device) -> infer. Inputs are bound identically to emit_ggml (one token at
// position 0, attending only KV slot 0, zeroed caches) so the logits are directly comparable.
//
// No GGUFMakeStateful is registered, so the model stays stateless: every KV cache is a
// Parameter and a Result, which is what lets us bind them like ordinary inputs.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "input_model.hpp"
#include "openvino/frontend/gguf/frontend.hpp"
#include "openvino/openvino.hpp"

using namespace ov::frontend::gguf;

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: run_ov <model.gguf> [n_kv] [device]\n");
        return 1;
    }
    const std::string path = argv[1];
    const int n_kv = argc > 2 ? atoi(argv[2]) : 64;
    const std::string device = argc > 3 ? argv[3] : "CPU";
    const int n_tokens = 1;

    FrontEnd fe;
    auto im = fe.load(path);
    auto model = fe.convert(im);
    printf("model: %zu inputs, %zu outputs\n", model->inputs().size(), model->outputs().size());

    ov::Core core;
    auto compiled = core.compile_model(model, device);
    auto req = compiled.create_infer_request();
    printf("device: %s\n", device.c_str());

    for (const auto & in : compiled.inputs()) {
        printf("  in: %-22s %-22s %s\n", in.get_any_name().c_str(),
               in.get_partial_shape().to_string().c_str(),
               in.get_element_type().get_type_name().c_str());
    }
    for (const auto & in : compiled.inputs()) {
        const std::string name = in.get_any_name();
        const auto & ps = in.get_partial_shape();
        ov::Shape shape;
        for (size_t i = 0; i < ps.size(); i++) {
            if (ps[i].is_static()) {
                shape.push_back(ps[i].get_length());
            } else {
                // Only two axes are dynamic here: the token axis and the KV axis.
                const bool cache = name.rfind("cache_", 0) == 0;
                const bool mask = name.rfind("self_kq_mask", 0) == 0;
                if (cache) {
                    shape.push_back(n_kv);                       // [1, n_kv, heads, head_size]
                } else if (mask) {
                    shape.push_back(i == 2 ? n_tokens : n_kv);   // [1, 1, tokens, kv]
                } else {
                    shape.push_back(n_tokens);
                }
            }
        }
        ov::Tensor t(in.get_element_type(), shape);
        if (name.rfind("cache_", 0) != 0 && name.rfind("self_kq_mask", 0) != 0 &&
            name.rfind("inp_", 0) != 0) {
            printf("  aux input: %-24s %s\n", name.c_str(), t.get_shape().to_string().c_str());
        }
        const size_t n = t.get_size();
        if (name.rfind("cache_", 0) == 0) {
            memset(t.data(), 0, t.get_byte_size());
        } else if (name.rfind("self_kq_mask", 0) == 0) {
            auto * p = t.data<float>();
            for (size_t i = 0; i < n; i++) {
                p[i] = -INFINITY;
            }
            p[0] = 0.0f;  // the single token attends only to KV slot 0
        } else if (name == "inp_tokens") {
            t.data<int32_t>()[0] = 1;
        } else if (name.rfind("attention_size", 0) == 0) {
            for (size_t i = 0; i < n; i++) t.data<int64_t>()[i] = n_kv;
        } else if (name == "token_len_per_seq" || name == "n_seq_active" ||
                   name == "seq_active_end") {
            for (size_t i = 0; i < n; i++) t.data<int64_t>()[i] = n_tokens;
        } else {
            memset(t.data(), 0, t.get_byte_size());  // inp_pos / inp_out_ids / inp_kv_idx = 0
        }
        req.set_tensor(in, t);
    }

    req.infer();

    // The logits output is the one whose last dim is the vocab (much larger than any cache).
    ov::Tensor best_out;
    for (const auto & o : compiled.outputs()) {
        auto t = req.get_tensor(o);
        if (!best_out || t.get_size() > best_out.get_size()) {
            best_out = t;
        }
    }
    const size_t n_vocab = best_out.get_size();
    const auto * lg = best_out.data<float>();
    size_t best = 0;
    for (size_t i = 1; i < n_vocab; i++) {
        if (lg[i] > lg[best]) {
            best = i;
        }
    }
    printf("logits n=%zu argmax=%zu logit=%.4f (first 5: %.3f %.3f %.3f %.3f %.3f)\n", n_vocab, best,
           lg[best], lg[0], lg[1], lg[2], lg[3], lg[4]);
    return 0;
}
