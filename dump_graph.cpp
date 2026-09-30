// Dump what OpenVINO's GGUF frontend exposes for a .gguf file, through the PUBLIC consumer
// interface only: FrontEnd::load -> gguf::InputModel -> visit_subgraph -> GgufDecoder.
//
// This is the branch point for the ov2ggml POC: the same interface serves both the native
// .gguf builder path and the llama.cpp cgraph path, and every accessor used here is
// GGUF_FRONTEND_API-exported. Output drives the op set the ggml translator must cover.
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "input_model.hpp"  // ov::frontend::gguf::InputModel (declaration; symbols are exported)
#include "openvino/frontend/gguf/decoder.hpp"
#include "openvino/frontend/gguf/frontend.hpp"

using namespace ov::frontend::gguf;

namespace {

std::string any_brief(const ov::Any & a) {
    if (a.empty()) {
        return "<empty>";
    }
    if (a.is<bool>()) {
        return a.as<bool>() ? "true" : "false";
    }
    if (a.is<int>() || a.is<int64_t>() || a.is<float>() || a.is<double>() || a.is<std::string>()) {
        return a.as<std::string>();
    }
    if (a.is<ov::Tensor>()) {
        const auto t = a.as<ov::Tensor>();
        return "Tensor" + t.get_shape().to_string() + ":" + t.get_element_type().get_type_name();
    }
    return std::string("<") + a.type_info().name() + ">";
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: dump_graph <model.gguf> [--full]\n");
        return 1;
    }
    const bool full = argc > 2 && std::string(argv[2]) == "--full";

    FrontEnd fe;
    auto im_base = fe.load(std::string(argv[1]));
    auto im = std::dynamic_pointer_cast<InputModel>(im_base);
    if (!im) {
        fprintf(stderr, "load() did not return a gguf::InputModel\n");
        return 2;
    }

    const auto & inputs = im->get_model_inputs();
    const auto outputs = im->get_model_output_names();
    const auto rope = im->get_rope_config();
    const auto & recurrent = im->get_recurrent_states();
    const auto & decoder = im->get_model_decoder();

    printf("inputs=%zu outputs=%zu recurrent=%zu extra=%zu\n", inputs.size(), outputs.size(),
           recurrent.size(), decoder->get_model_extra_inputs().size());
    printf("rope: n_dims=%d n_ctx_orig=%d freq_base=%.1f freq_scale=%.4f per_op=%d imrope=%d\n",
           rope.n_dims, rope.n_ctx_orig, rope.freq_base, rope.freq_scale, (int) rope.per_op,
           (int) rope.is_imrope);
    printf("tokenizer_config keys=%zu\n", decoder->get_tokenizer_config().size());

    std::map<std::string, int> by_op;
    std::map<std::string, std::set<int>> cases;
    std::map<std::string, std::set<std::string>> attrs_by_op;
    size_t n_nodes = 0, n_weight_leaves = 0;

    im->visit_subgraph([&](std::shared_ptr<GgufDecoder> nd) {
        n_nodes++;
        const std::string op = nd->get_op_type();
        by_op[op]++;
        const auto oc = nd->get_attribute("op_case");
        cases[op].insert(oc.empty() ? 0 : oc.as<int>());
        if (!nd->get_attribute("gguf_weight").empty()) {
            n_weight_leaves++;
        }
        // Probe the attribute keys the translators are known to read.
        static const char * keys[] = {"eps", "scale", "bias", "max_bias", "swapped", "op_case",
                                      "output_type", "rope_config", "kq_soft_cap", "softmax_axis",
                                      "layer_idx", "view_slice", "input_ggml_shape", "gguf_weight",
                                      "gguf_qtype", "quant_type", "data", "clamp_min", "clamp_max",
                                      "glu_alpha", "glu_limit", "ffn_geglu"};
        for (const char * k : keys) {
            if (!nd->get_attribute(k).empty()) {
                attrs_by_op[op].insert(k);
            }
        }
        if (full) {
            printf("%-24s case=%-4d %-46s %-20s <-", op.c_str(),
                   oc.empty() ? 0 : oc.as<int>(), nd->get_op_name().c_str(),
                   nd->get_output_shape().to_string().c_str());
            for (const auto & in : nd->get_input_names()) {
                printf(" %s", in.c_str());
            }
            printf("\n");
        }
    });

    printf("\nnodes=%zu (weight leaves=%zu)\n", n_nodes, n_weight_leaves);
    printf("\n-- op histogram: count, op_cases, attribute keys present --\n");
    for (const auto & kv : by_op) {
        printf("%-26s %5d  cases:", kv.first.c_str(), kv.second);
        for (int k : cases[kv.first]) {
            printf(" %d", k);
        }
        printf("   attrs:");
        for (const auto & a : attrs_by_op[kv.first]) {
            printf(" %s", a.c_str());
        }
        printf("\n");
    }

    printf("\n-- model inputs --\n");
    for (const auto & kv : inputs) {
        printf("  %-22s %s\n", kv.first.c_str(),
               kv.second->get_output_partial_shape(0).to_string().c_str());
    }
    printf("\n-- extra inputs --\n");
    for (const auto & kv : decoder->get_model_extra_inputs()) {
        printf("  %s\n", kv.first.c_str());
    }
    printf("\n-- outputs (%zu) --\n", outputs.size());
    for (size_t i = 0; i < outputs.size() && i < 8; i++) {
        printf("  %s\n", outputs[i].c_str());
    }
    if (outputs.size() > 8) {
        printf("  ... and %zu more\n", outputs.size() - 8);
    }
    return 0;
}
