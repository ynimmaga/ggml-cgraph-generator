// Does GenAI's Sampler actually run against logits produced by a ggml Vulkan backend?
//
// The claim under test: ov::genai::Sampler has zero coupling to ov::InferRequest -- sample()
// takes a plain ov::Tensor -- so the whole sampling stack (temperature, top-k/top-p, penalties,
// stop conditions, structured output) should work unchanged over a graph OpenVINO never executed.
#include <cstdio>
#include <string>
#include <vector>

#include "openvino/ggml_emitter/emitter.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/generation_config.hpp"
#include "sampling/sampler.hpp"
#include "sequence_group.hpp"

using ov::ggml_emitter::GgmlModel;

int main(int argc, char** argv) {
    const std::string path = argv[1];
    const std::string cgraph = argv[2];
    const std::string prompt = argc > 3 ? argv[3] : "The capital of France is";
    const int n_predict = argc > 4 ? atoi(argv[4]) : 12;

    ov::genai::Tokenizer tok(path);
    auto ids = tok.encode(prompt).input_ids;
    std::vector<int64_t> prompt_ids(ids.data<int64_t>(), ids.data<int64_t>() + ids.get_size());

    auto model = GgmlModel::from_cgraph(cgraph, path, "");
    const int n_kv = (int) model->context_size();
    printf("backend: %s  nodes: %zu\n", model->backend_name().c_str(), model->node_count());

    // Real GenAI sampling config -- not greedy.
    ov::genai::GenerationConfig cfg;
    cfg.max_new_tokens = n_predict;
    cfg.do_sample = true;
    cfg.temperature = 0.8f;
    cfg.top_p = 0.9f;
    cfg.top_k = 40;
    cfg.repetition_penalty = 1.1f;
    cfg.rng_seed = 42;

    ov::genai::Sampler sampler(tok);
    auto group = std::make_shared<ov::genai::SequenceGroup>(0, prompt_ids, cfg);
    group->schedule_tokens(prompt_ids.size());
    group->set_num_validated_tokens(0);

    const size_t n_vocab = model->logits_size();
    std::vector<float> logits(n_vocab);

    auto forward = [&](int32_t token, int32_t pos) {
        const int32_t oid = 0;
        const int64_t slot = pos;
        model->write_input("inp_tokens", &token, 4);
        model->write_input("inp_pos", &pos, 4);
        model->write_input("inp_out_ids", &oid, 4);
        for (const auto& n : model->input_names())
            if (n.rfind("inp_kv_idx", 0) == 0) model->write_input(n, &slot, 8);
        for (const auto& n : model->input_names()) {
            if (n.rfind("self_kq_mask", 0) != 0 && n.rfind("attn_inp_kq_mask", 0) != 0) continue;
            std::vector<uint16_t> m(model->input_size(n), 0xFC00);
            for (int j = 0; j <= pos && j < n_kv; j++) m[j] = 0;
            model->write_input(n, m.data(), m.size() * 2);
        }
        if (!model->compute()) { fprintf(stderr, "compute failed\n"); exit(2); }
        model->read_logits(logits.data());
    };

    int pos = 0;
    for (size_t i = 0; i < prompt_ids.size(); i++) forward((int32_t) prompt_ids[i], pos++);

    // Hand the ggml logits straight to GenAI. Shape is [batch, seq, vocab].
    std::vector<int64_t> generated;
    for (int i = 0; i < n_predict && pos < n_kv; i++) {
        ov::Tensor lg(ov::element::f32, ov::Shape{1, 1, n_vocab}, logits.data());
        auto out = sampler.sample({group}, lg);
        auto running = group->get_running_sequences();
        if (running.empty()) { printf("\n[stopped by GenAI sampler]\n"); break; }
        const auto& gen = running[0]->get_generated_ids();
        if (gen.size() <= generated.size()) break;
        const int64_t next = gen.back();
        generated.push_back(next);
        group->schedule_tokens(1);
        forward((int32_t) next, pos++);
    }

    printf("\nsampled (temp=0.8 top_p=0.9 top_k=40 rep_pen=1.1 seed=42):\n  %s%s\n",
           prompt.c_str(), tok.decode(generated).c_str());
    return 0;
}
