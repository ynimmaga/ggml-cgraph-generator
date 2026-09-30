// Consumer of the in-tree openvino_ggml_emitter component: generate text from a .gguf on a ggml
// backend using only the public OpenVINO header. No ggml headers, no llama.cpp.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "openvino/ggml_emitter/emitter.hpp"
#include "openvino/genai/tokenizer.hpp"

using ov::ggml_emitter::GgmlModel;

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: use_ov_ggml <model.gguf> [prompt] [n_predict] [n_kv]\n");
        return 1;
    }
    const std::string path = argv[1];
    const std::string prompt = argc > 2 ? argv[2] : "The capital of France is";
    const int n_predict = argc > 3 ? atoi(argv[3]) : 16;
    const int n_kv = argc > 4 ? atoi(argv[4]) : 128;

    ov::genai::Tokenizer tok(path);
    auto ids = tok.encode(prompt).input_ids;
    std::vector<int64_t> prompt_ids(ids.data<int64_t>(), ids.data<int64_t>() + ids.get_size());

    auto model = GgmlModel::build(path, n_kv);
    printf("backend: %s   nodes: %zu\n", model->backend_name().c_str(), model->node_count());

    std::vector<float> logits(model->logits_size());
    auto step = [&](int32_t token, int32_t pos) -> int32_t {
        const int32_t oid = 0;
        const int64_t slot = pos;
        model->write_input("inp_tokens", &token, 4);
        model->write_input("inp_pos", &pos, 4);
        model->write_input("inp_out_ids", &oid, 4);
        model->write_input("inp_kv_idx", &slot, 8);
        for (const auto& n : model->input_names()) {
            if (n.rfind("self_kq_mask", 0) != 0) continue;
            std::vector<uint16_t> m(model->input_size(n), 0xFC00);  // fp16 -inf
            for (int j = 0; j <= pos && j < n_kv; j++) m[j] = 0;
            model->write_input(n, m.data(), m.size() * 2);
        }
        if (!model->compute()) {
            fprintf(stderr, "compute failed at pos %d\n", pos);
            exit(2);
        }
        model->read_logits(logits.data());
        size_t best = 0;
        for (size_t i = 1; i < logits.size(); i++)
            if (logits[i] > logits[best]) best = i;
        return static_cast<int32_t>(best);
    };

    int pos = 0;
    int32_t next = 0;
    for (size_t i = 0; i < prompt_ids.size(); i++) next = step((int32_t) prompt_ids[i], pos++);

    std::vector<int64_t> gen;
    std::string shown;
    printf("\noutput: %s", prompt.c_str());
    fflush(stdout);
    for (int i = 0; i < n_predict && pos < n_kv; i++) {
        gen.push_back(next);
        const std::string full = tok.decode(gen);
        if (full.size() > shown.size()) {
            fwrite(full.data() + shown.size(), 1, full.size() - shown.size(), stdout);
            fflush(stdout);
            shown = full;
        }
        next = step(next, pos++);
    }
    printf("\n");
    return 0;
}
