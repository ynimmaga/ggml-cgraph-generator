// Raw end-to-end VLM wiring test (superseded by GgmlVLMPipeline for real use): image ->
// vision tower (Vulkan) -> patch embeddings -> embedding-input text decoder (Vulkan) -> greedy
// text. Both graphs are dumped llama.cpp cgraphs; llama.cpp is absent at runtime.
//
// Uses a synthetic mid-grey image, so this proves the wiring, not vision understanding.
//
// The embedding-input decoder's `inp_tokens`/GET_ROWS branch is dead -- unreachable from the
// output -- so text tokens are converted to embeddings on the host via
// GgmlModel::read_weight() against token_embd.weight, the row a GET_ROWS lookup would give.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"

#include "openvino/ggml_emitter/emitter.hpp"
#include "openvino/genai/tokenizer.hpp"

using ov::ggml_emitter::GgmlModel;

namespace {

std::vector<float> read_token_embedding(GgmlModel& decoder, int64_t token_id, size_t n_embd) {
    const size_t row_bytes = n_embd * sizeof(uint16_t);  // f16 table, confirmed at dump time
    std::vector<uint16_t> f16(n_embd);
    if (!decoder.read_weight("token_embd.weight", static_cast<size_t>(token_id) * row_bytes,
                            f16.data(), row_bytes)) {
        fprintf(stderr, "token_embd.weight lookup failed for token %lld\n", (long long) token_id);
        exit(1);
    }
    std::vector<float> out(n_embd);
    for (size_t i = 0; i < n_embd; i++) {
        out[i] = ggml_fp16_to_fp32(f16[i]);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr,
                "usage: vlm_e2e <vision.json> <mmproj.gguf> <decoder.json> <text.gguf> "
                "[prompt] [n_predict]\n");
        return 1;
    }
    const std::string vision_json = argv[1];
    const std::string mmproj = argv[2];
    const std::string decoder_json = argv[3];
    const std::string text_model = argv[4];
    const std::string prompt = argc > 5 ? argv[5] : "What is in the image?";
    const int n_predict = argc > 6 ? atoi(argv[6]) : 20;

    // ---- vision tower: one forward pass, no KV, no positions ------------------------------
    auto vision = GgmlModel::from_cgraph(vision_json, mmproj, "");
    printf("vision  backend=%s nodes=%zu\n", vision->backend_name().c_str(), vision->node_count());

    // inp_raw is [W, H, 3, 1]; the dump used 512x512 mid-grey, so reproduce that exactly --
    // the artifact's shapes are baked in and a mismatch would be silently wrong, not rejected.
    const size_t raw_elems = vision->input_size("inp_raw");
    std::vector<float> raw_img(raw_elems, 128.0f / 255.0f);
    if (!vision->write_input("inp_raw", raw_img.data(), raw_img.size() * sizeof(float))) {
        fprintf(stderr, "vision graph has no inp_raw input\n");
        return 2;
    }
    if (!vision->compute()) {
        fprintf(stderr, "vision compute failed\n");
        return 3;
    }
    const size_t n_embd = vision->logits_size();
    // The projector output is [n_embd, n_patches]; logits_size() only reports ne[0], so read
    // ne[1] off the raw tensor for the patch count.
    const size_t n_patches = vision->logits()->ne[1];
    std::vector<float> patch_embd(n_embd * n_patches);
    ggml_backend_tensor_get(vision->logits(), patch_embd.data(), 0,
                            patch_embd.size() * sizeof(float));
    printf("vision  produced %zu patches x %zu dims\n", n_patches, n_embd);

    // ---- text decoder: embedding-input graph, drives both image and text tokens -----------
    auto decoder = GgmlModel::from_cgraph(decoder_json, text_model, "");
    const int n_kv = static_cast<int>(decoder->context_size());
    printf("decoder backend=%s nodes=%zu ctx=%d embd_dim=%zu\n", decoder->backend_name().c_str(),
           decoder->node_count(), n_kv, decoder->logits_size());
    if (decoder->logits_size() != n_embd) {
        // logits_size() reports the OUTPUT vocab width for this graph, not n_embd -- kept as a
        // sanity print rather than an assert, since a real decoder's vocab and embedding widths
        // are unrelated in general (only true here by construction: SmolVLM's tied output pads
        // to n_embd... left as a print, not a hard requirement of the wiring).
    }

    ov::genai::Tokenizer tok(text_model);

    int pos = 0;
    auto step_embd = [&](const float* embd, int32_t p) {
        const int32_t oid = 0;
        const int64_t slot = p;
        decoder->write_input("embd", embd, n_embd * sizeof(float));
        decoder->write_input("inp_pos", &p, sizeof(p));
        decoder->write_input("inp_out_ids", &oid, sizeof(oid));
        for (const auto& name : decoder->input_names()) {
            if (name.rfind("inp_kv_idx", 0) == 0) {
                decoder->write_input(name, &slot, sizeof(slot));
            }
        }
        for (const auto& name : decoder->input_names()) {
            if (name.rfind("self_kq_mask", 0) != 0 && name.rfind("attn_inp_kq_mask", 0) != 0) {
                continue;
            }
            std::vector<uint16_t> mask(decoder->input_size(name), 0xFC00);  // f16 -inf
            for (int j = 0; j <= p && j < n_kv; j++) {
                mask[j] = 0;
            }
            decoder->write_input(name, mask.data(), mask.size() * sizeof(uint16_t));
        }
        if (!decoder->compute()) {
            fprintf(stderr, "decoder compute failed at pos %d\n", p);
            exit(4);
        }
    };

    // Image patches first -- each occupies one KV slot, exactly like a text token would.
    for (size_t i = 0; i < n_patches && pos < n_kv; i++) {
        step_embd(patch_embd.data() + i * n_embd, pos++);
    }
    printf("wrote %d image-patch positions into the KV cache\n", pos);

    // Then the text prompt, converted to embeddings on the host since the embd-input graph's
    // token/GET_ROWS branch is dead (see file header).
    auto ids = tok.encode(prompt).input_ids;
    std::vector<int64_t> prompt_ids(ids.data<int64_t>(), ids.data<int64_t>() + ids.get_size());
    printf("prompt: \"%s\" (%zu tokens)\n", prompt.c_str(), prompt_ids.size());

    std::vector<float> logits(decoder->logits_size());
    auto step_token = [&](int64_t token, int32_t p) -> int64_t {
        auto e = read_token_embedding(*decoder, token, n_embd);
        step_embd(e.data(), p);
        decoder->read_logits(logits.data());
        size_t best = 0;
        for (size_t i = 1; i < logits.size(); i++) {
            if (logits[i] > logits[best]) best = i;
        }
        return static_cast<int64_t>(best);
    };

    int64_t next = 0;
    for (size_t i = 0; i < prompt_ids.size(); i++) {
        next = step_token(prompt_ids[i], pos++);
    }

    std::vector<int64_t> generated;
    printf("\noutput: %s", prompt.c_str());
    fflush(stdout);
    std::string shown;
    for (int i = 0; i < n_predict && pos < n_kv; i++) {
        generated.push_back(next);
        const std::string full = tok.decode(generated);
        if (full.size() > shown.size()) {
            fwrite(full.data() + shown.size(), 1, full.size() - shown.size(), stdout);
            fflush(stdout);
            shown = full;
        }
        next = step_token(next, pos++);
    }
    printf("\n\n%d total KV positions used (%zu image + %zu prompt + generated)\n", pos, n_patches,
           prompt_ids.size());
    return 0;
}
