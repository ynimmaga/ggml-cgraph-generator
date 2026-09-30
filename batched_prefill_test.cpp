// Batched prefill vs. sequential single-token steps: same N tokens, two graphs, compare.
// N_TOKENS>1 dump has a real [n_kv, N] mask and N-wide inp_tokens/inp_pos/kv_idx -- this checks
// that one batched forward pass gives the same result as N single-token forwards.
#include <cstdio>
#include <vector>
#include "openvino/ggml_emitter/emitter.hpp"

using ov::ggml_emitter::GgmlModel;

namespace {

void write_kv_idx(GgmlModel& m, const int64_t* slots, size_t n) {
    for (const auto& name : m.input_names()) {
        if (name.rfind("inp_kv_idx", 0) == 0) {
            m.write_input(name, slots, n * sizeof(int64_t));
        }
    }
}

int64_t argmax(const std::vector<float>& v) {
    size_t best = 0;
    for (size_t i = 1; i < v.size(); i++) {
        if (v[i] > v[best]) best = i;
    }
    return static_cast<int64_t>(best);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string model = argv[1];
    const std::vector<int32_t> toks = {791, 6864, 315, 9822};  // "The capital of France"

    // ---- sequential: N single-token forwards through the batch-1 artifact ----------------
    auto seq = GgmlModel::from_cgraph("/tmp/llama_batch1.json", model, "");
    std::vector<float> seq_logits(seq->logits_size());
    for (size_t i = 0; i < toks.size(); i++) {
        const int32_t pos = static_cast<int32_t>(i);
        const int32_t oid = 0;
        const int64_t slot = pos;
        seq->write_input("inp_tokens", &toks[i], sizeof(int32_t));
        seq->write_input("inp_pos", &pos, sizeof(pos));
        seq->write_input("inp_out_ids", &oid, sizeof(oid));
        write_kv_idx(*seq, &slot, 1);
        std::vector<uint16_t> mask(seq->input_size("attn_inp_kq_mask"), 0xFC00);
        for (int j = 0; j <= pos; j++) mask[j] = 0;
        seq->write_input("attn_inp_kq_mask", mask.data(), mask.size() * 2);
        seq->compute();
    }
    seq->read_logits(seq_logits.data());
    printf("sequential: argmax=%lld logit=%.4f\n", (long long) argmax(seq_logits), seq_logits[argmax(seq_logits)]);

    // ---- batched: one forward pass through the N_TOKENS=4 artifact ------------------------
    auto batched = GgmlModel::from_cgraph("/tmp/llama_batch4.json", model, "");
    const size_t n_kv = batched->context_size();
    batched->write_input("inp_tokens", toks.data(), toks.size() * sizeof(int32_t));
    std::vector<int32_t> pos(toks.size());
    std::vector<int64_t> slots(toks.size());
    for (size_t i = 0; i < toks.size(); i++) {
        pos[i] = static_cast<int32_t>(i);
        slots[i] = static_cast<int64_t>(i);
    }
    batched->write_input("inp_pos", pos.data(), pos.size() * sizeof(int32_t));
    write_kv_idx(*batched, slots.data(), slots.size());
    const int32_t out_id = static_cast<int32_t>(toks.size() - 1);  // last row's logits
    batched->write_input("inp_out_ids", &out_id, sizeof(out_id));

    // Row r attends to columns 0..r; layout is row-major, ne[0]=n_kv columns fastest.
    std::vector<uint16_t> mask(batched->input_size("attn_inp_kq_mask"), 0xFC00);
    for (size_t r = 0; r < toks.size(); r++) {
        for (size_t c = 0; c <= r; c++) mask[r * n_kv + c] = 0;
    }
    batched->write_input("attn_inp_kq_mask", mask.data(), mask.size() * 2);
    if (!batched->compute()) {
        fprintf(stderr, "batched compute failed\n");
        return 1;
    }
    std::vector<float> batched_logits(batched->logits_size());
    batched->read_logits(batched_logits.data());
    printf("batched:    argmax=%lld logit=%.4f\n", (long long) argmax(batched_logits),
           batched_logits[argmax(batched_logits)]);

    const bool match = argmax(seq_logits) == argmax(batched_logits);
    printf("%s\n", match ? "MATCH" : "MISMATCH");
    return match ? 0 : 2;
}
