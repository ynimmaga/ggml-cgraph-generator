// What happens feeding 3 real tokens into a graph dumped for N_TOKENS=4?
#include <cstdio>
#include <vector>
#include "openvino/ggml_emitter/emitter.hpp"

using ov::ggml_emitter::GgmlModel;

namespace {
void write_kv_idx(GgmlModel& m, const int64_t* slots, size_t n) {
    for (const auto& name : m.input_names())
        if (name.rfind("inp_kv_idx", 0) == 0) m.write_input(name, slots, n * sizeof(int64_t));
}
int64_t argmax(const std::vector<float>& v) {
    size_t b = 0;
    for (size_t i = 1; i < v.size(); i++) if (v[i] > v[b]) b = i;
    return (int64_t) b;
}
}  // namespace

int main(int argc, char** argv) {
    const std::string model = argv[1];
    const std::vector<int32_t> toks = {791, 6864, 315};  // "The capital of" -- only 3 tokens

    // reference: 3 sequential single-token forwards (known-good path)
    auto seq = GgmlModel::from_cgraph("/tmp/llama_batch1.json", model, "");
    std::vector<float> ref(seq->logits_size());
    for (size_t i = 0; i < toks.size(); i++) {
        int32_t pos = (int32_t) i, oid = 0;
        int64_t slot = pos;
        seq->write_input("inp_tokens", &toks[i], 4);
        seq->write_input("inp_pos", &pos, 4);
        seq->write_input("inp_out_ids", &oid, 4);
        write_kv_idx(*seq, &slot, 1);
        std::vector<uint16_t> mask(seq->input_size("attn_inp_kq_mask"), 0xFC00);
        for (int j = 0; j <= pos; j++) mask[j] = 0;
        seq->write_input("attn_inp_kq_mask", mask.data(), mask.size() * 2);
        seq->compute();
    }
    seq->read_logits(ref.data());
    printf("reference (3 sequential):      argmax=%lld\n", (long long) argmax(ref));

    // naive: batch-4 graph, only write 3 slots, leave the 4th whatever the backend gave us
    {
        auto b4 = GgmlModel::from_cgraph("/tmp/llama_batch4.json", model, "");
        const size_t n_kv = b4->context_size();
        b4->write_input("inp_tokens", toks.data(), toks.size() * 4);      // only 12 of 16 bytes
        std::vector<int32_t> pos = {0, 1, 2};
        b4->write_input("inp_pos", pos.data(), pos.size() * 4);          // only 3 of 4
        std::vector<int64_t> slots = {0, 1, 2};
        write_kv_idx(*b4, slots.data(), slots.size());                   // only 3 of 4
        int32_t oid = 2;                                                 // row 2 = last real token
        b4->write_input("inp_out_ids", &oid, 4);
        std::vector<uint16_t> mask(b4->input_size("attn_inp_kq_mask"), 0xFC00);
        for (size_t r = 0; r < 3; r++)
            for (size_t c = 0; c <= r; c++) mask[r * n_kv + c] = 0;
        b4->write_input("attn_inp_kq_mask", mask.data(), mask.size() * 2);  // row 3 left as -inf
        bool ok = b4->compute();
        std::vector<float> out(b4->logits_size());
        b4->read_logits(out.data());
        printf("naive (3-of-4, row 4 untouched): compute=%s argmax=%lld\n",
               ok ? "ok" : "FAILED", ok ? (long long) argmax(out) : -1);
    }

    // padded: row 3 gets a valid token, a scratch KV slot no real row's mask can see
    {
        auto b4 = GgmlModel::from_cgraph("/tmp/llama_batch4.json", model, "");
        const size_t n_kv = b4->context_size();
        const int64_t scratch_slot = (int64_t) n_kv - 1;  // never used by real generation here

        std::vector<int32_t> tok4 = {toks[0], toks[1], toks[2], 0};       // pad token = 0
        std::vector<int32_t> pos4 = {0, 1, 2, 0};                        // pad pos irrelevant
        std::vector<int64_t> slots4 = {0, 1, 2, scratch_slot};
        b4->write_input("inp_tokens", tok4.data(), tok4.size() * 4);
        b4->write_input("inp_pos", pos4.data(), pos4.size() * 4);
        write_kv_idx(*b4, slots4.data(), slots4.size());
        int32_t oid = 2;
        b4->write_input("inp_out_ids", &oid, 4);

        // Real rows (0..2) only ever unmask columns 0..r, so column `scratch_slot` (255) is
        // never visible to them -- no extra masking work needed for that part. Row 3's own
        // mask doesn't matter since its output is discarded.
        std::vector<uint16_t> mask(b4->input_size("attn_inp_kq_mask"), 0xFC00);
        for (size_t r = 0; r < 3; r++)
            for (size_t c = 0; c <= r; c++) mask[r * n_kv + c] = 0;
        b4->write_input("attn_inp_kq_mask", mask.data(), mask.size() * 2);

        bool ok = b4->compute();
        std::vector<float> out(b4->logits_size());
        b4->read_logits(out.data());
        printf("padded (row 4 = dummy token, scratch KV slot): compute=%s argmax=%lld\n",
               ok ? "ok" : "FAILED", ok ? (long long) argmax(out) : -1);
    }
    return 0;
}
