// Bucketed prefill: pick the smallest dumped batch size >= the prompt length, pad the rest with
// a dummy token routed to a scratch KV slot, run one forward pass. Generalizes the single
// L=3-into-N=4 case: several prompt lengths, several bucket sizes, each checked against the
// known-good sequential single-token path.
#include <cstdio>
#include <map>
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

int64_t sequential_reference(const std::string& model, const std::vector<int32_t>& toks) {
    auto seq = GgmlModel::from_cgraph("/tmp/llama_batch1.json", model, "");
    std::vector<float> out(seq->logits_size());
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
    seq->read_logits(out.data());
    return argmax(out);
}

// Real tokens in rows 0..L-1, dummy token (id 0) padded into rows L..N-1, each padding row's KV
// write pointed at a scratch slot (n_kv-1, n_kv-2, ...) so it can never collide with a slot a
// real future token will use. Real rows only ever unmask columns 0..r, so the scratch slots are
// already outside every real row's mask -- no extra masking needed.
int64_t bucketed_prefill(const std::string& model, const std::string& artifact, int bucket,
                         const std::vector<int32_t>& toks) {
    auto m = GgmlModel::from_cgraph(artifact, model, "");
    const size_t n_kv = m->context_size();
    const size_t L = toks.size();

    std::vector<int32_t> tok_buf(bucket, 0), pos_buf(bucket, 0);
    std::vector<int64_t> slot_buf(bucket, 0);
    for (size_t i = 0; i < L; i++) {
        tok_buf[i] = toks[i];
        pos_buf[i] = (int32_t) i;
        slot_buf[i] = (int64_t) i;
    }
    for (size_t i = L; i < (size_t) bucket; i++) {
        slot_buf[i] = (int64_t) n_kv - 1 - (i - L);  // distinct scratch slots per padding row
    }

    m->write_input("inp_tokens", tok_buf.data(), tok_buf.size() * 4);
    m->write_input("inp_pos", pos_buf.data(), pos_buf.size() * 4);
    write_kv_idx(*m, slot_buf.data(), slot_buf.size());
    int32_t oid = (int32_t) L - 1;
    m->write_input("inp_out_ids", &oid, 4);

    std::vector<uint16_t> mask(m->input_size("attn_inp_kq_mask"), 0xFC00);
    for (size_t r = 0; r < L; r++)
        for (size_t c = 0; c <= r; c++) mask[r * n_kv + c] = 0;
    m->write_input("attn_inp_kq_mask", mask.data(), mask.size() * 2);

    if (!m->compute()) return -1;
    std::vector<float> out(m->logits_size());
    m->read_logits(out.data());
    return argmax(out);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string model = argv[1];
    const std::map<int, std::string> buckets = {
        {1, "/tmp/llama_batch1.json"}, {4, "/tmp/llama_batch4.json"}, {8, "/tmp/llama_batch8.json"}};

    // A handful of prompts of different lengths, some landing exactly on a bucket, some needing
    // real padding.
    const std::vector<std::vector<int32_t>> prompts = {
        {791},                                   // L=1, exact fit (bucket 1)
        {791, 6864},                             // L=2, pads into bucket 4
        {791, 6864, 315},                        // L=3, pads into bucket 4 (already proven)
        {791, 6864, 315, 9822},                  // L=4, exact fit (bucket 4)
        {791, 6864, 315, 9822, 374, 279, 6864},  // L=7, pads into bucket 8
    };

    int failures = 0;
    for (const auto& toks : prompts) {
        const int L = (int) toks.size();
        int chosen = -1;
        for (const auto& kv : buckets)
            if (kv.first >= L) { chosen = kv.first; break; }
        if (chosen < 0) {
            printf("L=%d: no bucket large enough, skipped\n", L);
            continue;
        }
        const int64_t ref = sequential_reference(model, toks);
        const int64_t got = bucketed_prefill(model, buckets.at(chosen), chosen, toks);
        const bool ok = ref == got;
        failures += !ok;
        printf("L=%d -> bucket=%-2d  ref=%-6lld got=%-6lld  %s\n", L, chosen, (long long) ref,
               (long long) got, ok ? "MATCH" : "MISMATCH");
    }
    printf("%s\n", failures == 0 ? "ALL MATCH" : "SOME MISMATCHED");
    return failures == 0 ? 0 : 1;
}
