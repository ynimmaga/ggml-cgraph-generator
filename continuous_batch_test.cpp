// Real multi-sequence continuous batching: two INDEPENDENT sequences occupy two of four decode
// slots in ONE fixed-size graph, each with its own KV partition and mask, sampling its own next
// token from ONE joint forward pass. Checks each sequence's result against running it alone --
// the crux question is whether co-batching unrelated sequences cross-contaminates either one.
#include <cstdio>
#include <vector>
#include "openvino/ggml_emitter/emitter.hpp"

using ov::ggml_emitter::GgmlModel;

namespace {

void write_kv_idx(GgmlModel& m, const int64_t* slots, size_t n) {
    for (const auto& name : m.input_names())
        if (name.rfind("inp_kv_idx", 0) == 0) m.write_input(name, slots, n * sizeof(int64_t));
}
int64_t argmax(const float* v, size_t n) {
    size_t b = 0;
    for (size_t i = 1; i < n; i++) if (v[i] > v[b]) b = i;
    return (int64_t) b;
}

// Independent reference: run one sequence alone, sequentially, through the trusted batch-1
// artifact. Returns the argmax after processing all of `toks`.
int64_t sequential_reference(const std::string& model, const std::vector<int32_t>& toks) {
    auto m = GgmlModel::from_cgraph("/tmp/llama_batch1.json", model, "");
    std::vector<float> out(m->logits_size());
    for (size_t i = 0; i < toks.size(); i++) {
        int32_t pos = (int32_t) i, oid = 0;
        int64_t slot = pos;
        m->write_input("inp_tokens", &toks[i], 4);
        m->write_input("inp_pos", &pos, 4);
        m->write_input("inp_out_ids", &oid, 4);
        write_kv_idx(*m, &slot, 1);
        std::vector<uint16_t> mask(m->input_size("attn_inp_kq_mask"), 0xFC00);
        for (int j = 0; j <= pos; j++) mask[j] = 0;
        m->write_input("attn_inp_kq_mask", mask.data(), mask.size() * 2);
        m->compute();
    }
    m->read_logits(out.data());
    return argmax(out.data(), out.size());
}

}  // namespace

int main(int argc, char** argv) {
    const std::string model = argv[1];
    const int B = 4;               // decode slots in the dumped graph
    const int part = 64;           // KV positions reserved per slot (256 / 4)

    // Two unrelated prompts, occupying slot 0 and slot 1. Slots 2-3 stay empty all along.
    const std::vector<int32_t> seqA = {791, 6864, 315};        // "The capital of"
    const std::vector<int32_t> seqB = {791, 6864, 315, 9822};  // "The capital of France" (longer)

    printf("reference (A alone): argmax=%lld\n", (long long) sequential_reference(model, seqA));
    printf("reference (B alone): argmax=%lld\n", (long long) sequential_reference(model, seqB));

    auto cb = GgmlModel::from_cgraph("/tmp/llama_cb4.json", model, "");
    const size_t n_kv = cb->context_size();
    const size_t n_vocab = cb->output_size() / B;

    auto step = [&](const std::vector<int32_t>& tok_row, const std::vector<int32_t>& pos_row,
                    const std::vector<bool>& active) {
        std::vector<int32_t> tok_buf(B, 0), pos_buf(B, 0);
        std::vector<int64_t> slot_buf(B, 0);
        std::vector<uint16_t> mask(n_kv * B, 0xFC00);
        for (int s = 0; s < B; s++) {
            if (active[s]) {
                tok_buf[s] = tok_row[s];
                pos_buf[s] = pos_row[s];
                slot_buf[s] = s * part + pos_row[s];  // this slot's own KV partition
                // Row s attends only within its own partition, up to its own position -- never
                // slot t!=s's columns, so co-batched sequences cannot see each other's KV.
                for (int j = 0; j <= pos_row[s]; j++) mask[s * n_kv + s * part + j] = 0;
            } else {
                slot_buf[s] = (int64_t) n_kv - 1 - s;  // scratch, outside every partition
            }
        }
        cb->write_input("inp_tokens", tok_buf.data(), tok_buf.size() * 4);
        cb->write_input("inp_pos", pos_buf.data(), pos_buf.size() * 4);
        write_kv_idx(*cb, slot_buf.data(), slot_buf.size());
        std::vector<int32_t> out_ids = {0, 1, 2, 3};  // request every row's logits
        cb->write_input("inp_out_ids", out_ids.data(), out_ids.size() * 4);
        cb->write_input("attn_inp_kq_mask", mask.data(), mask.size() * 2);
        cb->compute();
    };

    // Prefill A into slot 0 and B into slot 1, one token per pass, in lockstep (co-batched from
    // the very first token -- this IS the continuous-batching case, not sequential-then-joined).
    // A and B have different lengths, so each sequence's "next-token" result must be captured
    // right after ITS OWN last active step -- once a sequence is shorter, later steps' output
    // for its row is a discarded padding row, not that sequence's prediction.
    const size_t L = std::max(seqA.size(), seqB.size());
    std::vector<float> logits(n_vocab * B);
    int64_t gotA = -1, gotB = -1;
    for (size_t i = 0; i < L; i++) {
        std::vector<int32_t> tok = {0, 0, 0, 0};
        std::vector<int32_t> pos = {(int32_t) i, (int32_t) i, 0, 0};
        std::vector<bool> active = {i < seqA.size(), i < seqB.size(), false, false};
        if (active[0]) tok[0] = seqA[i];
        if (active[1]) tok[1] = seqB[i];
        step(tok, pos, active);
        cb->read_output(logits.data());
        if (active[0]) gotA = argmax(logits.data() + 0 * n_vocab, n_vocab);
        if (active[1]) gotB = argmax(logits.data() + 1 * n_vocab, n_vocab);
    }
    printf("co-batched slot 0 (A): argmax=%lld\n", (long long) gotA);
    printf("co-batched slot 1 (B): argmax=%lld\n", (long long) gotB);

    const int64_t refA = sequential_reference(model, seqA);
    const int64_t refB = sequential_reference(model, seqB);
    bool ok = gotA == refA && gotB == refB;
    printf("%s\n", ok ? "NO CROSS-CONTAMINATION -- MATCH" : "MISMATCH");

    // The other defining property of CONTINUOUS batching: A finished and its slot (0) is free.
    // Admit a brand-new sequence C into that same, already-written-to partition -- without
    // waiting for B (still running in slot 1) to finish. This step interleaves an ADMISSION
    // (C, fresh position 0) with an ONGOING sequence (B, continuing at position 4).
    const std::vector<int32_t> seqC = {9906};  // a single fresh token, unrelated to A or B
    std::vector<int32_t> tok = {seqC[0], 100, 0, 0};
    std::vector<int32_t> pos = {0, (int32_t) L, 0, 0};  // C starts at 0; B continues from L
    std::vector<bool> active = {true, true, false, false};
    step(tok, pos, active);
    cb->read_output(logits.data());
    const int64_t gotC = argmax(logits.data() + 0 * n_vocab, n_vocab);
    const int64_t refC = sequential_reference(model, seqC);  // C alone, from a clean cache
    printf("admitted C into A's freed slot 0 (partition previously held A's data): argmax=%lld\n",
           (long long) gotC);
    printf("reference (C alone, fresh):                                            argmax=%lld\n",
           (long long) refC);
    const bool admission_ok = gotC == refC;
    printf("%s\n", admission_ok ? "ADMISSION INTO REUSED SLOT -- MATCH (no leftover contamination)"
                                : "ADMISSION MISMATCH");
    ok = ok && admission_ok;

    printf("\n%s\n", ok ? "ALL CHECKS PASS" : "SOME CHECK FAILED");
    return ok ? 0 : 1;
}
