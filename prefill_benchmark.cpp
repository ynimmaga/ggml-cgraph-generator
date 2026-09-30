// Isolated prefill latency: L sequential single-token forwards vs. one batched forward through
// the smallest fitting bucket, for a range of prompt lengths. No sampler/tokenizer -- pure
// GgmlModel::compute() timing, averaged over repeats with a warmup pass first.
#include <chrono>
#include <cstdio>
#include <map>
#include <vector>
#include "openvino/ggml_emitter/emitter.hpp"

using ov::ggml_emitter::GgmlModel;
using Clock = std::chrono::steady_clock;

namespace {

void write_kv_idx(GgmlModel& m, const int64_t* slots, size_t n) {
    for (const auto& name : m.input_names())
        if (name.rfind("inp_kv_idx", 0) == 0) m.write_input(name, slots, n * sizeof(int64_t));
}

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

double time_sequential(GgmlModel& m, int L, int reps) {
    std::vector<int32_t> toks(L);
    for (int i = 0; i < L; i++) toks[i] = 100 + i;  // values irrelevant, only shape/count matters
    double total = 0;
    for (int r = 0; r < reps; r++) {
        auto t0 = Clock::now();
        for (int i = 0; i < L; i++) {
            int32_t pos = i, oid = 0;
            int64_t slot = pos;
            m.write_input("inp_tokens", &toks[i], 4);
            m.write_input("inp_pos", &pos, 4);
            m.write_input("inp_out_ids", &oid, 4);
            write_kv_idx(m, &slot, 1);
            std::vector<uint16_t> mask(m.input_size("attn_inp_kq_mask"), 0xFC00);
            for (int j = 0; j <= pos; j++) mask[j] = 0;
            m.write_input("attn_inp_kq_mask", mask.data(), mask.size() * 2);
            m.compute();
        }
        if (r > 0) total += ms_since(t0);  // discard first rep as warmup
    }
    return total / (reps - 1);
}

double time_batched(GgmlModel& m, int L, int bucket, size_t n_kv, int reps) {
    std::vector<int32_t> tok_buf(bucket, 0), pos_buf(bucket, 0);
    std::vector<int64_t> slot_buf(bucket, 0);
    for (int i = 0; i < L; i++) { tok_buf[i] = 100 + i; pos_buf[i] = i; slot_buf[i] = i; }
    for (int i = L; i < bucket; i++) slot_buf[i] = (int64_t) n_kv - 1 - (i - L);
    int32_t oid = L - 1;
    double total = 0;
    for (int r = 0; r < reps; r++) {
        auto t0 = Clock::now();
        m.write_input("inp_tokens", tok_buf.data(), tok_buf.size() * 4);
        m.write_input("inp_pos", pos_buf.data(), pos_buf.size() * 4);
        write_kv_idx(m, slot_buf.data(), slot_buf.size());
        m.write_input("inp_out_ids", &oid, 4);
        std::vector<uint16_t> mask(m.input_size("attn_inp_kq_mask"), 0xFC00);
        for (int row = 0; row < L; row++)
            for (int c = 0; c <= row; c++) mask[row * n_kv + c] = 0;
        m.write_input("attn_inp_kq_mask", mask.data(), mask.size() * 2);
        m.compute();
        if (r > 0) total += ms_since(t0);
    }
    return total / (reps - 1);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string model = argv[1];
    const int reps = argc > 2 ? atoi(argv[2]) : 6;
    const std::map<int, std::string> buckets = {
        {1, "/tmp/llama_batch1.json"}, {4, "/tmp/llama_batch4.json"}, {8, "/tmp/llama_batch8.json"}};

    std::map<int, std::shared_ptr<GgmlModel>> models;
    for (const auto& kv : buckets) models[kv.first] = GgmlModel::from_cgraph(kv.second, model, "");
    const size_t n_kv = models.at(1)->context_size();

    printf("%-6s %-8s %10s %10s %8s\n", "L", "bucket", "seq(ms)", "batch(ms)", "speedup");
    for (int L : {1, 2, 3, 4, 5, 6, 7, 8}) {
        int bucket = -1;
        for (const auto& kv : buckets) if (kv.first >= L) { bucket = kv.first; break; }
        if (bucket < 0) continue;
        double seq = time_sequential(*models.at(1), L, reps);
        double batch = time_batched(*models.at(bucket), L, bucket, n_kv, reps);
        printf("%-6d %-8d %10.2f %10.2f %7.2fx\n", L, bucket, seq, batch, seq / batch);
    }
    return 0;
}
