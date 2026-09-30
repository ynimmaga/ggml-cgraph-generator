// GgmlPipeline with bucketed prefill wired in: same text as the plain single-bucket path.
// Timed with a warmup call discarded (Vulkan shader-compile / pipeline-cache cost on first
// use is real and otherwise swamps the signal) and several repeats averaged.
#include <chrono>
#include <cstdio>
#include "openvino/genai/ggml_pipeline.hpp"

using namespace ov::genai;
using Clock = std::chrono::steady_clock;

namespace {
double timed_avg(GgmlPipeline& p, const std::string& prompt, const GenerationConfig& cfg,
                 std::string& text_out, int reps) {
    double total = 0;
    for (int r = 0; r < reps; r++) {
        auto t0 = Clock::now();
        auto res = p.generate(prompt, cfg);
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        text_out = res.texts[0];
        if (r > 0) total += ms;  // discard first rep as warmup
    }
    return total / (reps - 1);
}
}  // namespace

int main(int argc, char** argv) {
    const std::string model = argv[1];
    const std::string prompt = argc > 2 ? argv[2] : "The capital of France is";
    const int reps = argc > 3 ? atoi(argv[3]) : 5;

    GenerationConfig cfg;
    cfg.max_new_tokens = 12;

    GgmlPipeline plain(std::filesystem::path("/tmp/llama_batch1.json"), model, "");
    std::string t1;
    double ms1 = timed_avg(plain, prompt, cfg, t1, reps);
    printf("plain  (bucket={1}):    avg %.1fms  %s\n", ms1, t1.c_str());

    std::vector<std::pair<size_t, std::filesystem::path>> buckets = {
        {1, "/tmp/llama_batch1.json"}, {8, "/tmp/llama_batch8.json"}};
    GgmlPipeline bucketed(buckets, model, "");
    std::string t2;
    double ms2 = timed_avg(bucketed, prompt, cfg, t2, reps);
    printf("bucketed ({1,4,8}):     avg %.1fms  %s\n", ms2, t2.c_str());

    printf("speedup: %.2fx   %s\n", ms1 / ms2, t1 == t2 ? "IDENTICAL TEXT" : "TEXT DIFFERS");
    return t1 == t2 ? 0 : 1;
}
