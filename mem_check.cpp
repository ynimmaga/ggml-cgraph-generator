#include <cstdio>
#include <fstream>
#include <string>
#include "openvino/genai/ggml_pipeline.hpp"
using namespace ov::genai;

long rss_kb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) return atol(line.c_str() + 6);
    }
    return -1;
}

int main(int argc, char** argv) {
    const std::string model = argv[1];
    printf("baseline RSS: %ld MB\n", rss_kb() / 1024);
    {
        GgmlPipeline plain(std::filesystem::path("/tmp/llama_batch1.json"), model, "");
        GenerationConfig cfg; cfg.max_new_tokens = 1;
        plain.generate("hi", cfg);
        printf("1 bucket  RSS: %ld MB\n", rss_kb() / 1024);
    }
    {
        std::vector<std::pair<size_t, std::filesystem::path>> b = {
            {1, "/tmp/llama_batch1.json"}, {4, "/tmp/llama_batch4.json"}, {8, "/tmp/llama_batch8.json"}};
        GgmlPipeline bucketed(b, model, "");
        GenerationConfig cfg; cfg.max_new_tokens = 1;
        bucketed.generate("hi", cfg);
        printf("3 buckets RSS: %ld MB\n", rss_kb() / 1024);
    }
    return 0;
}
