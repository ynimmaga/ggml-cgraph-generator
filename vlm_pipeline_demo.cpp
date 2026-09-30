// GgmlVLMPipeline: the real pipeline class, not the raw wiring test. Same sampler/streaming
// stack as GgmlPipeline, now driving image + text.
#include <cstdio>
#include <cstring>
#include "openvino/genai/ggml_vlm_pipeline.hpp"

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: vlm_pipeline_demo <vision.json> <mmproj.gguf> <decoder.json> "
                        "<text.gguf> [prompt] [raw_rgb_file H W]\n");
        return 1;
    }
    ov::genai::GgmlVLMPipeline pipe(argv[1], argv[2], argv[3], argv[4]);
    printf("backend: %s   ctx: %zu\n", pipe.backend_name().c_str(), pipe.context_size());

    ov::Tensor image;
    if (argc > 8) {
        // A real decoded image: raw interleaved HWC uint8 RGB, e.g. from
        //   PIL.Image.open(path).convert("RGB").tobytes()
        const int H = atoi(argv[7]), W = atoi(argv[8]);
        image = ov::Tensor(ov::element::u8, ov::Shape{(size_t) H, (size_t) W, 3});
        FILE* f = fopen(argv[6], "rb");
        fread(image.data<uint8_t>(), 1, image.get_byte_size(), f);
        fclose(f);
        printf("loaded %dx%d image from %s\n", H, W, argv[6]);
    } else {
        // No real image given: synthetic mid-grey placeholder, exercises the pipeline
        // machinery (sampler, streaming, KV bookkeeping), not visual understanding.
        image = ov::Tensor(ov::element::u8, ov::Shape{384, 384, 3});
        std::memset(image.data<uint8_t>(), 128, image.get_byte_size());
    }

    ov::genai::GenerationConfig cfg;
    cfg.max_new_tokens = 20;
    auto r = pipe.generate(argc > 5 ? argv[5] : "What is in the image?", image, cfg);
    printf("output: %s\n", r.texts[0].c_str());

    printf("\n[streaming] ");
    ov::genai::GenerationConfig cfg2;
    cfg2.max_new_tokens = 15;
    pipe.generate("Describe this.", image, cfg2, [](std::string s) {
        printf("%s", s.c_str());
        fflush(stdout);
        return ov::genai::StreamingStatus::RUNNING;
    });
    printf("\n");
    return 0;
}
