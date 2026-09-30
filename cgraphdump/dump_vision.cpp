// Offline tool: dump the VISION TOWER cgraph of a llama.cpp VLM (mmproj) to the same
// "ov-cgraph-v1" artifact the text-side dumper produces.
//
// A llama.cpp VLM is three pieces: (1) image preprocessing -- plain C++ in clip.cpp, not a
// graph, so it cannot be dumped; (2) vision encoder + projector -- a ggml_cgraph, captured
// here; (3) text decoder on embeddings, dumped separately in embd-input mode.
//
// mtmd_context_params carries cb_eval straight through to the clip scheduler, so the same
// public-API hook the text dumper uses works here with no patch to llama.cpp. Preprocessing
// parameters (image_size/mean/std) are ordinary GGUF metadata on the mmproj, so a runtime
// consumer reimplements (1) by reading those rather than reusing llama.cpp's C++.
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "ggml.h"
#include "llama.h"
#include "mtmd.h"

#include "cgraph_capture.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: dump_vision <mmproj.gguf> <text-model.gguf> [out.json] [px]\n");
        return 1;
    }
    const std::string mmproj = argv[1];
    const std::string text_model = argv[2];
    const std::string out = argc > 3 ? argv[3] : "vision.json";
    // Image size matters: it determines the number of patches and therefore every shape in the
    // dumped graph. The artifact is only valid for images that preprocess to this same size.
    const int px = argc > 4 ? atoi(argv[4]) : 512;

    llama_backend_init();

    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(text_model.c_str(), mp);
    if (!model) {
        fprintf(stderr, "failed to load text model %s\n", text_model.c_str());
        return 2;
    }

    cgraph::Capture cap;
    auto cp = mtmd_context_params_default();
    cp.use_gpu = false;  // topology, not performance
    cp.cb_eval = cgraph::eval_cb;
    cp.cb_eval_user_data = &cap;
    cp.print_timings = false;
    // mtmd_init_from_file runs its own internal warmup encode by default, using this same
    // cb_eval, BEFORE our explicit encode below. That pollutes the capture with a second,
    // structurally-identical graph build whose tensors can land at RECYCLED addresses from the
    // first -- corrupting our pointer-keyed node identity (a node can appear to reference
    // itself). Disable it; we only want the one real pass.
    cp.warmup = false;

    mtmd_context* mctx = mtmd_init_from_file(mmproj.c_str(), model, cp);
    if (!mctx) {
        fprintf(stderr, "mtmd_init_from_file failed for %s\n", mmproj.c_str());
        return 3;
    }

    // Discard whatever clip_init's internal flash-attention probe captured; see
    // Capture::reset(). Only the encode below should end up in the artifact.
    cap.reset();

    // A synthetic mid-grey image. Pixel VALUES are irrelevant -- we are capturing topology, and
    // the graph shape depends only on the resolution.
    std::vector<unsigned char> rgb(static_cast<size_t>(px) * px * 3, 128);
    mtmd_bitmap* bmp = mtmd_bitmap_init(px, px, rgb.data());

    mtmd_input_text txt{};
    txt.text = mtmd_default_marker();  // the media placeholder the chat template expects
    txt.text_len = std::strlen(txt.text);  // POD struct: an unset length is read as garbage
    txt.add_special = false;
    txt.parse_special = true;

    const mtmd_bitmap* bitmaps[1] = {bmp};
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    if (mtmd_tokenize(mctx, chunks, &txt, bitmaps, 1) != 0) {
        fprintf(stderr, "mtmd_tokenize failed\n");
        return 4;
    }

    // Encode every media chunk; text chunks are ignored by mtmd_encode_chunk.
    size_t n_image_chunks = 0;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        const mtmd_input_chunk* ch = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(ch) != MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            continue;
        }
        n_image_chunks++;
        if (mtmd_encode_chunk(mctx, ch) != 0) {
            fprintf(stderr, "mtmd_encode_chunk failed\n");
            return 5;
        }
        break;  // one chunk is enough for the topology
    }
    if (n_image_chunks == 0) {
        fprintf(stderr, "no image chunk produced -- is this really an mmproj?\n");
        return 6;
    }

    cgraph::write_json(cap, out, mmproj, "vision", /*n_tokens=*/0, /*n_kv=*/0);
    cgraph::report(cap, "vision encoder + projector");
    printf("image size   : %dx%d\n", px, px);
    printf("wrote %s\n", out.c_str());

    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bmp);
    mtmd_free(mctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
