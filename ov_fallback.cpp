// Whole-model fallback: try to run the model on the OpenVINO backend; if anything in the OV
// path fails (device init, model load, graph compile, or the first decode), tear it down and
// re-run the entire model on Vulkan.
//
// Scope note: this catches FAILURES that surface as a C++ exception or a non-success
// ggml_status. It cannot catch a ggml_abort()/GGML_ASSERT inside a backend, which kills the
// process -- see the report for why probing in a subprocess is the only way to cover that.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "llama.h"

namespace {

ggml_backend_dev_t find_device(const char * prefix) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (strncmp(ggml_backend_dev_name(d), prefix, strlen(prefix)) == 0) {
            return d;
        }
    }
    return nullptr;
}

struct Runner {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;

    void free_all() {
        if (ctx) {
            llama_free(ctx);
            ctx = nullptr;
        }
        if (model) {
            llama_model_free(model);
            model = nullptr;
        }
    }

    // Load + context + one warmup decode on `dev`. Returns false on any failure.
    bool try_device(const std::string & path, ggml_backend_dev_t dev, int n_ctx) {
        // Fault injection, so the fallback path is testable without waiting for a real OV
        // failure: OV_FALLBACK_FORCE_FAIL=1 makes the OpenVINO attempt report failure.
        if (getenv("OV_FALLBACK_FORCE_FAIL") &&
            strncmp(ggml_backend_dev_name(dev), "OPENVINO", 8) == 0) {
            fprintf(stderr, "  [injected] forcing OpenVINO failure\n");
            return false;
        }
        ggml_backend_dev_t devs[2] = {dev, nullptr};

        auto mp = llama_model_default_params();
        mp.devices = devs;
        mp.n_gpu_layers = 999;

        try {
            model = llama_model_load_from_file(path.c_str(), mp);
        } catch (const std::exception & e) {
            fprintf(stderr, "  load threw: %s\n", e.what());
            model = nullptr;
        }
        if (!model) {
            fprintf(stderr, "  model load failed\n");
            return false;
        }

        auto cp = llama_context_default_params();
        cp.n_ctx = n_ctx;
        cp.n_batch = n_ctx;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

        try {
            ctx = llama_init_from_model(model, cp);
        } catch (const std::exception & e) {
            fprintf(stderr, "  context init threw: %s\n", e.what());
            ctx = nullptr;
        }
        if (!ctx) {
            fprintf(stderr, "  context init failed\n");
            free_all();
            return false;
        }

        // Warmup decode: this is what actually exercises graph compile + execution, so it is
        // the step that catches a backend that loads fine but cannot run the graph.
        const llama_vocab * vocab = llama_model_get_vocab(model);
        llama_token bos = llama_vocab_bos(vocab);
        if (bos < 0) {
            bos = 1;
        }
        llama_batch b = llama_batch_get_one(&bos, 1);
        int rc = -1;
        try {
            rc = llama_decode(ctx, b);
        } catch (const std::exception & e) {
            fprintf(stderr, "  warmup decode threw: %s\n", e.what());
            rc = -1;
        }
        if (rc != 0) {
            fprintf(stderr, "  warmup decode failed (rc=%d)\n", rc);
            free_all();
            return false;
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        return true;
    }
};

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: ov_fallback <model.gguf> [n_predict]\n");
        return 1;
    }
    const std::string path = argv[1];
    const int n_predict = argc > 2 ? atoi(argv[2]) : 16;

    llama_backend_init();
    ggml_backend_load_all();

    ggml_backend_dev_t ov = find_device("OPENVINO");
    ggml_backend_dev_t vk = find_device("Vulkan");
    printf("devices: OV=%s  Vulkan=%s\n", ov ? ggml_backend_dev_name(ov) : "(none)",
           vk ? ggml_backend_dev_name(vk) : "(none)");

    Runner r;
    const char * chosen = nullptr;

    if (ov && r.try_device(path, ov, 512)) {
        chosen = "OpenVINO";
    } else {
        printf("OpenVINO path unavailable -> falling back to Vulkan for the WHOLE model\n");
        if (vk && r.try_device(path, vk, 512)) {
            chosen = "Vulkan";
        }
    }
    if (!chosen) {
        fprintf(stderr, "both backends failed\n");
        return 2;
    }
    printf("RUNNING ON: %s\n", chosen);

    // Greedy generation from a fixed prompt so the two paths are comparable.
    const llama_vocab * vocab = llama_model_get_vocab(r.model);
    const std::string prompt = "The capital of France is";
    std::vector<llama_token> toks(64);
    const int n_tok = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), toks.data(),
                                     (int) toks.size(), true, false);
    if (n_tok <= 0) {
        fprintf(stderr, "tokenize failed\n");
        return 3;
    }
    toks.resize(n_tok);

    llama_batch b = llama_batch_get_one(toks.data(), n_tok);
    if (llama_decode(r.ctx, b) != 0) {
        fprintf(stderr, "prefill failed\n");
        return 4;
    }

    printf("output: %s", prompt.c_str());
    fflush(stdout);
    int n_past = n_tok;
    for (int i = 0; i < n_predict; i++) {
        const float * logits = llama_get_logits_ith(r.ctx, -1);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        int best = 0;
        for (int t = 1; t < n_vocab; t++) {
            if (logits[t] > logits[best]) {
                best = t;
            }
        }
        if (llama_vocab_is_eog(vocab, best)) {
            break;
        }
        char buf[256];
        const int nc = llama_token_to_piece(vocab, best, buf, sizeof(buf), 0, true);
        if (nc > 0) {
            fwrite(buf, 1, nc, stdout);
            fflush(stdout);
        }
        llama_token next = best;
        llama_batch nb = llama_batch_get_one(&next, 1);
        if (llama_decode(r.ctx, nb) != 0) {
            fprintf(stderr, "\ndecode failed at %d\n", i);
            break;
        }
        n_past++;
    }
    printf("\n");

    r.free_all();
    llama_backend_free();
    return 0;
}
