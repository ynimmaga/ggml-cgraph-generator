// Offline tool: dump a whisper.cpp speech-recognition model to "ov-cgraph-v1" artifacts plus a
// GGUF of its weights, so a ggml runtime can replay it with no whisper.cpp present.
//
// Outputs (all from one whisper.cpp ggml-*.bin model):
//   <prefix>.encoder.json -- log-mel -> conv stem -> audio encoder -> cross-attention K/V
//                            projections, captured as ONE graph. whisper.cpp runs these as three
//                            separately scheduled graphs; they are chained through tensors
//                            (embd_conv, embd_enc), which the pointer-identity capture keeps
//                            connected. Its observable outputs are the cache_{k,v}_cross tensors.
//                            With SPLIT_CROSS=1 the encoder artifact stops at the encoder output
//                            (embd_enc) and the K/V projections go to <prefix>.cross.json, which
//                            takes embd_enc as its input -- the split an OpenVINO-style
//                            encoder/decoder pair needs (encoder -> last_hidden_state).
//   <prefix>.decoder.json -- one text token per pass, over the full self-attention KV cache,
//                            reading cache_{k,v}_cross.
//   <prefix>.gguf         -- every weight under whisper.cpp's own tensor names, plus whisper.*
//                            metadata (hparams, special tokens, language codes, vocabulary). The
//                            cgraph loader reads weights only from GGUF, and the runtime pipeline
//                            reads everything else it needs from here too.
//
// WHY THIS COMPILES whisper.cpp IN: unlike llama.cpp/mtmd, whisper.cpp has no public cb_eval, and
// its graph builders (whisper_build_graph_*) are file-static. Including its source is the only
// way to reach them without patching it. Nothing here modifies whisper.cpp; the graphs are built
// exactly as whisper.cpp builds them and walked directly (no compute is needed for topology).
//
// THE ONE REWRITE: whisper.cpp stores a decoded token's K/V by ggml_cpy into a view whose byte
// offset bakes in the cache write position (kv_head). A shape-static artifact replayed for every
// step would then always write the same slot. Those copies are re-expressed as SET_ROWS into a
// per-layer view of the cache, indexed by a graph input "inp_kv_idx" -- the same form llama.cpp's
// own graphs use. Values, layout and the reading side are unchanged.
#include "whisper.cpp"  // NOLINT -- see above

#include "cgraph_capture.hpp"
#include "gguf.h"

namespace {

// Name every weight with its whisper.cpp name: the artifact references weights by name, and the
// GGUF written below uses the same names. whisper.cpp itself only keys them in model.tensors.
void name_weights(whisper_context& ctx) {
    for (auto& kv : ctx.model.tensors) {
        ggml_set_name(kv.second, kv.first.c_str());
    }
}

// The loader zeroes and publishes every leaf whose name starts with "cache_".
void name_caches(whisper_state& st) {
    ggml_set_name(st.kv_self.k, "cache_k_self");
    ggml_set_name(st.kv_self.v, "cache_v_self");
    ggml_set_name(st.kv_cross.k, "cache_k_cross");
    ggml_set_name(st.kv_cross.v, "cache_v_cross");
    ggml_set_name(st.kv_pad.k, "cache_k_pad");
    ggml_set_name(st.kv_pad.v, "cache_v_pad");
}

void capture_graph(cgraph::Capture& cap, ggml_cgraph* gf) {
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        cgraph::eval_cb(ggml_graph_node(gf, i), true, &cap);
    }
}

// Capture the decoder, turning each "cpy(x, view(kv_self.{k,v}, offset(layer, kv_head)))" into
// "set_rows(view(kv_self.{k,v}, layer), x, inp_kv_idx)". See the file header.
void capture_decoder(cgraph::Capture& cap, ggml_cgraph* gf, whisper_state& st, int n_tokens,
                     ggml_context* scratch, bool cross_qk) {
    const ggml_tensor* caches[2] = {st.kv_self.k, st.kv_self.v};
    auto is_self_cache = [&](const ggml_tensor* t) {
        return t && (t == caches[0] || t == caches[1]);
    };

    ggml_tensor* kv_idx = ggml_new_tensor_1d(scratch, GGML_TYPE_I64, n_tokens);
    ggml_set_name(kv_idx, "inp_kv_idx");
    ggml_set_input(kv_idx);

    const int64_t n_ctx = st.kv_self.size;
    int n_rewritten = 0, n_cross_qk = 0;
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        ggml_tensor* t = ggml_graph_node(gf, i);
        // The fixed-offset destination views exist only to feed the copies being replaced.
        if (t->op == GGML_OP_VIEW && is_self_cache(t->view_src)) {
            bool feeds_cpy_only = false;
            for (int j = i + 1; j < ggml_graph_n_nodes(gf); j++) {
                ggml_tensor* u = ggml_graph_node(gf, j);
                if (u->op == GGML_OP_CPY && u->src[1] == t) {
                    feeds_cpy_only = true;
                    break;
                }
            }
            if (feeds_cpy_only) {
                continue;
            }
        }
        if (t->op == GGML_OP_CPY && t->src[1] && is_self_cache(t->src[1]->view_src)) {
            ggml_tensor* cache = t->src[1]->view_src;
            ggml_tensor* x = t->src[0];  // [n_text_state, n_tokens], already captured
            GGML_ASSERT(x->ne[1] == n_tokens && x->ne[2] == 1 && x->ne[3] == 1);
            const int64_t row = x->ne[0];
            const size_t row_bytes = ggml_row_size(cache->type, row);
            const int64_t layer = static_cast<int64_t>(t->src[1]->view_offs / row_bytes) / n_ctx;
            ggml_tensor* layer_view = ggml_view_2d(scratch, cache, row, n_ctx, row_bytes,
                                                   static_cast<size_t>(layer * n_ctx) * row_bytes);
            ggml_format_name(layer_view, "%s_l%d", ggml_get_name(cache), static_cast<int>(layer));
            ggml_tensor* sr = ggml_set_rows(scratch, layer_view, x, kv_idx);
            ggml_format_name(sr, "%s_store_l%d", ggml_get_name(cache), static_cast<int>(layer));
            cgraph::eval_cb(layer_view, true, &cap);
            cgraph::eval_cb(sr, true, &cap);
            n_rewritten++;
            continue;
        }
        cgraph::eval_cb(t, true, &cap);
        // CROSS_QK: also emit each layer's cross-attention scores, scale * Q.K over the audio
        // frames (what word-level timestamps align on), as named graph outputs. Same Q, K and
        // scale as the flash attention they shadow.
        if (cross_qk && t->op == GGML_OP_FLASH_ATTN_EXT && t->src[1] && t->src[1]->view_src == st.kv_cross.k) {
            float scale = 1.0f;
            memcpy(&scale, t->op_params, sizeof(float));
            ggml_tensor* kq = ggml_mul_mat(scratch, t->src[1], t->src[0]);  // [n_audio_ctx, n_tokens, n_head]
            ggml_tensor* scores = ggml_scale(scratch, kq, scale);
            ggml_format_name(kq, "cross_kq_l%d", n_cross_qk);
            ggml_format_name(scores, "cross_attention_qk_scaled_scores_%d", n_cross_qk);
            ggml_set_output(scores);
            cgraph::eval_cb(kq, true, &cap);
            cgraph::eval_cb(scores, true, &cap);
            n_cross_qk++;
        }
    }
    printf("kv stores rewritten to SET_ROWS: %d\n", n_rewritten);
    if (cross_qk) {
        printf("cross-attention score outputs: %d\n", n_cross_qk);
    }
}

bool write_gguf(whisper_context& ctx, const std::string& path) {
    const auto& hp = ctx.model.hparams;
    const auto& vocab = ctx.vocab;
    gguf_context* g = gguf_init_empty();

    gguf_set_val_str(g, "general.architecture", "whisper");
    gguf_set_val_i32(g, "whisper.n_vocab", hp.n_vocab);
    gguf_set_val_i32(g, "whisper.n_audio_ctx", hp.n_audio_ctx);
    gguf_set_val_i32(g, "whisper.n_audio_state", hp.n_audio_state);
    gguf_set_val_i32(g, "whisper.n_audio_head", hp.n_audio_head);
    gguf_set_val_i32(g, "whisper.n_audio_layer", hp.n_audio_layer);
    gguf_set_val_i32(g, "whisper.n_text_ctx", hp.n_text_ctx);
    gguf_set_val_i32(g, "whisper.n_text_state", hp.n_text_state);
    gguf_set_val_i32(g, "whisper.n_text_head", hp.n_text_head);
    gguf_set_val_i32(g, "whisper.n_text_layer", hp.n_text_layer);
    gguf_set_val_i32(g, "whisper.n_mels", hp.n_mels);
    gguf_set_val_i32(g, "whisper.multilingual", vocab.is_multilingual() ? 1 : 0);

    gguf_set_val_i32(g, "whisper.token.eot", vocab.token_eot);
    gguf_set_val_i32(g, "whisper.token.sot", vocab.token_sot);
    gguf_set_val_i32(g, "whisper.token.translate", vocab.token_translate);
    gguf_set_val_i32(g, "whisper.token.transcribe", vocab.token_transcribe);
    gguf_set_val_i32(g, "whisper.token.solm", vocab.token_solm);
    gguf_set_val_i32(g, "whisper.token.prev", vocab.token_prev);
    gguf_set_val_i32(g, "whisper.token.nosp", vocab.token_nosp);
    gguf_set_val_i32(g, "whisper.token.not", vocab.token_not);
    gguf_set_val_i32(g, "whisper.token.beg", vocab.token_beg);

    // Language codes in id order; the token for language i is sot + 1 + i.
    if (vocab.is_multilingual()) {
        std::vector<const char*> langs;
        for (int i = 0; i < vocab.num_languages() && i <= whisper_lang_max_id(); i++) {
            langs.push_back(whisper_lang_str(i));
        }
        gguf_set_arr_str(g, "whisper.languages", langs.data(), static_cast<int>(langs.size()));
    }

    // Vocabulary as raw bytes + offsets: GPT-2 byte-level tokens are arbitrary byte strings
    // (including NUL), which a GGUF string array -- written from NUL-terminated C strings --
    // cannot carry.
    std::vector<uint8_t> bytes;
    std::vector<int32_t> offsets{0};
    for (int i = 0; i < hp.n_vocab; i++) {
        auto it = vocab.id_to_token.find(i);
        if (it != vocab.id_to_token.end()) {
            bytes.insert(bytes.end(), it->second.begin(), it->second.end());
        }
        offsets.push_back(static_cast<int32_t>(bytes.size()));
    }
    gguf_set_arr_data(g, "whisper.vocab.bytes", GGUF_TYPE_UINT8, bytes.data(), bytes.size());
    gguf_set_arr_data(g, "whisper.vocab.offsets", GGUF_TYPE_INT32, offsets.data(), offsets.size());

    for (auto& kv : ctx.model.tensors) {
        ggml_tensor* t = kv.second;
        // gguf_write_to_file reads t->data directly, which is only the raw tensor for a plain
        // host buffer (not a repacked CPU "extra" buffer, not device memory).
        if (!t->buffer || !ggml_backend_buffer_is_host(t->buffer) ||
            strcmp(ggml_backend_buffer_name(t->buffer), "CPU") != 0) {
            fprintf(stderr, "weight %s is not in a plain CPU buffer (%s); cannot write GGUF\n",
                    kv.first.c_str(), t->buffer ? ggml_backend_buffer_name(t->buffer) : "none");
            gguf_free(g);
            return false;
        }
        gguf_add_tensor(g, t);
    }
    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: dump_whisper <ggml-model.bin> [out_prefix]\n"
                        "writes <out_prefix>.encoder.json, .decoder.json, .gguf\n");
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string prefix = argc > 2 ? argv[2] : "whisper";
    const char* split_env = getenv("SPLIT_CROSS");
    const bool split_cross = split_env && split_env[0] == '1';

    auto cp = whisper_context_default_params();
    // Weights must land in host memory to be written to GGUF; the graph topology does not depend
    // on the backend.
    cp.use_gpu = false;
    // Flash attention: decoder reads K and V row-major, which is what makes the SET_ROWS rewrite
    // a plain row store (the non-FA path keeps V transposed).
    cp.flash_attn = true;
    whisper_context* ctx = whisper_init_from_file_with_params_no_state(model_path.c_str(), cp);
    if (!ctx) {
        fprintf(stderr, "failed to load %s\n", model_path.c_str());
        return 2;
    }
    whisper_state* st = whisper_init_state(ctx);
    if (!st) {
        fprintf(stderr, "whisper_init_state failed\n");
        return 3;
    }
    name_weights(*ctx);
    name_caches(*st);
    const auto& hp = ctx->model.hparams;

    // ---- encoder: conv stem + encoder + cross-attention K/V, chained into one artifact ----
    {
        cgraph::Capture cap;
        cap.reset_on_im2col = false;  // two convolutions in one pass
        // Each builder carves its graph out of its own scheduler's meta buffer, so tensors of an
        // earlier graph stay valid (and keep distinct addresses) while the next one is built.
        capture_graph(cap, whisper_build_graph_conv(*ctx, *st));
        capture_graph(cap, whisper_build_graph_encoder(*ctx, *st));
        if (split_cross) {
            // Stop at the encoder output (embd_enc, [n_state, n_audio_ctx]); the cross K/V
            // projection becomes its own artifact that takes embd_enc as an input.
            const std::string out = prefix + ".encoder.json";
            cgraph::write_json(cap, out, model_path, "whisper-encoder", /*n_tokens=*/0, /*n_kv=*/0);
            cgraph::report(cap, "whisper encoder (conv + encoder)");
            printf("wrote %s\n\n", out.c_str());

            // The cross graph reads embd_enc, an op node of the encoder graph; build it over a
            // fresh input leaf of the same shape instead.
            ggml_context* leaf_ctx = ggml_init({ggml_tensor_overhead(), nullptr, /*no_alloc=*/true});
            ggml_tensor* enc_out = st->embd_enc;
            ggml_tensor* embd_enc = ggml_new_tensor(leaf_ctx, enc_out->type, GGML_MAX_DIMS, enc_out->ne);
            ggml_set_name(embd_enc, "embd_enc");
            ggml_set_input(embd_enc);
            st->embd_enc = embd_enc;
            cgraph::Capture cross;
            capture_graph(cross, whisper_build_graph_cross(*ctx, *st));
            st->embd_enc = enc_out;
            ggml_free(leaf_ctx);
            const std::string cross_out = prefix + ".cross.json";
            cgraph::write_json(cross, cross_out, model_path, "whisper-cross", /*n_tokens=*/0, /*n_kv=*/0);
            cgraph::report(cross, "whisper cross-attention K/V");
            printf("wrote %s\n\n", cross_out.c_str());
        } else {
            capture_graph(cap, whisper_build_graph_cross(*ctx, *st));
            const std::string out = prefix + ".encoder.json";
            cgraph::write_json(cap, out, model_path, "whisper-encoder", /*n_tokens=*/0, /*n_kv=*/0);
            cgraph::report(cap, "whisper encoder (conv + encoder + cross K/V)");
            printf("wrote %s\n\n", out.c_str());
        }
        printf("mel input    : [%d, %d]\n", 2 * hp.n_audio_ctx, hp.n_mels);
    }

    // ---- decoder: one token per pass over the whole self-attention cache ----
    {
        const int n_tokens = 1;
        whisper_batch batch = whisper_batch_init(n_tokens, 1);
        batch.n_tokens = n_tokens;
        batch.token[0] = ctx->vocab.token_sot;
        batch.pos[0] = 0;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        // worst_case: attend over all kv_self.size cells (the runtime masks the unused ones), the
        // shape-static equivalent of whisper.cpp's per-step n_kv.
        ggml_cgraph* gf = whisper_build_graph_decoder(*ctx, *st, batch, false, /*worst_case=*/true);

        ggml_context* scratch =
            ggml_init({ggml_tensor_overhead() * 512, nullptr, /*no_alloc=*/true});
        cgraph::Capture cap;
        const char* cross_qk_env = getenv("CROSS_QK");
        capture_decoder(cap, gf, *st, n_tokens, scratch, cross_qk_env && cross_qk_env[0] == '1');
        const std::string out = prefix + ".decoder.json";
        cgraph::write_json(cap, out, model_path, "whisper-decoder", n_tokens,
                           static_cast<int>(st->kv_self.size));
        cgraph::report(cap, "whisper decoder (1 token)");
        printf("kv cells     : %d (positions < n_text_ctx = %d)\n", st->kv_self.size,
               hp.n_text_ctx);
        printf("wrote %s\n\n", out.c_str());
        ggml_free(scratch);
        whisper_batch_free(batch);
    }

    const std::string gguf_out = prefix + ".gguf";
    if (!write_gguf(*ctx, gguf_out)) {
        fprintf(stderr, "failed to write %s\n", gguf_out.c_str());
        return 4;
    }
    printf("wrote %s (%zu weights)\n", gguf_out.c_str(), ctx->model.tensors.size());

    whisper_free_state(st);
    whisper_free(ctx);
    return 0;
}
