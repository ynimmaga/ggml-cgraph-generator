#!/usr/bin/env python3
"""Synthesize one tiny .gguf per llama.cpp architecture, driven by gguf-py's tables.

The point is to test OpenVINO's GGUF frontend against every architecture llama.cpp knows
about WITHOUT downloading a single model. gguf-py already carries, for each architecture,
the exact set of tensors it has (MODEL_TENSORS) and the name template for each
(TENSOR_NAMES). That is precisely the input OV's DecoderConfig auto-detection consumes --
it decides QK-norm / fused-QKV / MoE / biases / post-norms from which layer-0 weights exist.

So: expand the table into a real (tiny) GGUF, hand it to build_ggml_graph_from_gguf, and
see whether the builder produces a graph. No llama.cpp C++ is involved, and nothing from
llama.cpp ends up in the shipped product -- this is a build/test-time generator only.

gguf-py is MIT (llama.cpp); used here as a data source at test time.
"""
import os
import sys
import numpy as np
from gguf.constants import MODEL_ARCH_NAMES, MODEL_TENSORS, TENSOR_NAMES
from gguf import GGUFWriter, GGMLQuantizationType

# Deliberately tiny -- we are probing topology, not numerics.
N_LAYER, N_EMBD, N_HEAD, N_HEAD_KV, N_FF, N_VOCAB, N_EXPERT = 2, 64, 4, 2, 128, 32, 4
HEAD_SIZE = N_EMBD // N_HEAD           # 16
N_EMBD_K = N_HEAD_KV * HEAD_SIZE       # 32


# Tensors we deliberately do not model but which do NOT change decoder topology, so omitting
# them still yields a faithful probe. Calibrated against ground truth: llama, minicpm, smollm3
# and mistral3 are all verified-working in arch_registry.cpp and all carry attn_rot_embd.
#   attn_rot_embd - legacy rope tensor, unused by modern builders and absent from real files
#   nextn.*       - optional multi-token-prediction head bolted on beside the base decoder
# Anything else (ssm_*, time_mix_*, channel_mix_*, attn_gate) DOES change topology: omitting it
# silently turns a hybrid/recurrent model into a plain transformer, so those stay disqualifying.
BENIGN_UNMODELLED = ("attn_rot_embd", "nextn.")


def is_benign(name: str) -> bool:
    return any(tok in name for tok in BENIGN_UNMODELLED)


def shape_for(base: str):
    """GGUF shape for a tensor, from its name suffix. Returns None to skip.

    GGUF stores 2-D weights as [in_features, out_features]. Norms and biases are 1-D.
    """
    b = base
    # --- MoE expert stacks (3-D: [in, out, n_expert]) ---
    if b.endswith("_exps"):
        if "down" in b:
            return [N_FF, N_EMBD, N_EXPERT]
        return [N_EMBD, N_FF, N_EXPERT]
    if b.endswith("_exp_probs_b") or b.endswith("gate_inp"):
        return [N_EMBD, N_EXPERT] if b.endswith("gate_inp") else [N_EXPERT]

    # --- biases and norms are 1-D, sized by what they follow ---
    is_bias = b.endswith("_b") or b.endswith(".bias")
    is_norm = "norm" in b

    if is_norm or is_bias:
        if "attn_q_norm" in b or "attn_k_norm" in b:
            return [HEAD_SIZE]
        if "attn_k" in b or "attn_v" in b:
            return [N_EMBD_K]
        if "ffn_gate" in b or "ffn_up" in b:
            return [N_FF]
        return [N_EMBD]

    # --- 2-D projections ---
    if b.endswith("token_embd") or b.endswith("output"):
        return [N_EMBD, N_VOCAB]
    if "attn_qkv" in b:
        return [N_EMBD, N_EMBD + 2 * N_EMBD_K]
    if "attn_q" in b:
        return [N_EMBD, N_EMBD]
    if "attn_k" in b or "attn_v" in b:
        return [N_EMBD, N_EMBD_K]
    if "attn_out" in b:
        return [N_EMBD, N_EMBD]
    if "ffn_down" in b:
        return [N_FF, N_EMBD]
    if "ffn_gate" in b or "ffn_up" in b:
        return [N_EMBD, N_FF]
    if "rope_freqs" in b or "rope_factors" in b:
        return [HEAD_SIZE // 2]
    if "attn_sinks" in b:
        return [N_HEAD]
    return None  # unknown -> skip; the probe reports what was missing


def build(arch_enum, arch_name, outdir):
    tensors = MODEL_TENSORS.get(arch_enum, [])
    if not tensors:
        return None, "no tensor table"

    path = os.path.join(outdir, f"{arch_name.replace('/', '_')}.gguf")
    w = GGUFWriter(path, arch_name)

    w.add_block_count(N_LAYER)
    w.add_context_length(128)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_layer_norm_eps(1e-5)
    w.add_rope_dimension_count(HEAD_SIZE)
    w.add_rope_freq_base(10000.0)
    w.add_key_length(HEAD_SIZE)
    w.add_value_length(HEAD_SIZE)
    w.add_expert_count(N_EXPERT)
    w.add_expert_used_count(2)
    w.add_expert_feed_forward_length(N_FF)

    # Minimal vocab so the tokenizer metadata path has something to read.
    w.add_tokenizer_model("gpt2")
    w.add_token_list([f"t{i}" for i in range(N_VOCAB)])
    w.add_token_types([1] * N_VOCAB)

    skipped = []
    for t in tensors:
        tmpl = TENSOR_NAMES.get(t)
        if tmpl is None:
            continue
        per_layer = "{bid}" in tmpl
        for il in range(N_LAYER if per_layer else 1):
            name = tmpl.format(bid=il)
            shp = shape_for(name.replace(f"blk.{il}.", ""))
            if shp is None:
                if not is_benign(name):
                    skipped.append(name)  # topology-changing omission -> disqualifies the probe
                continue
            # GGUF/numpy shapes are reversed relative to the ne[] order above.
            w.add_tensor(name + ".weight",
                         np.zeros(list(reversed(shp)), dtype=np.float32),
                         raw_dtype=GGMLQuantizationType.F32)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path, skipped


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "synth"
    os.makedirs(outdir, exist_ok=True)
    ok = 0
    # Fidelity manifest. shape_for() returns None for tensors it does not model (SSM state,
    # RWKV time-mix, vision towers, ...). Those get omitted, which QUIETLY TURNS A HYBRID
    # MODEL INTO A PLAIN TRANSFORMER -- and then the builder "succeeds" on a topology the real
    # architecture does not have. So record every omission: any architecture with a non-empty
    # skip list is INCONCLUSIVE, not supported, no matter what the probe returns.
    with open(os.path.join(outdir, "skipped.tsv"), "w") as mf:
        for arch_enum, arch_name in MODEL_ARCH_NAMES.items():
            try:
                path, skipped = build(arch_enum, arch_name, outdir)
                if path:
                    ok += 1
                    mf.write(f"{arch_name}\t{len(skipped)}\t{','.join(skipped)}\n")
                    print(f"GEN  {arch_name}"
                          + (f"  (skipped {len(skipped)} unmodelled tensors)" if skipped else ""))
                else:
                    print(f"SKIP {arch_name}: {skipped}")
            except Exception as e:
                print(f"FAIL {arch_name}: {type(e).__name__}: {e}")
    print(f"\ngenerated {ok}/{len(MODEL_ARCH_NAMES)} synthetic models in {outdir}/")


if __name__ == "__main__":
    main()
