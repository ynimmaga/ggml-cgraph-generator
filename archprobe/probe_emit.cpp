// Op-gap probe: which GGML ops does an architecture need that the ggml emitter cannot map?
//
// probe_arch.cpp only checks that OpenVINO's builder assembles a graph. Granite passed that and
// still dumped core, because the gap was in the EMITTER (no GGML_OP_SCALE handler -> nullptr ->
// segfault). This probe closes that hole so a newly added llama.cpp architecture reports its
// missing op mappings BY NAME instead of crashing.
//
// It records the op inventory the builder emits and diffs it against handled_ops(). No ggml
// tensors are created, deliberately: the synthetic models carry placeholder shapes, and ggml
// asserts on shape consistency during graph construction, which would mask the op-level answer
// we actually want. This therefore detects "op not handled" -- not shape or numerical problems.
#include <cstdio>
#include <exception>
#include <map>
#include <set>
#include <string>

#include "../ggml_emitter.hpp"
#include "builder/gguf_builder.hpp"

using namespace ov::frontend::gguf;

namespace {

// Records op types without translating them, so no ggml shape validation is involved.
class RecordingEmitter : public GraphEmitter {
public:
    RecordingEmitter(std::unordered_map<std::string, ov::Tensor> & w,
                     std::unordered_map<std::string, GgufTensorType> & q,
                     std::string arch,
                     std::map<std::string, int> * seen)
        : GraphEmitter(w, q, std::move(arch)), m_seen(seen) {}

    std::string add_op(const std::string & op_type,
                       const std::string & name,
                       const std::vector<std::string> &,
                       const ov::PartialShape & out_shape,
                       ov::element::Type out_type,
                       int,
                       std::map<std::string, ov::Any>) override {
        record_tensor_meta(name, out_shape, out_type);  // blocks/ query shape_of_tensor()
        (*m_seen)[op_type]++;
        return name;
    }

private:
    std::map<std::string, int> * m_seen;
};

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: probe_emit <arch-name> <model.gguf>\n");
        return 1;
    }
    const std::string arch = argv[1];
    std::map<std::string, int> seen;

    auto factory = [&](std::unordered_map<std::string, ov::Tensor> & w,
                       std::unordered_map<std::string, GgufTensorType> & q,
                       const std::string & a) -> std::unique_ptr<GraphEmitter> {
        return std::make_unique<RecordingEmitter>(w, q, a, &seen);
    };

    try {
        build_ggml_graph_from_gguf(argv[2], factory);
    } catch (const std::exception & e) {
        std::string m = e.what();
        for (auto & c : m) {
            if (c == '\n' || c == '\t') c = ' ';
        }
        if (m.size() > 160) m = m.substr(0, 160) + "...";
        printf("FAIL\t%s\t%s\n", arch.c_str(), m.c_str());
        return 3;
    }

    std::string missing;
    for (const auto & kv : seen) {
        if (!ov2ggml::handled_ops().count(kv.first)) {
            missing += (missing.empty() ? "" : ",") + kv.first;
        }
    }
    if (missing.empty()) {
        printf("OK\t%s\t%zu ops\n", arch.c_str(), seen.size());
        return 0;
    }
    printf("GAP\t%s\t%s\n", arch.c_str(), missing.c_str());
    return 4;
}
