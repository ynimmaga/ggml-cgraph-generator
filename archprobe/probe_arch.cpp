// Probe whether OpenVINO's generic GGUF decoder builder can build a given architecture.
//
// Run against the synthetic models from gen_synthetic.py with OV_GGUF_ANY_ARCH=1, this answers
// the question "which of llama.cpp's architectures would work if we just added the name to
// arch_registry.cpp?" -- separating architectures that need real graph work from ones the
// existing tensor-table auto-detection already covers.
//
// Prints ONE line so the driver can tabulate. Building is necessary but NOT sufficient for
// support: a graph that assembles still has to be validated numerically against a reference.
#include <cstdio>
#include <exception>
#include <string>

#include "builder/gguf_builder.hpp"
#include "builder/gguf_graph.hpp"

using namespace ov::frontend::gguf;

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: probe_arch <arch-name> <model.gguf>\n");
        return 1;
    }
    const std::string arch = argv[1];
    try {
        auto g = build_ggml_graph_from_gguf(argv[2], EmitterFactory());
        if (!g) {
            printf("FAIL\t%s\tnull graph\n", arch.c_str());
            return 2;
        }
        printf("OK\t%s\tnodes=%zu inputs=%zu outputs=%zu recurrent=%zu rope=%d\n",
               arch.c_str(), g->nodes.size(), g->model_inputs.size(),
               g->model_output_names.size(), g->recurrent_states.size(), (int) g->has_rope);
        return 0;
    } catch (const std::exception & e) {
        // Collapse to one line; the driver buckets on the message.
        std::string m = e.what();
        for (auto & c : m) {
            if (c == '\n' || c == '\t') c = ' ';
        }
        if (m.size() > 240) m = m.substr(0, 240) + "...";
        printf("FAIL\t%s\t%s\n", arch.c_str(), m.c_str());
        return 3;
    }
}
