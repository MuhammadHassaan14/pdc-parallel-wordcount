// args.hpp - command line options shared by both programs
#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>

#include "textproc.hpp"

namespace tp {

struct Args {
    std::string input;
    std::string out_dir = "output";
    std::string csv;           // append timings here if set
    std::string dataset;       // label used in the CSV (defaults to file name)
    std::string merge = "alltoall";  // MPI only: "gather" or "alltoall"
    Options opt;
};

static inline void usage(const char* prog, bool mpi) {
    std::fprintf(stderr,
        "Usage: %s <corpus.txt> [options]\n"
        "  --out DIR          output directory (default: output)\n"
        "  --topk K           number of top words / bigrams to report (default 50)\n"
        "  --no-bigrams       skip bigram counting\n"
        "  --no-index         skip inverted index\n"
        "  --dump-counts      also write full word/bigram count tables (for verification)\n"
        "  --block-mb N       read block size in MB (default 64)\n"
        "  --block-kb N       read block size in KB (for testing boundary handling)\n"
        "  --csv FILE         append timing row to FILE\n"
        "  --dataset NAME     dataset label for the CSV\n", prog);
    if (mpi)
        std::fprintf(stderr,
        "  --merge MODE       gather   : every rank sends everything to rank 0 (MPI_Gatherv)\n"
        "                     alltoall : hash-partition words to owner ranks (MPI_Alltoallv)\n");
}

static inline bool parse_args(int argc, char** argv, Args& a, bool mpi) {
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", what); std::exit(1); }
            return argv[++i];
        };
        if (s == "--out") a.out_dir = next("--out");
        else if (s == "--topk") a.opt.topk = std::strtoull(next("--topk").c_str(), nullptr, 10);
        else if (s == "--no-bigrams") a.opt.bigrams = false;
        else if (s == "--no-index") a.opt.index = false;
        else if (s == "--dump-counts") a.opt.dump_counts = true;
        else if (s == "--block-mb") a.opt.block_size = std::strtoull(next("--block-mb").c_str(), nullptr, 10) << 20;
        else if (s == "--block-kb") a.opt.block_size = std::strtoull(next("--block-kb").c_str(), nullptr, 10) << 10;
        else if (s == "--csv") a.csv = next("--csv");
        else if (s == "--dataset") a.dataset = next("--dataset");
        else if (mpi && s == "--merge") a.merge = next("--merge");
        else if (s == "-h" || s == "--help") return false;
        else if (!s.empty() && s[0] == '-') { std::fprintf(stderr, "unknown option %s\n", s.c_str()); return false; }
        else a.input = s;
    }
    if (a.input.empty()) return false;
    if (mpi && a.merge != "gather" && a.merge != "alltoall") {
        std::fprintf(stderr, "--merge must be gather or alltoall\n");
        return false;
    }
    if (a.dataset.empty()) {
        size_t p = a.input.find_last_of('/');
        a.dataset = (p == std::string::npos) ? a.input : a.input.substr(p + 1);
    }
    if (a.opt.block_size < 4096) a.opt.block_size = 4096;
    return true;
}

}  // namespace tp
