// output.hpp - writing result files (shared by sequential and MPI versions)
#pragma once

#include <cstdio>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "textproc.hpp"

namespace tp {

static inline void make_dir(const std::string& d) { mkdir(d.c_str(), 0755); }

static inline std::string part_name(const std::string& dir, int part) {
    char name[64];
    std::snprintf(name, sizeof(name), "/part-%05d.tsv", part);
    return dir + name;
}

static inline void write_topk(const std::string& path, const std::vector<CountPair>& v) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) { std::perror(path.c_str()); return; }
    std::fprintf(f, "rank\tterm\tcount\n");
    for (size_t i = 0; i < v.size(); ++i)
        std::fprintf(f, "%zu\t%s\t%llu\n", i + 1, v[i].first.c_str(),
                     static_cast<unsigned long long>(v[i].second));
    std::fclose(f);
}

// Inverted index part file. One line per word:
//   word <TAB> document_frequency <TAB> doc1,doc2,doc3,...   (doc ids sorted)
// Words inside a part are sorted so the output is deterministic.
static inline void write_index_part(const std::string& path, WordMap& words) {
    std::vector<const std::pair<const std::string, WordInfo>*> rows;
    rows.reserve(words.size());
    for (auto& kv : words) {
        std::sort(kv.second.docs.begin(), kv.second.docs.end());
        rows.push_back(&kv);
    }
    std::sort(rows.begin(), rows.end(), [](auto* a, auto* b) { return a->first < b->first; });
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) { std::perror(path.c_str()); return; }
    std::vector<char> big(1 << 20);
    std::setvbuf(f, big.data(), _IOFBF, big.size());
    for (auto* r : rows) {
        std::fprintf(f, "%s\t%zu\t", r->first.c_str(), r->second.docs.size());
        const auto& d = r->second.docs;
        for (size_t i = 0; i < d.size(); ++i)
            std::fprintf(f, i ? ",%llu" : "%llu", static_cast<unsigned long long>(d[i]));
        std::fputc('\n', f);
    }
    std::fclose(f);
}

// Full word-count table (only with --dump-counts, used by verify.py).
static inline void write_counts_part(const std::string& path, const WordMap& words,
                                     const BigramMap& bigrams) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) { std::perror(path.c_str()); return; }
    std::vector<char> big(1 << 20);
    std::setvbuf(f, big.data(), _IOFBF, big.size());
    for (const auto& kv : words)
        std::fprintf(f, "W\t%s\t%llu\n", kv.first.c_str(), static_cast<unsigned long long>(kv.second.count));
    for (const auto& kv : bigrams)
        std::fprintf(f, "B\t%s\t%llu\n", kv.first.c_str(), static_cast<unsigned long long>(kv.second));
    std::fclose(f);
}

static inline void write_summary(const std::string& path, const std::string& input, int procs,
                                 const std::string& mode, uint64_t bytes, uint64_t docs,
                                 uint64_t tokens, uint64_t uniq_words, uint64_t uniq_bigrams) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) { std::perror(path.c_str()); return; }
    std::fprintf(f, "input\t%s\nprocesses\t%d\nmode\t%s\n", input.c_str(), procs, mode.c_str());
    std::fprintf(f, "bytes\t%llu\ndocuments\t%llu\ntokens\t%llu\nunique_words\t%llu\nunique_bigrams\t%llu\n",
                 (unsigned long long)bytes, (unsigned long long)docs, (unsigned long long)tokens,
                 (unsigned long long)uniq_words, (unsigned long long)uniq_bigrams);
    std::fclose(f);
}

// Appends one row to a CSV used by plot_results.py.
static inline void append_timing_csv(const std::string& path, const std::string& dataset,
                                     uint64_t bytes, int procs, const std::string& mode,
                                     double t_read, double t_compute, double t_comm,
                                     double t_merge, double t_output, double t_total) {
    bool fresh = false;
    if (FILE* t = std::fopen(path.c_str(), "r")) std::fclose(t); else fresh = true;
    FILE* f = std::fopen(path.c_str(), "a");
    if (!f) { std::perror(path.c_str()); return; }
    if (fresh) std::fprintf(f, "dataset,bytes,procs,mode,read,compute,comm,merge,output,total\n");
    std::fprintf(f, "%s,%llu,%d,%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", dataset.c_str(),
                 (unsigned long long)bytes, procs, mode.c_str(), t_read, t_compute, t_comm,
                 t_merge, t_output, t_total);
    std::fclose(f);
}

}  // namespace tp
