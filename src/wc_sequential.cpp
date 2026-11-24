// wc_sequential.cpp
// Sequential baseline: word frequencies, bigram counts and inverted index.
// This is T(1) for the speedup calculation and the reference output for
// verifying the MPI version.

#include <chrono>
#include <cstdio>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "args.hpp"
#include "output.hpp"
#include "textproc.hpp"

static double now_sec() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    tp::Args a;
    if (!tp::parse_args(argc, argv, a, false)) { tp::usage(argv[0], false); return 1; }

    double t_start = now_sec();

    int fd = open(a.input.c_str(), O_RDONLY);
    if (fd < 0) { std::perror(a.input.c_str()); return 1; }
    struct stat st;
    fstat(fd, &st);
    uint64_t fsize = static_cast<uint64_t>(st.st_size);

    // read callback using pread (same interface as the MPI-IO callback)
    auto reader = [fd](uint64_t off, char* buf, size_t len) -> size_t {
        size_t done = 0;
        while (done < len) {
            ssize_t r = pread(fd, buf + done, len - done, static_cast<off_t>(off + done));
            if (r <= 0) break;
            done += static_cast<size_t>(r);
        }
        return done;
    };

    // ---- read + compute ----
    tp::Counter counter(a.opt);
    tp::ChunkedLineReader lr(reader, fsize, 0, fsize, a.opt.block_size);
    double t_read = 0.0;
    double t0 = now_sec();
    lr.for_each_line([&](const char* p, size_t n, uint64_t off) { counter.process_line(p, n, off); },
                     now_sec, t_read);
    double t_compute = (now_sec() - t0) - t_read;
    close(fd);

    // ---- "merge": in the sequential version this is only the top-K selection ----
    t0 = now_sec();
    auto top_w = tp::topk_words(counter.words, a.opt.topk);
    auto top_b = tp::topk_bigrams(counter.bigrams, a.opt.topk);
    double t_merge = now_sec() - t0;

    // ---- output ----
    t0 = now_sec();
    tp::make_dir(a.out_dir);
    tp::write_topk(a.out_dir + "/top_words.tsv", top_w);
    if (a.opt.bigrams) tp::write_topk(a.out_dir + "/top_bigrams.tsv", top_b);
    if (a.opt.index) {
        tp::make_dir(a.out_dir + "/index");
        tp::write_index_part(tp::part_name(a.out_dir + "/index", 0), counter.words);
    }
    if (a.opt.dump_counts) {
        tp::make_dir(a.out_dir + "/counts");
        tp::write_counts_part(tp::part_name(a.out_dir + "/counts", 0), counter.words, counter.bigrams);
    }
    tp::write_summary(a.out_dir + "/summary.tsv", a.input, 1, "sequential", counter.stats.bytes,
                      counter.stats.docs, counter.stats.tokens, counter.words.size(),
                      counter.bigrams.size());
    double t_output = now_sec() - t0;
    double t_total = now_sec() - t_start;

    std::printf("[sequential] %s : %.1f MB, %llu docs, %llu tokens, %zu unique words, %zu unique bigrams\n",
                a.input.c_str(), fsize / 1048576.0, (unsigned long long)counter.stats.docs,
                (unsigned long long)counter.stats.tokens, counter.words.size(), counter.bigrams.size());
    std::printf("  read %.3fs | compute %.3fs | merge %.3fs | output %.3fs | TOTAL %.3fs\n", t_read,
                t_compute, t_merge, t_output, t_total);
    if (!top_w.empty()) {
        std::printf("  top words:");
        for (size_t i = 0; i < std::min<size_t>(5, top_w.size()); ++i)
            std::printf(" %s(%llu)", top_w[i].first.c_str(), (unsigned long long)top_w[i].second);
        std::printf("\n");
    }

    if (!a.csv.empty())
        tp::append_timing_csv(a.csv, a.dataset, fsize, 1, "sequential", t_read, t_compute, 0.0, t_merge,
                              t_output, t_total);
    return 0;
}
