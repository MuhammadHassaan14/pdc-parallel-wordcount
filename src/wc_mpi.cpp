// wc_mpi.cpp
// MPI version of the word count / bigram / inverted index program.
//
// Pipeline (every rank):
//   1. Partition   : rank 0 gets the file size, broadcasts it; each rank takes
//                    the byte range [r*N/P, (r+1)*N/P).
//   2. Read        : MPI_File_read_at in 64 MB blocks, only inside own range
//                    (plus the tail of the last line that crosses the boundary).
//   3. Compute     : tokenize, count words + bigrams, build local index.
//   4. Merge       : all ranks MPI_Gatherv their tables to rank 0, rank 0 merges.
//   5. Output      : top-K words/bigrams and the inverted index (rank 0).
//   6. Stats       : totals with MPI_Reduce(SUM), phase timings with MPI_Reduce(MAX).

#include <mpi.h>

#include <climits>
#include <cstdio>
#include <string>
#include <vector>

#include "args.hpp"
#include "output.hpp"
#include "textproc.hpp"

using namespace tp;

static double wtime() { return MPI_Wtime(); }

// Convert a byte count to a count of 8-byte units for MPI (see Packer::pad8).
static int units8(size_t bytes, int rank, const char* what) {
    size_t u = bytes / 8;
    if (u > static_cast<size_t>(INT_MAX)) {
        std::fprintf(stderr, "rank %d: %s message too large (%zu bytes). Use more processes.\n",
                     rank, what, bytes);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    return static_cast<int>(u);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    Args a;
    if (!parse_args(argc, argv, a, true)) {
        if (rank == 0) usage(argv[0], true);
        MPI_Finalize();
        return 1;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = wtime();
    double t_read = 0, t_compute = 0, t_comm = 0, t_merge = 0, t_output = 0;

    // ------------------------------------------------------------------
    // 1. Partition
    // ------------------------------------------------------------------
    MPI_File fh;
    if (MPI_File_open(MPI_COMM_WORLD, a.input.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh) != MPI_SUCCESS) {
        if (rank == 0) std::fprintf(stderr, "cannot open %s\n", a.input.c_str());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    unsigned long long fsize = 0;
    if (rank == 0) {
        MPI_Offset sz;
        MPI_File_get_size(fh, &sz);
        fsize = static_cast<unsigned long long>(sz);
    }
    MPI_Bcast(&fsize, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    uint64_t r_start = fsize * static_cast<uint64_t>(rank) / P;
    uint64_t r_end = fsize * static_cast<uint64_t>(rank + 1) / P;

    // ------------------------------------------------------------------
    // 2 + 3. Read own byte range with MPI-IO and process it
    // ------------------------------------------------------------------
    auto reader = [&fh](uint64_t off, char* buf, size_t len) -> size_t {
        MPI_Status st;
        MPI_File_read_at(fh, static_cast<MPI_Offset>(off), buf, static_cast<int>(len), MPI_BYTE, &st);
        int got = 0;
        MPI_Get_count(&st, MPI_BYTE, &got);
        return static_cast<size_t>(got);
    };

    Counter counter(a.opt);
    {
        ChunkedLineReader lr(reader, fsize, r_start, r_end, a.opt.block_size);
        double t0 = wtime();
        lr.for_each_line([&](const char* p, size_t n, uint64_t off) { counter.process_line(p, n, off); },
                         wtime, t_read);
        t_compute = (wtime() - t0) - t_read;
    }
    MPI_File_close(&fh);

    WordMap& words = counter.words;
    BigramMap& bigrams = counter.bigrams;
    std::vector<CountPair> top_w, top_b;
    uint64_t my_unique_words = 0, my_unique_bigrams = 0;

    if (rank == 0) {
        make_dir(a.out_dir);
        if (a.opt.index) make_dir(a.out_dir + "/index");
        if (a.opt.dump_counts) make_dir(a.out_dir + "/counts");
    }
    // Make sure the directories exist before any rank writes a part file.
    // Waiting here is caused by load imbalance in the compute phase, so it is
    // counted as communication/synchronization time.
    {
        double tb = wtime();
        MPI_Barrier(MPI_COMM_WORLD);
        t_comm += wtime() - tb;
    }

    if (P == 1) {
        // ------------------------------------------------------------------
        // Single process: nothing to exchange.
        // ------------------------------------------------------------------
        double t0 = wtime();
        top_w = topk_words(words, a.opt.topk);
        top_b = topk_bigrams(bigrams, a.opt.topk);
        t_merge = wtime() - t0;
        t0 = wtime();
        if (a.opt.index) write_index_part(part_name(a.out_dir + "/index", 0), words);
        if (a.opt.dump_counts) write_counts_part(part_name(a.out_dir + "/counts", 0), words, bigrams);
        t_output = wtime() - t0;
        my_unique_words = words.size();
        my_unique_bigrams = bigrams.size();
    } else {
        // ------------------------------------------------------------------
        // 4. Centralized merge: MPI_Gather (sizes) + MPI_Gatherv (data)
        // ------------------------------------------------------------------
        double t0 = wtime();
        Packer pk;
        if (rank != 0) {  // rank 0 keeps its own maps and merges others into them
            section(pk, SEC_WORDS);
            for (const auto& kv : words) pack_word(pk, kv.first, kv.second);
            section(pk, SEC_BIGRAMS);
            for (const auto& kv : bigrams) pack_bigram(pk, kv.first, kv.second);
            section(pk, SEC_END);
            pk.pad8();
            WordMap().swap(words);
            BigramMap().swap(bigrams);
        }
        t_merge += wtime() - t0;

        t0 = wtime();
        int my_units = units8(pk.buf.size(), rank, "gather");
        std::vector<int> counts(rank == 0 ? P : 0), displs(rank == 0 ? P : 0);
        MPI_Gather(&my_units, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        std::vector<char> recv;
        if (rank == 0) {
            size_t total = 0;
            for (int i = 0; i < P; ++i) {
                if (total > static_cast<size_t>(INT_MAX)) {
                    std::fprintf(stderr, "gather: total data too large for rank 0\n");
                    MPI_Abort(MPI_COMM_WORLD, 2);
                }
                displs[i] = static_cast<int>(total);
                total += counts[i];
            }
            recv.resize(total * 8);
        }
        MPI_Gatherv(pk.buf.data(), my_units, MPI_UINT64_T, recv.data(), counts.data(), displs.data(),
                    MPI_UINT64_T, 0, MPI_COMM_WORLD);
        t_comm += wtime() - t0;
        std::vector<char>().swap(pk.buf);

        if (rank == 0) {
            t0 = wtime();
            for (int i = 1; i < P; ++i)
                unpack_merge(recv.data() + static_cast<size_t>(displs[i]) * 8,
                             static_cast<size_t>(counts[i]) * 8, words, bigrams);
            std::vector<char>().swap(recv);
            top_w = topk_words(words, a.opt.topk);
            top_b = topk_bigrams(bigrams, a.opt.topk);
            t_merge += wtime() - t0;

            t0 = wtime();
            if (a.opt.index) write_index_part(part_name(a.out_dir + "/index", 0), words);
            if (a.opt.dump_counts) write_counts_part(part_name(a.out_dir + "/counts", 0), words, bigrams);
            t_output = wtime() - t0;
            my_unique_words = words.size();
            my_unique_bigrams = bigrams.size();
        }
    }

    // ------------------------------------------------------------------
    // 5. Rank 0 writes top-K + summary
    // ------------------------------------------------------------------
    unsigned long long local_stats[5] = {counter.stats.bytes, counter.stats.docs, counter.stats.tokens,
                                         my_unique_words, my_unique_bigrams};
    unsigned long long global_stats[5] = {0, 0, 0, 0, 0};
    MPI_Reduce(local_stats, global_stats, 5, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    std::string mode = (P == 1) ? "mpi" : "gather";
    if (rank == 0) {
        double t0 = wtime();
        write_topk(a.out_dir + "/top_words.tsv", top_w);
        if (a.opt.bigrams) write_topk(a.out_dir + "/top_bigrams.tsv", top_b);
        write_summary(a.out_dir + "/summary.tsv", a.input, P, mode, global_stats[0], global_stats[1],
                      global_stats[2], global_stats[3], global_stats[4]);
        t_output += wtime() - t0;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_total = wtime() - t_start;

    // ------------------------------------------------------------------
    // 6. Timings: the slowest rank decides each phase -> MPI_MAX
    // ------------------------------------------------------------------
    double local_t[6] = {t_read, t_compute, t_comm, t_merge, t_output, t_total};
    double max_t[6];
    MPI_Reduce(local_t, max_t, 6, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Per-rank load (bytes / tokens) to show load balance in the report
    unsigned long long my_tokens = counter.stats.tokens;
    std::vector<unsigned long long> all_tokens(rank == 0 ? P : 0);
    MPI_Gather(&my_tokens, 1, MPI_UNSIGNED_LONG_LONG, all_tokens.data(), 1, MPI_UNSIGNED_LONG_LONG, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf("[mpi p=%d merge=%s] %s : %.1f MB, %llu docs, %llu tokens, %llu unique words, %llu unique bigrams\n",
                    P, mode.c_str(), a.input.c_str(), fsize / 1048576.0, global_stats[1], global_stats[2],
                    global_stats[3], global_stats[4]);
        std::printf("  read %.3fs | compute %.3fs | comm %.3fs | merge %.3fs | output %.3fs | TOTAL %.3fs\n",
                    max_t[0], max_t[1], max_t[2], max_t[3], max_t[4], max_t[5]);
        unsigned long long mn = ~0ULL, mx = 0;
        for (auto t : all_tokens) { mn = std::min(mn, t); mx = std::max(mx, t); }
        std::printf("  load balance: tokens per rank min %llu, max %llu\n", mn, mx);
        if (!top_w.empty()) {
            std::printf("  top words:");
            for (size_t i = 0; i < std::min<size_t>(5, top_w.size()); ++i)
                std::printf(" %s(%llu)", top_w[i].first.c_str(), (unsigned long long)top_w[i].second);
            std::printf("\n");
        }
        if (!a.csv.empty())
            append_timing_csv(a.csv, a.dataset, fsize, P, mode, max_t[0], max_t[1], max_t[2], max_t[3],
                              max_t[4], max_t[5]);
    }

    MPI_Finalize();
    return 0;
}
