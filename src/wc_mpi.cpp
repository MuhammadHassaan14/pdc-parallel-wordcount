// wc_mpi.cpp
// MPI version of the word count / bigram / inverted index program.
//
// Pipeline (every rank):
//   1. Partition   : rank 0 gets the file size, broadcasts it; each rank takes
//                    the byte range [r*N/P, (r+1)*N/P).
//   2. Read        : MPI_File_read_at in 64 MB blocks, only inside own range
//                    (plus the tail of the last line that crosses the boundary).
//   3. Compute     : tokenize, count words + bigrams, build local index.
//   4. Merge       : two strategies (select with --merge):
//        gather    : all ranks MPI_Gatherv their tables to rank 0, rank 0 merges.
//        alltoall  : each word is owned by rank hash(word) % P. Ranks exchange
//                    data with MPI_Alltoallv so every owner merges only its share
//                    in parallel, then owners send their local top-K to rank 0.
//   5. Output      : top-K words/bigrams (rank 0) and the inverted index
//                    (one part file per owner in alltoall mode).
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
        std::fprintf(stderr, "rank %d: %s message too large (%zu bytes). Use more processes or --merge alltoall.\n",
                     rank, what, bytes);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    return static_cast<int>(u);
}

// Pack a list of (term,count) pairs (used to send local top-K lists to rank 0)
static void pack_pairs(Packer& pk, const std::vector<CountPair>& v) {
    pk.u64(v.size());
    for (const auto& p : v) { pk.str(p.first); pk.u64(p.second); }
}
static void unpack_pairs(Unpacker& u, std::vector<CountPair>& out) {
    uint64_t n = u.u64();
    std::string s;
    for (uint64_t i = 0; i < n; ++i) { u.str(s); uint64_t c = u.u64(); out.emplace_back(s, c); }
}

// Count word and bigram records in the received buffers (used to pre-size maps).
static void count_records(const char* buf, size_t, const std::vector<int>& counts,
                          const std::vector<int>& displs, size_t& wrec, size_t& brec) {
    for (size_t s = 0; s < counts.size(); ++s) {
        Unpacker u(buf + static_cast<size_t>(displs[s]) * 8, static_cast<size_t>(counts[s]) * 8);
        uint32_t sec = SEC_WORDS;
        while (u.has_more()) {
            uint32_t len = u.u32();
            if (len == 0) {
                if (!u.has_more()) break;
                sec = u.u32();
                if (sec != SEC_WORDS && sec != SEC_BIGRAMS) break;
                continue;
            }
            u.p += len;
            if (sec == SEC_WORDS) { u.p += 8; uint32_t nd = u.u32(); u.p += static_cast<size_t>(nd) * 8; ++wrec; }
            else { u.p += 8; ++brec; }
        }
    }
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
    } else if (a.merge == "gather") {
        // ------------------------------------------------------------------
        // 4a. Centralized merge: MPI_Gather (sizes) + MPI_Gatherv (data)
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
                    std::fprintf(stderr, "gather: total data too large for rank 0, use --merge alltoall\n");
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
    } else {
        // ------------------------------------------------------------------
        // 4b. Distributed merge: hash partition + MPI_Alltoallv
        // ------------------------------------------------------------------
        double t0 = wtime();
        // Entries this rank owns itself are not packed and sent to itself:
        // they are moved into the "mine" tables with C++17 node extraction
        // (no copy, no new allocation). Only the other P-1 shares are packed.
        std::vector<Packer> out(P);
        WordMap my_words;
        BigramMap my_bigrams;
        for (int d = 0; d < P; ++d) section(out[d], SEC_WORDS);
        for (auto it = words.begin(); it != words.end();) {
            int owner = static_cast<int>(fnv1a(it->first) % P);
            if (owner == rank) { my_words.insert(words.extract(it++)); continue; }
            pack_word(out[owner], it->first, it->second);
            ++it;
        }
        for (int d = 0; d < P; ++d) section(out[d], SEC_BIGRAMS);
        for (auto it = bigrams.begin(); it != bigrams.end();) {
            int owner = static_cast<int>(fnv1a(it->first) % P);
            if (owner == rank) { my_bigrams.insert(bigrams.extract(it++)); continue; }
            pack_bigram(out[owner], it->first, it->second);
            ++it;
        }
        WordMap().swap(words);
        BigramMap().swap(bigrams);
        words.swap(my_words);
        bigrams.swap(my_bigrams);

        std::vector<int> scounts(P), sdispls(P), rcounts(P), rdispls(P);
        size_t stotal = 0;
        for (int d = 0; d < P; ++d) {
            section(out[d], SEC_END);
            out[d].pad8();
            scounts[d] = units8(out[d].buf.size(), rank, "alltoall");
            stotal += scounts[d];
        }
        if (stotal > static_cast<size_t>(INT_MAX)) {
            std::fprintf(stderr, "rank %d: send buffer too large, use more processes\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        std::vector<char> sendbuf(stotal * 8);
        size_t off = 0;
        for (int d = 0; d < P; ++d) {
            sdispls[d] = static_cast<int>(off);
            std::memcpy(sendbuf.data() + off * 8, out[d].buf.data(), out[d].buf.size());
            off += scounts[d];
            std::vector<char>().swap(out[d].buf);
        }
        t_merge += wtime() - t0;

        t0 = wtime();
        MPI_Alltoall(scounts.data(), 1, MPI_INT, rcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        size_t rtotal = 0;
        for (int s = 0; s < P; ++s) {
            if (rtotal > static_cast<size_t>(INT_MAX)) {
                std::fprintf(stderr, "rank %d: receive buffer too large, use more processes\n", rank);
                MPI_Abort(MPI_COMM_WORLD, 2);
            }
            rdispls[s] = static_cast<int>(rtotal);
            rtotal += rcounts[s];
        }
        std::vector<char> recvbuf(rtotal * 8);
        MPI_Alltoallv(sendbuf.data(), scounts.data(), sdispls.data(), MPI_UINT64_T, recvbuf.data(),
                      rcounts.data(), rdispls.data(), MPI_UINT64_T, MPI_COMM_WORLD);
        t_comm += wtime() - t0;
        std::vector<char>().swap(sendbuf);

        // Merge the share of words this rank owns.
        // Pre-size the tables (upper bound: everything received is new) so the
        // hash maps do not rehash repeatedly while merging.
        t0 = wtime();
        {
            size_t wrec = 0, brec = 0;
            count_records(recvbuf.data(), recvbuf.size(), rcounts, rdispls, wrec, brec);
            words.reserve(words.size() + wrec);
            if (a.opt.bigrams) bigrams.reserve(bigrams.size() + brec);
        }
        for (int s = 0; s < P; ++s)
            unpack_merge(recvbuf.data() + static_cast<size_t>(rdispls[s]) * 8,
                         static_cast<size_t>(rcounts[s]) * 8, words, bigrams);
        std::vector<char>().swap(recvbuf);
        // Each word lives on exactly one owner, so the global top-K is contained
        // in the union of the owners' local top-K lists.
        auto local_w = topk_words(words, a.opt.topk);
        auto local_b = topk_bigrams(bigrams, a.opt.topk);
        Packer tk;
        pack_pairs(tk, local_w);
        pack_pairs(tk, local_b);
        tk.pad8();
        t_merge += wtime() - t0;

        // Owners write their own index part in parallel
        t0 = wtime();
        if (a.opt.index) write_index_part(part_name(a.out_dir + "/index", rank), words);
        if (a.opt.dump_counts) write_counts_part(part_name(a.out_dir + "/counts", rank), words, bigrams);
        t_output += wtime() - t0;
        my_unique_words = words.size();
        my_unique_bigrams = bigrams.size();

        // Collect local top-K lists at rank 0 (small message)
        t0 = wtime();
        int my_units = units8(tk.buf.size(), rank, "topk");
        std::vector<int> counts(rank == 0 ? P : 0), displs(rank == 0 ? P : 0);
        MPI_Gather(&my_units, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        std::vector<char> recv;
        if (rank == 0) {
            int total = 0;
            for (int i = 0; i < P; ++i) { displs[i] = total; total += counts[i]; }
            recv.resize(static_cast<size_t>(total) * 8);
        }
        MPI_Gatherv(tk.buf.data(), my_units, MPI_UINT64_T, recv.data(), counts.data(), displs.data(),
                    MPI_UINT64_T, 0, MPI_COMM_WORLD);
        t_comm += wtime() - t0;

        if (rank == 0) {
            t0 = wtime();
            for (int i = 0; i < P; ++i) {
                Unpacker u(recv.data() + static_cast<size_t>(displs[i]) * 8, static_cast<size_t>(counts[i]) * 8);
                unpack_pairs(u, top_w);
                unpack_pairs(u, top_b);
            }
            topk_inplace(top_w, a.opt.topk);
            topk_inplace(top_b, a.opt.topk);
            t_merge += wtime() - t0;
        }
    }

    // ------------------------------------------------------------------
    // 5. Rank 0 writes top-K + summary
    // ------------------------------------------------------------------
    unsigned long long local_stats[5] = {counter.stats.bytes, counter.stats.docs, counter.stats.tokens,
                                         my_unique_words, my_unique_bigrams};
    unsigned long long global_stats[5] = {0, 0, 0, 0, 0};
    MPI_Reduce(local_stats, global_stats, 5, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    std::string mode = (P == 1) ? "mpi" : a.merge;
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
