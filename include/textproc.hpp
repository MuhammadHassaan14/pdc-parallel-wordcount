// textproc.hpp
// Shared code used by both the sequential baseline and the MPI version.
// Keeping this in one place guarantees that both programs tokenize the text
// in exactly the same way, which is what makes the correctness check meaningful.
//
// Contents:
//   1. Tokenizer rules
//   2. Local counting structures (words, bigrams, inverted index)
//   3. ChunkedLineReader - reads a byte range in blocks and returns whole lines
//   4. Byte buffer pack / unpack helpers (used for MPI messages)
//   5. Top-K selection and output helpers

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tp {

// ---------------------------------------------------------------------------
// 1. Tokenizer
// ---------------------------------------------------------------------------
// A word is a maximal run of "word bytes". Word bytes are ASCII letters,
// ASCII digits and any byte >= 0x80 (so UTF-8 letters such as "é" stay inside
// the word instead of being cut out). Everything else (spaces, punctuation,
// tabs) is a separator. ASCII letters are lowercased.

static inline bool is_word_byte(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c >= 0x80;
}

static inline char to_lower_ascii(unsigned char c) {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c + 32) : static_cast<char>(c);
}

// 64-bit FNV-1a hash. Used to decide which process "owns" a word or bigram
// in the Alltoallv merge. It must be identical on every process, which is why
// we do not use std::hash (implementation defined).
static inline uint64_t fnv1a(const char* s, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) {
        h ^= static_cast<unsigned char>(s[i]);
        h *= 1099511628211ULL;
    }
    return h;
}
static inline uint64_t fnv1a(const std::string& s) { return fnv1a(s.data(), s.size()); }

// ---------------------------------------------------------------------------
// 2. Local counting structures
// ---------------------------------------------------------------------------
struct WordInfo {
    uint64_t count = 0;
    std::vector<uint64_t> docs;  // inverted index postings (doc ids)
};

using WordMap = std::unordered_map<std::string, WordInfo>;
using BigramMap = std::unordered_map<std::string, uint64_t>;

struct Options {
    bool bigrams = true;
    bool index = true;
    size_t topk = 50;
    size_t block_size = 64u << 20;  // 64 MB read blocks
    bool dump_counts = false;       // write full word count tables (for verification)
};

struct LocalStats {
    uint64_t docs = 0;
    uint64_t tokens = 0;
    uint64_t bytes = 0;
};

class Counter {
public:
    WordMap words;
    BigramMap bigrams;
    LocalStats stats;

    explicit Counter(const Options& o) : opt_(o) {
        words.reserve(1 << 20);
        if (opt_.bigrams) bigrams.reserve(1 << 22);
    }

    // Process one document (one line of the corpus).
    // Line format: optional "<numeric id>\t" prefix, then the article text.
    // If there is no numeric id, the absolute byte offset of the line is used
    // as the document id. Both are globally unique, and neither needs any
    // communication between processes.
    void process_line(const char* p, size_t n, uint64_t line_offset) {
        stats.bytes += n + 1;  // +1 for the newline
        if (n == 0) return;

        uint64_t doc_id = line_offset;
        size_t i = 0;
        {
            size_t j = 0;
            uint64_t id = 0;
            while (j < n && j < 20 && p[j] >= '0' && p[j] <= '9') { id = id * 10 + (p[j] - '0'); ++j; }
            if (j > 0 && j < n && p[j] == '\t') { doc_id = id; i = j + 1; }
        }
        stats.docs++;

        bool have_prev = false;
        while (i < n) {
            while (i < n && !is_word_byte(static_cast<unsigned char>(p[i]))) ++i;
            if (i >= n) break;
            tok_.clear();
            while (i < n && is_word_byte(static_cast<unsigned char>(p[i]))) {
                tok_.push_back(to_lower_ascii(static_cast<unsigned char>(p[i])));
                ++i;
            }
            stats.tokens++;

            WordInfo& wi = words[tok_];
            wi.count++;
            // Postings: a document is processed completely before the next one,
            // so checking the last element is enough to avoid duplicates.
            if (opt_.index && (wi.docs.empty() || wi.docs.back() != doc_id)) wi.docs.push_back(doc_id);

            if (opt_.bigrams) {
                if (have_prev) {
                    key_.assign(prev_);
                    key_.push_back(' ');
                    key_.append(tok_);
                    bigrams[key_]++;
                }
                prev_.swap(tok_);
                have_prev = true;
            }
        }
        // Bigrams never cross a document (line) boundary. This is also what
        // makes the parallel bigram counts identical to the sequential ones.
    }

private:
    Options opt_;
    std::string tok_, prev_, key_;
};

// ---------------------------------------------------------------------------
// 3. ChunkedLineReader
// ---------------------------------------------------------------------------
// Reads the byte range [range_start, range_end) of a file in fixed size blocks
// and hands out complete lines. Boundary rule (same idea as Hadoop splits):
//
//   A process handles every line whose FIRST byte lies inside its range.
//
//   - Start: if range_start > 0 and the byte before it is not '\n', we are in
//     the middle of a line that belongs to the previous process, so we skip
//     forward to just after the next '\n'.
//   - End: a line that starts before range_end is read completely, even if it
//     runs past range_end (we keep reading until its '\n' or EOF).
//
// Because '\n' is whitespace, this is a stricter version of the
// "move to the next whitespace" rule in the proposal: no word is split, and
// additionally no document (and therefore no bigram) is split.
//
// The actual reading is done by a callback so the same class works with
// MPI_File_read_at (MPI version) and pread (sequential version).
class ChunkedLineReader {
public:
    using ReadFn = std::function<size_t(uint64_t offset, char* buf, size_t len)>;

    ChunkedLineReader(ReadFn fn, uint64_t file_size, uint64_t range_start, uint64_t range_end,
                      size_t block_size)
        : read_(std::move(fn)), fsize_(file_size), end_(range_end), block_(block_size) {
        if (range_start == 0) {
            pos_ = 0;
        } else {
            pos_ = range_start - 1;  // look at the byte just before our range
            skip_partial_ = true;
        }
        buf_.reserve(block_ + 1024);
    }

    // Calls fn(line_ptr, line_len, line_offset) for every owned line.
    // read_seconds accumulates the time spent inside the read callback.
    template <typename LineFn, typename Clock>
    void for_each_line(LineFn&& fn, Clock&& now, double& read_seconds) {
        std::string carry;          // partial line left over from the previous block
        uint64_t carry_offset = 0;  // file offset of carry[0]
        uint64_t next_line_start = pos_;
        bool done = false;

        if (fsize_ == 0 || pos_ >= end_) return;

        while (!done) {
            if (pos_ >= fsize_) {
                // EOF: flush the last line if it has no trailing newline
                if (!carry.empty() && !skip_partial_ && carry_offset < end_)
                    fn(carry.data(), carry.size(), carry_offset);
                break;
            }
            size_t want = static_cast<size_t>(std::min<uint64_t>(block_, fsize_ - pos_));
            buf_.resize(want);
            double t0 = now();
            size_t got = read_(pos_, buf_.data(), want);
            read_seconds += now() - t0;
            if (got == 0) break;
            uint64_t block_offset = pos_;
            pos_ += got;

            size_t i = 0;
            if (skip_partial_) {
                const char* nl = static_cast<const char*>(std::memchr(buf_.data(), '\n', got));
                if (!nl) continue;  // still inside the previous process's line
                i = static_cast<size_t>(nl - buf_.data()) + 1;
                next_line_start = block_offset + i;
                skip_partial_ = false;
                if (next_line_start >= end_) { done = true; break; }
            }

            while (i < got) {
                const char* nl = static_cast<const char*>(std::memchr(buf_.data() + i, '\n', got - i));
                if (!nl) {
                    // incomplete line, keep it for the next block
                    if (carry.empty()) carry_offset = block_offset + i;
                    carry.append(buf_.data() + i, got - i);
                    break;
                }
                size_t j = static_cast<size_t>(nl - buf_.data());
                if (!carry.empty()) {
                    carry.append(buf_.data() + i, j - i);
                    fn(carry.data(), carry.size(), carry_offset);
                    carry.clear();
                } else {
                    fn(buf_.data() + i, j - i, block_offset + i);
                }
                next_line_start = block_offset + j + 1;
                i = j + 1;
                if (next_line_start >= end_) { done = true; break; }
            }
        }
    }

private:
    ReadFn read_;
    uint64_t fsize_, end_;
    size_t block_;
    uint64_t pos_ = 0;
    bool skip_partial_ = false;
    std::vector<char> buf_;
};

// ---------------------------------------------------------------------------
// 4. Pack / unpack helpers
// ---------------------------------------------------------------------------
// MPI sends bytes, so maps are flattened into a byte buffer:
//   word record   : [u32 len][bytes][u64 count][u32 ndocs][u64 doc]*ndocs
//   bigram record : [u32 len][bytes][u64 count]
struct Packer {
    std::vector<char> buf;
    void u32(uint32_t v) { raw(&v, 4); }
    void u64(uint64_t v) { raw(&v, 8); }
    void str(const std::string& s) { u32(static_cast<uint32_t>(s.size())); raw(s.data(), s.size()); }
    void raw(const void* p, size_t n) {
        const char* c = static_cast<const char*>(p);
        buf.insert(buf.end(), c, c + n);
    }
    // MPI counts are 32-bit ints. We send in units of 8 bytes (MPI_UINT64_T),
    // which raises the per-message limit from 2 GB to 16 GB.
    void pad8() { while (buf.size() % 8) buf.push_back(0); }
};

struct Unpacker {
    const char* p;
    const char* end;
    Unpacker(const char* b, size_t n) : p(b), end(b + n) {}
    bool has_more() const { return p + 4 <= end; }
    uint32_t u32() { uint32_t v; std::memcpy(&v, p, 4); p += 4; return v; }
    uint64_t u64() { uint64_t v; std::memcpy(&v, p, 8); p += 8; return v; }
    void str(std::string& s) { uint32_t n = u32(); s.assign(p, n); p += n; }
};

static inline void pack_word(Packer& pk, const std::string& w, const WordInfo& wi) {
    pk.str(w);
    pk.u64(wi.count);
    pk.u32(static_cast<uint32_t>(wi.docs.size()));
    if (!wi.docs.empty()) pk.raw(wi.docs.data(), wi.docs.size() * 8);
}

static inline void pack_bigram(Packer& pk, const std::string& b, uint64_t c) {
    pk.str(b);
    pk.u64(c);
}

// Section markers so that word records and bigram records can travel in one buffer.
// A record with length 0 is never produced by the tokenizer, so we use
// "length 0" + tag as a separator.
enum : uint32_t { SEC_WORDS = 1, SEC_BIGRAMS = 2, SEC_END = 3 };
static inline void section(Packer& pk, uint32_t tag) { pk.u32(0); pk.u32(tag); }

// Merge one packed buffer into maps. Padding zeros at the end look like a
// zero length record followed by tag 0, which we treat as "end".
static inline void unpack_merge(const char* data, size_t n, WordMap& words, BigramMap& bigrams) {
    Unpacker u(data, n);
    uint32_t sec = SEC_WORDS;
    std::string key;
    while (u.has_more()) {
        uint32_t len = u.u32();
        if (len == 0) {
            if (!u.has_more()) break;
            sec = u.u32();
            if (sec != SEC_WORDS && sec != SEC_BIGRAMS) break;
            continue;
        }
        key.assign(u.p, len);
        u.p += len;
        if (sec == SEC_WORDS) {
            uint64_t c = u.u64();
            uint32_t nd = u.u32();
            WordInfo& wi = words[key];
            wi.count += c;
            if (nd) {
                size_t old = wi.docs.size();
                wi.docs.resize(old + nd);
                std::memcpy(wi.docs.data() + old, u.p, static_cast<size_t>(nd) * 8);
                u.p += static_cast<size_t>(nd) * 8;
            }
        } else {
            bigrams[key] += u.u64();
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Top-K and output helpers
// ---------------------------------------------------------------------------
using CountPair = std::pair<std::string, uint64_t>;

// Deterministic order: higher count first, ties broken alphabetically.
// Without the tie-break, sequential and parallel output could differ in
// order even when the counts are identical.
static inline bool count_order(const CountPair& a, const CountPair& b) {
    if (a.second != b.second) return a.second > b.second;
    return a.first < b.first;
}

static inline void topk_inplace(std::vector<CountPair>& v, size_t k) {
    if (v.size() > k) {
        std::partial_sort(v.begin(), v.begin() + k, v.end(), count_order);
        v.resize(k);
    } else {
        std::sort(v.begin(), v.end(), count_order);
    }
}

// Top-K straight from a hash map: sort pointers to the entries and copy only
// the K winners (copying every key into a vector first was ~2x slower for
// multi-million entry bigram tables).
template <typename Map, typename GetCount>
static inline std::vector<CountPair> topk_from_map(const Map& m, size_t k, GetCount get) {
    using Entry = const typename Map::value_type*;
    std::vector<Entry> v;
    v.reserve(m.size());
    for (const auto& kv : m) v.push_back(&kv);
    auto cmp = [&](Entry a, Entry b) {
        uint64_t ca = get(*a), cb = get(*b);
        if (ca != cb) return ca > cb;
        return a->first < b->first;
    };
    if (v.size() > k) {
        std::nth_element(v.begin(), v.begin() + k, v.end(), cmp);
        v.resize(k);
    }
    std::sort(v.begin(), v.end(), cmp);
    std::vector<CountPair> out;
    out.reserve(v.size());
    for (Entry e : v) out.emplace_back(e->first, get(*e));
    return out;
}

static inline std::vector<CountPair> topk_words(const WordMap& m, size_t k) {
    return topk_from_map(m, k, [](const WordMap::value_type& e) { return e.second.count; });
}

static inline std::vector<CountPair> topk_bigrams(const BigramMap& m, size_t k) {
    return topk_from_map(m, k, [](const BigramMap::value_type& e) { return e.second; });
}

}  // namespace tp
