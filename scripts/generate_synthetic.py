#!/usr/bin/env python3
"""
generate_synthetic.py - create a Zipf-distributed English-like corpus.

Useful for testing the programs before the Wikipedia dump is ready, and for
quick benchmarks. Word frequencies in natural language follow Zipf's law, so
this gives a similar hash-map workload to real text.

Output format is the same as wiki_to_lines.py:  <doc id> TAB <text> NEWLINE

Usage: python3 scripts/generate_synthetic.py --size-mb 500 --out data/synthetic_500MB.txt
"""
import argparse
import numpy as np

SYLLABLES = ["ka", "lo", "mi", "ne", "ra", "su", "ti", "vo", "an", "el", "is", "or",
             "un", "pre", "con", "ing", "ed", "ly", "tion", "er", "st", "ch", "th", "qu"]
COMMON = ("the of and in to a is was for on as by with he that at from his it an "
          "were are which this also be or has had first one their its new after "
          "who they two her she been other when there all during into school time").split()


def make_vocab(n, rng):
    vocab = list(COMMON)
    seen = set(vocab)
    while len(vocab) < n:
        k = rng.integers(1, 5)
        w = "".join(SYLLABLES[i] for i in rng.integers(0, len(SYLLABLES), k))
        if w not in seen:
            seen.add(w)
            vocab.append(w)
    return np.array(vocab, dtype=object)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size-mb", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    vocab = make_vocab(args.vocab, rng)
    # Zipf with exponent ~1.07 (typical for English text).
    # Sampling = binary search of uniform randoms in the cumulative distribution.
    ranks = np.arange(1, len(vocab) + 1)
    cdf = np.cumsum(1.0 / ranks ** 1.07)
    cdf /= cdf[-1]
    # Pre-built token strings: word + punctuation + space. Row 3 is capitalized
    # (start of a sentence) so the lowercasing code is exercised too.
    table = np.empty((4, len(vocab)), dtype=object)
    table[0] = [w + " " for w in vocab]
    table[1] = [w + ", " for w in vocab]
    table[2] = [w + ". " for w in vocab]
    table[3] = [w.capitalize() + " " for w in vocab]
    punct_p = [0.82, 0.08, 0.06, 0.04]

    target = int(args.size_mb * 1024 * 1024)
    written, doc_id = 0, 1
    with open(args.out, "w", encoding="utf-8") as f:
        while written < target:
            ndocs = 2000
            lengths = (rng.lognormal(6.0, 0.9, ndocs)).astype(np.int64) + 5  # words per article
            total = int(lengths.sum())
            idx = np.searchsorted(cdf, rng.random(total))
            kind = rng.choice(4, size=total, p=punct_p)
            tokens = table[kind, idx]
            out, pos = [], 0
            for n in lengths:
                out.append(f"{doc_id}\t")
                out.append("".join(tokens[pos:pos + n]).rstrip())
                out.append("\n")
                pos += n
                doc_id += 1
            chunk = "".join(out)
            f.write(chunk)
            written += len(chunk.encode("utf-8"))
    print(f"wrote {args.out}: {written / 1048576:.1f} MB, {doc_id - 1} documents")


if __name__ == "__main__":
    main()
