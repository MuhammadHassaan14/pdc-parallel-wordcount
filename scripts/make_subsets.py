#!/usr/bin/env python3
"""
make_subsets.py - cut prefixes of a corpus at line boundaries.

Usage: python3 scripts/make_subsets.py data/enwiki.txt 256M 1G 5G
Creates data/wiki_256M.txt, data/wiki_1G.txt, ... (skips sizes larger than the input)
"""
import os
import sys

UNITS = {"K": 2**10, "M": 2**20, "G": 2**30}


def parse(s):
    s = s.upper()
    return int(float(s[:-1]) * UNITS[s[-1]]) if s[-1] in UNITS else int(s)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    src = sys.argv[1]
    total = os.path.getsize(src)
    out_dir = os.path.dirname(src) or "."
    for label in sys.argv[2:]:
        limit = parse(label)
        if limit > total:
            print(f"skip {label}: input is only {total / 2**30:.2f} GB")
            continue
        dst = os.path.join(out_dir, f"wiki_{label.upper()}.txt")
        written = 0
        with open(src, "rb") as fin, open(dst, "wb") as fout:
            for line in fin:
                if written + len(line) > limit:
                    break
                fout.write(line)
                written += len(line)
        print(f"{dst}: {written / 2**20:.0f} MB")


if __name__ == "__main__":
    main()
