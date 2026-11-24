#!/usr/bin/env python3
"""
wiki_to_lines.py - convert WikiExtractor --json output into the corpus format:

    <article id> TAB <title. article text with newlines replaced by spaces> NEWLINE

One article per line means a document never spans two processes (see the
boundary handling in include/textproc.hpp), so the inverted index and the
bigram counts stay exact.

Usage: python3 scripts/wiki_to_lines.py <extracted_dir> <output.txt>
"""
import json
import os
import sys


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    files = []
    for root, _, names in os.walk(src):
        files += [os.path.join(root, n) for n in names if n.startswith("wiki_")]
    files.sort()
    docs, size = 0, 0
    with open(dst, "w", encoding="utf-8") as out:
        for fn in files:
            with open(fn, encoding="utf-8") as f:
                for line in f:
                    art = json.loads(line)
                    text = art.get("text", "").replace("\n", " ").replace("\t", " ").strip()
                    if not text:
                        continue  # redirects / empty pages
                    row = f"{art['id']}\t{art['title']}. {text}\n"
                    out.write(row)
                    docs += 1
                    size += len(row.encode("utf-8"))
            print(f"  {fn}: {docs} articles, {size / 2**30:.2f} GB so far", flush=True)
    print(f"done: {dst}  {docs} articles, {size / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
