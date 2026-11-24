#!/usr/bin/env python3
"""verify.py - check that two output folders contain identical results.

Usage:
    python3 scripts/verify.py out_seq out_mpi [--max-diffs N]

Compares everything that must not depend on the number of processes or on the
merge strategy:

  summary.tsv         bytes, documents, tokens, unique_words, unique_bigrams
                      (input, processes and mode are ignored)
  top_words.tsv       exact match (both programs break ties alphabetically)
  top_bigrams.tsv     exact match
  index/part-*.tsv    all parts taken together: same words, same document
                      frequencies, same posting lists. The sequential run writes
                      one part, the alltoall run writes one part per rank.
                      A word that appears in two parts is reported.
  counts/part-*.tsv   full word and bigram tables (only with --dump-counts)

A file missing from both folders is skipped (for example after --no-bigrams);
a file missing from only one folder is a failure.

Memory: nothing is loaded as a whole. Index parts are already sorted by word,
so they are merged as streams. Count parts are unsorted and can be very large,
so they are sorted with the system `sort` (which spills to disk) and compared
as streams.

Exit code: 0 identical, 1 differences found, 2 usage error.
"""

import argparse
import glob
import heapq
import os
import shutil
import subprocess
import sys

SUMMARY_KEYS = ("bytes", "documents", "tokens", "unique_words", "unique_bigrams")
SHOW_CHARS = 120  # longest line excerpt printed in a difference report
INDENT = " " * 7


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def show(line):
    s = line.decode("utf-8", errors="replace")
    return s if len(s) <= SHOW_CHARS else s[:SHOW_CHARS] + "..."


def read_lines(path):
    """Lines of a file as bytes, without the trailing newline."""
    with open(path, "rb") as f:
        for line in f:
            yield line.rstrip(b"\n")


def read_summary(path):
    out = {}
    for line in read_lines(path):
        key, _, value = line.decode("utf-8", errors="replace").partition("\t")
        out[key] = value
    return out


def parts(folder, sub):
    return sorted(glob.glob(os.path.join(folder, sub, "part-*.tsv")))


def index_key(line):
    # word <TAB> doc_freq <TAB> doc ids
    return line.split(b"\t", 1)[0]


def counts_key(line):
    # W|B <TAB> term <TAB> count  ->  key is everything before the count
    return line.rsplit(b"\t", 1)[0]


def merged_index(paths):
    """All index parts as one stream sorted by word. Each part is written
    sorted by word (byte order), so a k-way merge is enough."""
    return heapq.merge(*(read_lines(p) for p in paths), key=index_key)


def sorted_counts(paths):
    """All count parts as one stream in byte order. Uses the system sort, which
    works on files larger than memory; falls back to sorting in memory."""
    if not paths:
        return
    sort_bin = shutil.which("sort")
    if sort_bin is None or os.name == "nt":
        lines = []
        for p in paths:
            lines.extend(read_lines(p))
        lines.sort()
        yield from lines
        return
    env = dict(os.environ, LC_ALL="C")  # plain byte order, same as the C++ programs
    proc = subprocess.Popen([sort_bin, "--"] + paths, stdout=subprocess.PIPE, env=env)
    finished = False
    try:
        for line in proc.stdout:
            yield line.rstrip(b"\n")
        finished = True
    finally:
        if not finished:
            proc.kill()
        proc.stdout.close()
        if proc.wait() != 0 and finished:
            raise RuntimeError("sort failed while reading " + os.path.dirname(paths[0]))


# ---------------------------------------------------------------------------
# Comparing
# ---------------------------------------------------------------------------
class Diffs:
    """Counts differences and keeps the first few for printing."""

    def __init__(self, limit):
        self.limit = limit
        self.count = 0
        self.examples = []

    def add(self, text):
        self.count += 1
        if len(self.examples) < self.limit:
            self.examples.append(text)


def checked(stream, key, label, diffs):
    """Pass lines through, reporting keys that repeat or go backwards."""
    prev = None
    for line in stream:
        k = key(line)
        if prev is not None and k <= prev:
            what = "duplicate" if k == prev else "out of order"
            diffs.add("%s: %s key %s" % (label, what, show(k)))
        prev = k
        yield line


def diff_sorted(a, b, key, diffs):
    """Merge-join two streams sorted by key. Returns the row counts of A and B."""
    na = nb = 0
    la, lb = next(a, None), next(b, None)
    while la is not None or lb is not None:
        if lb is None or (la is not None and key(la) < key(lb)):
            diffs.add("only in A: " + show(la))
            na += 1
            la = next(a, None)
        elif la is None or key(lb) < key(la):
            diffs.add("only in B: " + show(lb))
            nb += 1
            lb = next(b, None)
        else:
            if la != lb:
                diffs.add("different:\n%s  A: %s\n%s  B: %s" % (INDENT, show(la), INDENT, show(lb)))
            na += 1
            nb += 1
            la, lb = next(a, None), next(b, None)
    return na, nb


class Report:
    def __init__(self, max_diffs):
        self.max_diffs = max_diffs
        self.checked = 0
        self.failed = 0

    def _line(self, status, name, detail):
        print("[%s] %-16s %s" % (status, name, detail))

    def skip(self, name, why):
        self._line("SKIP", name, why)

    def fail(self, name, why):
        self.checked += 1
        self.failed += 1
        self._line("FAIL", name, why)

    def done(self, name, diffs, detail):
        self.checked += 1
        if diffs.count == 0:
            self._line("PASS", name, detail)
            return
        self.failed += 1
        self._line("FAIL", name, "%d difference(s); %s" % (diffs.count, detail))
        for e in diffs.examples:
            print(INDENT + e)
        if diffs.count > len(diffs.examples):
            print(INDENT + "... %d more" % (diffs.count - len(diffs.examples)))


def present(rep, name, in_a, in_b):
    """True if both folders have the item; reports SKIP / FAIL otherwise."""
    if not in_a and not in_b:
        rep.skip(name, "not in either folder")
        return False
    if in_a != in_b:
        rep.fail(name, "only in %s" % ("A" if in_a else "B"))
        return False
    return True


def check_summary(rep, a, b):
    pa, pb = os.path.join(a, "summary.tsv"), os.path.join(b, "summary.tsv")
    if not present(rep, "summary.tsv", os.path.isfile(pa), os.path.isfile(pb)):
        return
    sa, sb = read_summary(pa), read_summary(pb)
    diffs = Diffs(rep.max_diffs)
    for k in SUMMARY_KEYS:
        if sa.get(k) != sb.get(k):
            diffs.add("%s: A=%s B=%s" % (k, sa.get(k), sb.get(k)))
    detail = "%s docs, %s tokens, %s unique words, %s unique bigrams" % (
        sa.get("documents"), sa.get("tokens"), sa.get("unique_words"), sa.get("unique_bigrams"))
    rep.done("summary.tsv", diffs, detail)


def check_topk(rep, a, b, fname):
    pa, pb = os.path.join(a, fname), os.path.join(b, fname)
    if not present(rep, fname, os.path.isfile(pa), os.path.isfile(pb)):
        return
    la, lb = list(read_lines(pa)), list(read_lines(pb))
    diffs = Diffs(rep.max_diffs)
    for i in range(max(len(la), len(lb))):
        ra = la[i] if i < len(la) else b"(missing)"
        rb = lb[i] if i < len(lb) else b"(missing)"
        if ra != rb:
            diffs.add("line %d:  A: %s  |  B: %s" % (i + 1, show(ra), show(rb)))
    rep.done(fname, diffs, "%d rows" % max(len(la) - 1, 0))


def check_index(rep, a, b):
    pa, pb = parts(a, "index"), parts(b, "index")
    if not present(rep, "index", bool(pa), bool(pb)):
        return
    diffs = Diffs(rep.max_diffs)
    sa = checked(merged_index(pa), index_key, "A", diffs)
    sb = checked(merged_index(pb), index_key, "B", diffs)
    na, _ = diff_sorted(sa, sb, index_key, diffs)
    rep.done("index", diffs, "%d words (%d part(s) vs %d)" % (na, len(pa), len(pb)))


def check_counts(rep, a, b):
    pa, pb = parts(a, "counts"), parts(b, "counts")
    if not pa and not pb:
        rep.skip("counts", "not in either folder (run both with --dump-counts)")
        return
    if not present(rep, "counts", bool(pa), bool(pb)):
        return
    diffs = Diffs(rep.max_diffs)
    sa = checked(sorted_counts(pa), counts_key, "A", diffs)
    sb = checked(sorted_counts(pb), counts_key, "B", diffs)
    na, _ = diff_sorted(sa, sb, counts_key, diffs)
    rep.done("counts", diffs, "%d word + bigram rows (%d part(s) vs %d)" % (na, len(pa), len(pb)))


def describe(folder):
    path = os.path.join(folder, "summary.tsv")
    if not os.path.isfile(path):
        return folder
    s = read_summary(path)
    return "%s  (mode=%s, processes=%s)" % (folder, s.get("mode", "?"), s.get("processes", "?"))


def main():
    ap = argparse.ArgumentParser(
        description="Check that two output folders (e.g. sequential and MPI) contain identical results.")
    ap.add_argument("a", metavar="DIR_A", help="reference output folder, e.g. out_seq")
    ap.add_argument("b", metavar="DIR_B", help="output folder to check, e.g. out_mpi")
    ap.add_argument("--max-diffs", type=int, default=10, metavar="N",
                    help="differences to print per file (default 10)")
    args = ap.parse_args()

    for d in (args.a, args.b):
        if not os.path.isdir(d):
            print("verify.py: %s is not a directory" % d, file=sys.stderr)
            return 2

    print("A: " + describe(args.a))
    print("B: " + describe(args.b))
    rep = Report(args.max_diffs)
    check_summary(rep, args.a, args.b)
    check_topk(rep, args.a, args.b, "top_words.tsv")
    check_topk(rep, args.a, args.b, "top_bigrams.tsv")
    check_index(rep, args.a, args.b)
    check_counts(rep, args.a, args.b)

    if rep.checked == 0:
        print("verify.py: no result files found in either folder", file=sys.stderr)
        return 2
    if rep.failed:
        print("RESULT: DIFFERENT (%d of %d checks failed)" % (rep.failed, rep.checked))
        return 1
    print("RESULT: IDENTICAL (%d checks)" % rep.checked)
    return 0


if __name__ == "__main__":
    sys.exit(main())
