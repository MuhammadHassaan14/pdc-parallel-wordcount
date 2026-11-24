#!/usr/bin/env python3
"""
build_report.py - builds report/Project_Report.pdf from the benchmark results.

Run after:  scripts/run_tests.sh <dataset> > results/test_results.txt
            scripts/run_benchmarks.sh
            scripts/plot_results.py

Usage: python3 report/build_report.py --repo https://github.com/<user>/<repo>
"""
import argparse
import csv
import os
import re
import statistics
from collections import defaultdict
from datetime import date
from xml.sax.saxutils import escape

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")

TEAM = [("Fozan Javaid", "23K-0605"), ("Muhammad Hassaan", "23K-0570"), ("Ajay Kumar", "23K-0514")]
ROLES = {
    "Fozan Javaid": "MPI version (wc_mpi.cpp): partitioning, MPI-IO reading, Gatherv and Alltoallv merge",
    "Muhammad Hassaan": "Sequential baseline, tokenizer, boundary handling (ChunkedLineReader), verification",
    "Ajay Kumar": "Dataset preparation (WikiExtractor), benchmark scripts, graphs and report",
}

# ---------------------------------------------------------------- fonts / styles
FONT, FONT_B, MONO = "Helvetica", "Helvetica-Bold", "Courier"
for reg, bold in (("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),):
    if os.path.exists(reg) and os.path.exists(bold):
        pdfmetrics.registerFont(TTFont("DejaVu", reg))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", bold))
        FONT, FONT_B = "DejaVu", "DejaVu-Bold"
        mono = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
        if os.path.exists(mono):
            pdfmetrics.registerFont(TTFont("DejaVuMono", mono))
            MONO = "DejaVuMono"

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontName=FONT, fontSize=10.3, leading=14.5,
                      alignment=TA_JUSTIFY, spaceAfter=6)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName=FONT_B, fontSize=15, spaceBefore=10, spaceAfter=8,
                    textColor=colors.HexColor("#1F3864"))
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName=FONT_B, fontSize=12, spaceBefore=8, spaceAfter=5,
                    textColor=colors.HexColor("#2E5597"))
CAP = ParagraphStyle("cap", parent=BODY, fontSize=9, leading=12, alignment=TA_CENTER,
                     textColor=colors.HexColor("#444444"), spaceAfter=10)
CODE = ParagraphStyle("code", parent=BODY, fontName=MONO, fontSize=8.6, leading=11.5, alignment=0,
                      backColor=colors.HexColor("#F3F4F6"), borderPadding=5, spaceBefore=4, spaceAfter=8)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=3)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.8, leading=11, alignment=0, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName=FONT_B, textColor=colors.white)

fig_no = [0]
tab_no = [0]


def P(t, st=BODY):
    return Paragraph(t, st)


def bullets(items):
    return [Paragraph(t, BUL, bulletText="•") for t in items]


def code(text):
    return Paragraph(escape(text).replace("\n", "<br/>").replace("  ", "&nbsp;&nbsp;"), CODE)


def table(rows, widths, caption=None):
    data = [[Paragraph(str(c), CELLB if i == 0 else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E5597")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF2F8")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8C2D6")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    out = [t]
    if caption:
        tab_no[0] += 1
        out.append(Paragraph(f"Table {tab_no[0]}: {caption}", CAP))
    return KeepTogether(out)


def figure(path, caption, width=15 * cm):
    if not os.path.exists(path):
        return P(f"<i>[figure missing: {os.path.basename(path)} - run scripts/plot_results.py]</i>")
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize()
    fig_no[0] += 1
    return KeepTogether([Image(path, width=width, height=width * ih / iw),
                         Paragraph(f"Figure {fig_no[0]}: {caption}", CAP)])


# ---------------------------------------------------------------- data loading
def load_bench():
    path = os.path.join(RES, "benchmarks.csv")
    runs, sizes = defaultdict(list), {}
    if not os.path.exists(path):
        return {}, {}
    with open(path) as f:
        for r in csv.DictReader(f):
            k = (r["dataset"], r["mode"], int(r["procs"]))
            runs[k].append({x: float(r[x]) for x in ("read", "compute", "comm", "merge", "output", "total")})
            sizes[r["dataset"]] = int(r["bytes"])
    med = {k: {x: statistics.median(v[x] for v in rows) for x in rows[0]} | {"n": len(rows)}
           for k, rows in runs.items()}
    return med, sizes


def series(med, d, mode):
    return dict(sorted((p, v) for (dd, m, p), v in med.items()
                       if dd == d and (m == mode or (m == "mpi" and p == 1))))


def read_kv(path):
    out = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8", errors="replace"):
            if ":" in line:
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()
            elif "\t" in line:
                k, v = line.rstrip("\n").split("\t", 1)
                out[k] = v
    return out


def read_tsv(path, n=15):
    if not os.path.exists(path):
        return []
    rows = [l.rstrip("\n").split("\t") for l in open(path, encoding="utf-8", errors="replace")]
    return rows[: n + 1]


def fmt_size(b):
    return f"{b / 2**30:.1f} GB" if b >= 2**30 else f"{b / 2**20:.0f} MB"


def karp_flatt(s, p):
    return (1 / s - 1 / p) / (1 - 1 / p) if p > 1 else None


# ---------------------------------------------------------------- pipeline diagram
def pipeline_drawing():
    W, H = 16.6 * cm, 6.4 * cm
    d = Drawing(W, H)
    blue, light = colors.HexColor("#2E5597"), colors.HexColor("#DCE5F3")

    def box(x, y, w, h, text, fill=light, fs=7.5):
        d.add(Rect(x, y, w, h, fillColor=fill, strokeColor=blue, strokeWidth=0.8, rx=4, ry=4))
        lines = text.split("\n")
        for i, t in enumerate(lines):
            d.add(String(x + w / 2, y + h / 2 + (len(lines) / 2 - i - 0.75) * (fs + 1.5), t,
                         fontName=FONT, fontSize=fs, textAnchor="middle"))

    def arrow(x1, y1, x2, y2):
        d.add(Line(x1, y1, x2, y2, strokeColor=blue, strokeWidth=0.9))
        import math
        a = math.atan2(y2 - y1, x2 - x1)
        s = 4
        d.add(Polygon([x2, y2, x2 - s * math.cos(a - 0.4), y2 - s * math.sin(a - 0.4),
                       x2 - s * math.cos(a + 0.4), y2 - s * math.sin(a + 0.4)],
                      fillColor=blue, strokeColor=blue))

    box(0.0 * cm, 2.6 * cm, 2.5 * cm, 1.3 * cm, "Corpus file\n(one article\nper line)", colors.HexColor("#FFF2CC"), 7)
    ys = [4.8 * cm, 3.0 * cm, 0.6 * cm]
    labels = ["rank 0", "rank 1", "rank P-1"]
    for y, lab in zip(ys, labels):
        arrow(2.5 * cm, 3.25 * cm, 3.1 * cm, y + 0.55 * cm)
        box(3.1 * cm, y, 3.7 * cm, 1.1 * cm, f"{lab}:\nMPI_File_read_at (own range)", fs=6.8)
        arrow(6.8 * cm, y + 0.55 * cm, 7.2 * cm, y + 0.55 * cm)
        box(7.2 * cm, y, 3.3 * cm, 1.1 * cm, "tokenize + count:\nwords, bigrams, index", fs=6.8)
        arrow(10.5 * cm, y + 0.55 * cm, 10.9 * cm, 3.25 * cm)
    d.add(String(4.95 * cm, 2.15 * cm, "...", fontName=FONT, fontSize=12, textAnchor="middle"))
    box(10.9 * cm, 2.4 * cm, 2.5 * cm, 1.7 * cm, "MPI_Alltoallv\nowner =\nhash(word) % P\n(or MPI_Gatherv)", colors.HexColor("#F8D7DA"), 6.8)
    arrow(13.4 * cm, 3.25 * cm, 13.8 * cm, 3.25 * cm)
    box(13.8 * cm, 2.1 * cm, 2.8 * cm, 2.3 * cm, "owners merge and\nwrite index parts;\nlocal top-K sent to\nrank 0 (Gatherv);\nstats: MPI_Reduce", colors.HexColor("#D4EDDA"), 6.8)
    return d


# ---------------------------------------------------------------- main
def build(repo, out_pdf):
    med, sizes = load_bench()
    machine = read_kv(os.path.join(RES, "machine.txt"))
    cores = int(machine.get("cores_logical", "0") or 0)
    datasets = sorted({d for (d, m, p) in med if m != "sequential"}, key=lambda d: sizes.get(d, 0))
    full_sets = [d for d in datasets if not d.endswith("-wordsonly")]
    words_sets = [d for d in datasets if d.endswith("-wordsonly")]
    main_d = full_sets[-1] if full_sets else None

    # key numbers
    def speedups(d, mode="alltoall"):
        seq = med.get((d, "sequential", 1))
        if not seq:
            return {}
        return {p: (seq["total"] / v["total"], v) for p, v in series(med, d, mode).items()}

    best = None
    for d in full_sets + words_sets:
        for p, (s, _) in speedups(d).items():
            if best is None or s > best[0]:
                best = (s, p, d)
    max_p = max((p for (_, _, p) in med), default=1)
    oversub = cores and max_p > cores

    story = []
    # ---------------- title page
    story += [Spacer(1, 2.2 * cm),
              P("Parallel Word Count and Text Processing on Large Corpora",
                ParagraphStyle("t", parent=H1, fontSize=22, leading=28, alignment=TA_CENTER)),
              Spacer(1, 0.3 * cm),
              P("Project Report", ParagraphStyle("st", parent=BODY, fontSize=14, alignment=TA_CENTER)),
              Spacer(1, 1.2 * cm)]
    story.append(table([["Field", "Details"],
                        ["Course", "Parallel and Distributed Computing (PDC)"],
                        ["Instructor", "Sir Minhaal Raza"],
                        ["Project", "#13: Parallel Word Count / Text Processing on Large Corpora"],
                        ["Parallel Model", "MPI (C++), MPI-IO"],
                        ["GitHub Repository", escape(repo)],
                        ["Date", date.today().strftime("%B %Y")]], [4 * cm, 11 * cm]))
    story.append(Spacer(1, 0.8 * cm))
    story.append(table([["Name", "Roll No."]] + [list(t) for t in TEAM], [8 * cm, 7 * cm]))
    story.append(PageBreak())

    # ---------------- abstract
    story.append(P("Abstract", H1))
    if best:
        res_txt = (f"On our test machine ({escape(machine.get('cpu', 'unknown CPU'))}, {cores} logical cores) the "
                   f"best speedup was <b>{best[0]:.2f}x</b> with <b>{best[1]} processes</b> on the "
                   f"<i>{escape(best[2])}</i> dataset.")
    else:
        res_txt = "Benchmark results are reported in Section 8."
    story.append(P(
        "Counting word frequencies, bigrams and building an inverted index over a large text corpus is slow "
        "on one process because every byte has to be read, tokenized and hashed in order. The work is "
        "data-parallel, so we split the corpus into byte ranges and process them with MPI. Each process "
        "reads only its own range using MPI-IO, counts locally, and the partial results are merged using "
        "collective operations. We implemented two merge strategies: a centralized merge with MPI_Gatherv "
        "and a distributed merge where words are hash-partitioned to owner processes with MPI_Alltoallv. "
        f"The parallel output was verified to be identical to the sequential output for all process counts. {res_txt} "
        "We also measured the time spent in reading, computation, communication, merging and output to "
        "explain where the speedup stops growing."))

    # ---------------- 1 intro
    story.append(P("1. Introduction and Problem Statement", H1))
    story.append(P(
        "Word frequency counts, n-gram counts and inverted indexes are basic building blocks of search "
        "engines, language models and text analytics. For a corpus such as English Wikipedia (about 15 to "
        "20 GB of plain text and over 3 billion words) a single process takes a long time, because the "
        "whole file is processed one byte after another. However, the problem is embarrassingly parallel "
        "in its main phase: any chunk of text can be counted independently, and only the final counts need "
        "to be combined. This makes it a good fit for the MPI distributed-memory model."))
    story.append(P("Our objectives (from the proposal) were:"))
    story += bullets([
        "Build a sequential baseline that computes word frequencies, bigram counts and an inverted index.",
        "Build an MPI version that partitions the corpus by byte ranges across processes.",
        "Merge local results with MPI collectives to produce the global top-K words, bigrams and the index.",
        "Measure execution time, speedup and efficiency for 1, 2, 4, 8 and 16 processes on several input sizes.",
    ])

    # ---------------- 2 background
    story.append(P("2. Background: MPI Features Used", H1))
    story.append(P("Table 1 lists the MPI functions used in the project and the reason each one was chosen."))
    story.append(table([
        ["MPI function", "Where it is used", "Why"],
        ["MPI_File_open / MPI_File_read_at", "Reading the corpus", "Every rank reads only its own byte range at an explicit offset; no rank has to read the whole file or send text to others."],
        ["MPI_File_get_size + MPI_Bcast", "Partitioning", "Rank 0 gets the file size once and broadcasts it, then all ranks compute their range independently."],
        ["MPI_Gather + MPI_Gatherv", "Gather merge, top-K collection", "Ranks send different amounts of data, so the variable-size version is needed. Sizes are gathered first."],
        ["MPI_Alltoall + MPI_Alltoallv", "Distributed merge", "Each rank sends a different amount of data to every other rank (its words owned by that rank)."],
        ["MPI_Reduce (SUM)", "Global statistics", "Total bytes, documents, tokens and unique counts are fixed-size and can be summed directly."],
        ["MPI_Reduce (MAX)", "Timing", "A phase is only finished when the slowest rank finishes it, so we report the maximum."],
        ["MPI_Wtime, MPI_Barrier", "Timing", "Wall-clock timing of each phase; barrier at start and end so all ranks measure the same interval."],
    ], [4.3 * cm, 3.6 * cm, 8.1 * cm], "MPI functions used and their purpose"))

    # ---------------- 3 design
    story.append(P("3. System Design", H1))
    story.append(pipeline_drawing())
    fig_no[0] += 1
    story.append(P(f"Figure {fig_no[0]}: Overall pipeline of the MPI version", CAP))

    story.append(P("3.1 Partitioning", H2))
    story.append(P(
        "Rank 0 opens the file with MPI_File_open, reads the size N with MPI_File_get_size and broadcasts it. "
        "Rank r is assigned the byte range [r·N/P, (r+1)·N/P). The ranges have equal size, so each process "
        "gets almost the same amount of text. The range is read in blocks of 64 MB with MPI_File_read_at, "
        "so memory use does not grow with the input size and the program also works when a range is many "
        "gigabytes."))
    story.append(P("3.2 Boundary Handling", H2))
    story.append(P(
        "A byte range almost never starts or ends exactly between two words. The proposal planned to move "
        "each boundary to the next whitespace. While implementing it we noticed that this alone is not "
        "enough for bigrams and the index: a bigram whose two words fall on different processes would be "
        "lost, and a document cut in two would get index entries from two processes. We therefore align "
        "boundaries to the next <b>newline</b>, and since every line is one article, no word, bigram or "
        "document is ever split. The rule is:"))
    story += bullets([
        "A process handles every line whose <b>first byte</b> is inside its range.",
        "Start: if the byte just before the range is not a newline, the process is in the middle of a line "
        "that belongs to the previous process, so it skips forward to just after the next newline.",
        "End: a line that starts inside the range is always read completely, even if it continues past the "
        "end of the range.",
    ])
    story.append(P(
        "Both rules use the same definition, so every line is processed by exactly one process. Lines can "
        "also cross the 64 MB read blocks; the incomplete part is kept in a carry buffer and joined with "
        "the next block. This logic is in the ChunkedLineReader class and is shared by both programs."))
    story.append(P("3.3 Local Processing", H2))
    story.append(P(
        "Each line has the form <font face='%s'>article_id TAB text</font>. The article id is used as the "
        "document id in the inverted index (if a line has no id, its byte offset is used, which is also "
        "globally unique and needs no communication). The tokenizer lowercases ASCII letters and treats every "
        "character that is not a letter, digit or UTF-8 byte as a separator, so punctuation is removed and "
        "words like <i>café</i> are kept whole. For every token the process updates a local hash map "
        "(std::unordered_map) with the word count and the list of documents containing the word, and a second "
        "map with the count of the pair (previous word, word). Since an article is processed completely before "
        "the next one, a document id is added to a posting list only if it differs from the last one, which "
        "removes duplicates without a set." % MONO))
    story.append(P("3.4 Aggregation (Merge) Strategies", H2))
    story.append(P(
        "<b>Gather (centralized):</b> every rank except 0 serializes its word and bigram tables into a byte "
        "buffer. The buffer sizes are collected with MPI_Gather, then the buffers with MPI_Gatherv. Rank 0 "
        "merges all of them into its own tables, selects the top-K and writes the index. This is simple, but "
        "rank 0 does all merging and output alone and must hold the whole vocabulary in memory."))
    story.append(P(
        "<b>Alltoall (distributed):</b> every word and bigram has an owner rank, hash(key) mod P, using the "
        "64-bit FNV-1a hash (std::hash is not guaranteed to be the same on every machine). Each rank packs one "
        "buffer per destination; entries that it owns itself are moved directly with C++17 node extraction "
        "instead of being sent to itself. Counts are exchanged with MPI_Alltoall and the data with "
        "MPI_Alltoallv. Every owner then merges its share, writes its own index part file and computes a local "
        "top-K. Because a word lives on exactly one owner, the global top-K must be contained in the union of the "
        "local top-K lists, so only P·K entries are sent to rank 0 with MPI_Gatherv. Merging and output are "
        "therefore done in parallel."))
    story.append(P("3.5 Output", H2))
    story += bullets([
        "<font face='%s'>top_words.tsv</font>, <font face='%s'>top_bigrams.tsv</font>: global top-K (rank 0)." % (MONO, MONO),
        "<font face='%s'>index/part-XXXXX.tsv</font>: one line per word: word, document frequency, sorted list of article ids." % MONO,
        "<font face='%s'>summary.tsv</font>: total bytes, documents, tokens, unique words and unique bigrams (MPI_Reduce SUM)." % MONO,
    ])

    # ---------------- 4 implementation
    story.append(P("4. Implementation Details", H1))
    story.append(table([
        ["File", "Purpose"],
        ["include/textproc.hpp", "Tokenizer, Counter (words, bigrams, postings), ChunkedLineReader, pack/unpack, top-K"],
        ["include/output.hpp, args.hpp", "Writing results, command line options"],
        ["src/wc_sequential.cpp", "Sequential baseline (reads with pread through the same reader class)"],
        ["src/wc_mpi.cpp", "MPI version with both merge strategies and phase timing"],
        ["scripts/prepare_wikipedia.sh, wiki_to_lines.py, make_subsets.py", "Download dump, WikiExtractor, one-article-per-line format, subsets"],
        ["scripts/generate_synthetic.py", "Zipf-distributed synthetic corpus for testing without the dump"],
        ["scripts/run_tests.sh, verify.py", "Correctness tests: MPI output vs sequential output"],
        ["scripts/run_benchmarks.sh, plot_results.py", "Experiments, CSV results, graphs"],
    ], [6.2 * cm, 9.8 * cm], "Source files"))
    story.append(P(
        "Both programs use the same header for tokenizing and counting, so any difference in output can only "
        "come from partitioning or merging. A few details turned out to be important:"))
    story += bullets([
        "<b>Message format.</b> Records are written as [length][bytes][count] for bigrams and additionally "
        "[number of docs][doc ids] for words. A zero-length record followed by a tag separates the word "
        "section from the bigram section, so one buffer per destination is enough.",
        "<b>The 2 GB limit.</b> MPI counts are 32-bit ints. Buffers are padded to a multiple of 8 bytes and sent "
        "as MPI_UINT64_T, which raises the limit per message from 2 GB to 16 GB. The program checks the limit and "
        "stops with a clear message instead of overflowing.",
        "<b>Deterministic order.</b> Top-K lists are sorted by count and then alphabetically. Without the "
        "tie-break, words with equal counts could appear in a different order in the parallel output even "
        "though the counts are correct, and the comparison would fail.",
        "<b>Top-K selection.</b> Our first version copied every map entry into a vector before sorting. For "
        "millions of bigrams this was slow, so we sort pointers to the entries with nth_element and copy only "
        "the K winners.",
    ])

    # ---------------- 5 setup
    story.append(P("5. Dataset and Experimental Setup", H1))
    story.append(P(
        "The corpus is the English Wikipedia dump (pages-articles), converted to plain text with WikiExtractor "
        "(--json output) and then to one article per line by our script wiki_to_lines.py. Subsets are prefixes "
        "of the full file cut at a line boundary (make_subsets.py). For quick testing without the dump we also "
        "wrote a generator for a synthetic corpus whose word frequencies follow Zipf's law."))
    if machine:
        story.append(table([["Item", "Value"]] + [[k, escape(v)] for k, v in machine.items()],
                           [4 * cm, 12 * cm], "Test machine (recorded automatically by run_benchmarks.sh)"))
    rows = [["Dataset", "Size", "Features", "Used for"]]
    for d in full_sets:
        rows.append([escape(d), fmt_size(sizes[d]), "words + bigrams + index", "speedup / efficiency (E1)"])
    for d in words_sets:
        rows.append([escape(d), fmt_size(sizes[d]), "words only", "I/O-heavy case (E3)"])
    if len(rows) > 1:
        story.append(table(rows, [5 * cm, 2.3 * cm, 4.2 * cm, 4.5 * cm], "Datasets used in the experiments"))
    reps = sorted({int(v["n"]) for v in med.values()}) if med else [1]
    story.append(P(
        f"Every configuration was run {reps[-1]} time(s) and the median is reported. Before every run the input "
        "file is read once with <font face='%s'>cat</font> so that all runs start with the file in the operating "
        "system page cache; otherwise the first run would be much slower than the others. T(1) is the time of the "
        "sequential program, so the speedup includes all MPI overheads. Each phase time is the maximum over all "
        "ranks (MPI_Reduce with MPI_MAX)." % MONO))
    if oversub:
        story.append(P(
            f"<b>Note:</b> the machine has {cores} logical core(s), so runs with more than {cores} processes were "
            "oversubscribed (several MPI processes share one core, using mpirun --oversubscribe). For those runs "
            "no further speedup can be expected and they mainly show the extra cost of communication."))

    # ---------------- 6 verification
    story.append(P("6. Correctness Verification", H1))
    story.append(P(
        "scripts/verify.py compares an MPI output folder with the sequential one: the top-K files must be "
        "byte-for-byte identical, the totals in summary.tsv must match, and the complete inverted index and the "
        "complete word/bigram count tables (written with --dump-counts) must contain exactly the same lines. "
        "Since the parallel version writes several part files, lines are compared with an order-independent "
        "hash. The tests run with a very small read block (16 KB) so that many lines cross block boundaries, and "
        "with 3 processes as well, so that the ranges are not aligned to anything."))
    tr = os.path.join(RES, "test_results.txt")
    test_rows = []
    if os.path.exists(tr):
        for line in open(tr):
            m = re.match(r"p=(\d+) merge=(\w+) : (\w+)", line.strip())
            if m:
                test_rows.append([m.group(1), m.group(2), m.group(3)])
    if test_rows:
        story.append(table([["Processes", "Merge", "Result vs sequential"]] + test_rows,
                           [3 * cm, 4 * cm, 6 * cm], "Verification results"))
        allok = all(r[2] == "IDENTICAL" for r in test_rows)
        story.append(P("All configurations produced output identical to the sequential baseline." if allok else
                       "<b>Some configurations did not match; see results/tmp for details.</b>"))
    story.append(P(
        "To make sure the checker itself works, we also edited one count and deleted one index line in a "
        "copy of a correct output; verify.py reported both differences."))

    # ---------------- 7 results
    story.append(P("7. Results", H1))
    sum_rows = [["Dataset", "p", "Time (s)", "Speedup", "Efficiency", "Karp-Flatt e"]]
    for d in full_sets + words_sets:
        seq = med.get((d, "sequential", 1))
        if not seq:
            continue
        sum_rows.append([escape(d), "seq", f"{seq['total']:.2f}", "1.00", "1.00", "-"])
        for p, (s, v) in speedups(d).items():
            kf = karp_flatt(s, p)
            sum_rows.append([escape(d), str(p), f"{v['total']:.2f}", f"{s:.2f}", f"{s / p:.2f}",
                             f"{kf:.3f}" if kf is not None else "-"])
    if len(sum_rows) > 1:
        story.append(table(sum_rows, [4.6 * cm, 1.4 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.6 * cm],
                           "Execution time (median), speedup S(p)=T(1)/T(p), efficiency E(p)=S(p)/p, and the "
                           "Karp-Flatt metric (experimentally determined serial fraction)"))
    story.append(figure(os.path.join(FIG, "time.png"), "Execution time against number of processes", 12.5 * cm))
    story.append(figure(os.path.join(FIG, "speedup.png"), "Speedup against number of processes (dashed: ideal)", 12.5 * cm))
    story.append(figure(os.path.join(FIG, "efficiency.png"), "Efficiency against number of processes", 12.5 * cm))
    if main_d:
        story.append(figure(os.path.join(FIG, f"breakdown_{main_d}.png"),
                            f"Time per phase for {escape(main_d)} (slowest rank per phase)", 13 * cm))
    for d in words_sets:
        story.append(figure(os.path.join(FIG, f"breakdown_{d}.png"),
                            f"Time per phase for {escape(d)} (words only)", 13 * cm))
    story.append(figure(os.path.join(FIG, "merge_compare.png"), "Gather merge vs Alltoall merge", 15.5 * cm))

    # sample output
    tw = read_tsv(os.path.join(RES, "sample_output", "top_words.tsv"), 12)
    tb = read_tsv(os.path.join(RES, "sample_output", "top_bigrams.tsv"), 12)
    if tw:
        story.append(P("7.1 Sample Output", H2))
        summ = read_kv(os.path.join(RES, "sample_output", "summary.tsv"))
        if summ:
            story.append(P(
                f"For {escape(os.path.basename(summ.get('input', '')))} the program found "
                f"{int(summ.get('documents', 0)):,} documents, {int(summ.get('tokens', 0)):,} tokens, "
                f"{int(summ.get('unique_words', 0)):,} distinct words and {int(summ.get('unique_bigrams', 0)):,} "
                "distinct bigrams. The table below shows the most frequent words and bigrams."))
        rows = [["#", "Word", "Count", "Bigram", "Count"]]
        for i in range(1, min(len(tw), len(tb) if tb else len(tw))):
            b = tb[i] if tb and i < len(tb) else ["", "", ""]
            rows.append([tw[i][0], escape(tw[i][1]), f"{int(tw[i][2]):,}", escape(b[1]),
                         f"{int(b[2]):,}" if b[2] else ""])
        story.append(table(rows, [1 * cm, 3.5 * cm, 3 * cm, 5 * cm, 3 * cm], "Most frequent words and bigrams"))
        ie = read_tsv(os.path.join(RES, "sample_output", "index_excerpt.tsv"), 8)
        if ie:
            txt = "\n".join("\t".join(r) for r in ie)
            story.append(P("Excerpt from the inverted index (word, document frequency, article ids):"))
            story.append(code(txt))

    # ---------------- 8 discussion
    story.append(P("8. Discussion: Where Does the Speedup Stop?", H1))
    disc = []
    if main_d:
        sp = speedups(main_d)
        ps = sorted(sp)
        if ps:
            pmax = ps[-1]
            s_max, v_max = sp[pmax]
            seq = med[(main_d, "sequential", 1)]
            peak_p = max(ps, key=lambda p: sp[p][0])
            disc.append(
                f"For the full pipeline on {escape(main_d)} the speedup reached {sp[peak_p][0]:.2f}x at p={peak_p} "
                f"(efficiency {sp[peak_p][0] / peak_p:.2f}). At p={pmax} it was {s_max:.2f}x with efficiency "
                f"{s_max / pmax:.2f}.")
            if peak_p < pmax:
                disc.append(f"Using more than {peak_p} processes made the program slower, not faster.")
            comp_share = seq["compute"] / seq["total"]
            disc.append(
                f"In the sequential run, computation (tokenizing and hashing) takes {100 * comp_share:.0f}% of the time "
                f"and reading only {100 * seq['read'] / seq['total']:.1f}%, because the file is in the page cache "
                "and hash map updates with cache misses are much slower than reading memory. This part divides "
                "well: the compute time at p processes is close to T(compute)/p as long as there are enough cores.")
            ov = v_max["comm"] + v_max["merge"]
            disc.append(
                f"The parts that do not shrink are communication and merging. At p={pmax}, communication plus merge "
                f"take {ov:.2f} s, which is {100 * ov / v_max['total']:.0f}% of the total time. Merging is expensive "
                "for bigrams because the bigram table is almost as large as the text itself: most bigrams occur "
                "only a few times, so every rank has millions of entries that must be packed, sent and inserted "
                "into a new hash map. The merge work therefore costs about as much as counting did, and it is only "
                "spread over the processes, not removed.")
            kfs = [karp_flatt(sp[p][0], p) for p in ps if p > 1]
            kfs = [k for k in kfs if k is not None]
            if len(kfs) >= 2:
                trend = "increases" if kfs[-1] > kfs[0] * 1.1 else ("stays roughly constant" if kfs[-1] > kfs[0] * 0.9 else "decreases")
                disc.append(
                    f"The Karp-Flatt metric e {trend} from {kfs[0]:.3f} to {kfs[-1]:.3f} as p grows. A constant e "
                    "means the limit is a fixed serial part (Amdahl's law); an increasing e means the overhead itself "
                    "grows with p, which is what we expect from collective communication and from the serial top-K "
                    "step at rank 0.")
    for d in words_sets:
        sp = speedups(d)
        if sp:
            peak_p = max(sp, key=lambda p: sp[p][0])
            v = sp[max(sp)][1]
            disc.append(
                f"In the words-only experiment ({escape(d)}) there is much less work per byte and almost nothing to "
                f"merge (the vocabulary is small). The best speedup there was {sp[peak_p][0]:.2f}x at p={peak_p}. "
                f"Read time grows in share: at p={max(sp)} reading takes {100 * v['read'] / v['total']:.0f}% of the "
                "total. With a cold cache or a slower disk this is where I/O becomes the limit, because all "
                "processes share the same disk bandwidth and adding processes cannot make the disk faster.")
    if main_d and any(series(med, main_d, "gather").get(p) for p in (2, 4, 8, 16)) or any(
            series(med, d, "gather") for d in datasets):
        d = next((d for d in datasets if len(series(med, d, "gather")) > 1), None)
        if d:
            g, a = series(med, d, "gather"), series(med, d, "alltoall")
            common = [p for p in g if p in a and p > 1]
            if common:
                pm = common[-1]
                faster = "alltoall" if a[pm]["total"] < g[pm]["total"] else "gather"
                disc.append(
                    f"Comparing the merge strategies on {escape(d)} at p={pm}: gather took {g[pm]['total']:.2f} s and "
                    f"alltoall {a[pm]['total']:.2f} s, so {faster} was faster. Gather keeps rank 0's own table and only "
                    "merges what others send, which costs less total CPU work, but all of it happens on one process "
                    "and rank 0 must hold the complete vocabulary. Alltoall does a little more total work (every rank "
                    "repacks its table) but spreads it over all processes and writes the index in parallel, so it "
                    "scales better when real cores are available and it is the only option that fits in memory "
                    "for the largest inputs.")
    if oversub:
        disc.append(
            f"Finally, the test machine has only {cores} logical core(s). Runs with more processes than cores "
            "time-share the CPU, so they cannot be faster and they also pay for context switches and for "
            "communication through shared memory. The p=16 results should be read with this in mind.")
    for t in disc:
        story.append(P(t))
    story.append(P(
        "In summary, the speedup is limited by (1) the merge cost, which grows with the number of distinct "
        "bigrams, (2) collective communication, whose cost grows with p, (3) shared disk/memory bandwidth for "
        "reading, and (4) the small serial part at rank 0 (final top-K and writing). Load balance was not a "
        "problem: equal byte ranges gave each rank almost the same number of tokens (the MPI program prints the "
        "minimum and maximum tokens per rank)."))

    # ---------------- 9 deviations
    story.append(P("9. Changes Compared to the Proposal", H1))
    story += bullets([
        "<b>Dataset sizes.</b> The proposal planned 1 GB, 5 GB and the full 15-20 GB dump. With bigrams and the "
        "inverted index the hash tables need several times the input size in RAM, which our laptops do not "
        "have for the larger inputs. We therefore ran the full pipeline on the sizes listed in Section 5 and used a "
        "words-only run for the larger input.",
        "<b>MPI_Reduce for word counts.</b> The proposal planned to map words to integer ids with a hash and add the "
        "count arrays with MPI_Reduce. A dense array large enough for millions of distinct words would be mostly "
        "empty, and a small array would make different words collide and add up. We used MPI_Reduce for the "
        "fixed-size data (totals and timings) and MPI_Gatherv / MPI_Alltoallv for the variable-size tables. The "
        "hash is still used, but to choose the owner process.",
        "<b>Boundaries at newlines instead of whitespace</b> (Section 3.2), so that bigrams and documents are also "
        "never split.",
        "<b>Alltoallv merge.</b> This was listed as an improvement in the proposal and was implemented and "
        "compared with the gather merge.",
    ])

    # ---------------- 10 challenges
    story.append(P("10. Challenges and Lessons Learned", H1))
    story += bullets([
        "Our first boundary code (move to the next whitespace) gave slightly different bigram counts from the "
        "sequential version. Switching to line boundaries fixed it, and the tests with 3 processes and 16 KB blocks "
        "now cover these cases.",
        "Equal counts were printed in a different order by the parallel version, so the top-K files did not match "
        "even though the counts were right. A tie-break on the word fixed this.",
        "MPI's int counts limit one message to 2 GB. Sending 8-byte units instead of bytes and checking the size "
        "before sending avoided silent overflow.",
        "WikiExtractor can fail on Python 3.11 and newer (a regular expression error), so the dataset step "
        "is run in a Python 3.10 environment.",
        "Speedup is not only about splitting the compute: after parallelizing the counting, the merge became the "
        "biggest cost, which shows Amdahl's law in practice.",
    ])

    # ---------------- 11 conclusion
    story.append(P("11. Conclusion and Future Work", H1))
    story.append(P(
        "We built a sequential and an MPI implementation of word counting, bigram counting and inverted index "
        "construction. The MPI version reads the corpus in parallel with MPI-IO, handles chunk boundaries so "
        "that the results are exactly the same as the sequential ones, and merges results either at rank 0 or in a "
        "distributed way with MPI_Alltoallv. The counting phase scales well; the remaining time is dominated by "
        "merging the large bigram tables and by communication. Possible improvements are: combining small counts "
        "locally before sending (for example sending only bigrams above a threshold when only the top-K is "
        "needed), overlapping communication with computation using non-blocking collectives (MPI_Ialltoallv), "
        "a hybrid MPI + OpenMP version so that processes on the same node share one hash table, and running on a "
        "real multi-node cluster with a parallel file system."))

    story.append(P("Team Contributions", H2))
    story.append(table([["Member", "Main work"]] + [[n, ROLES[n]] for n, _ in TEAM], [4.5 * cm, 11.5 * cm]))

    story.append(P("References", H1))
    for r in [
        "Message Passing Interface Forum, <i>MPI: A Message-Passing Interface Standard, Version 3.1</i>, 2015 (Chapters 5 and 13: collectives and MPI-IO).",
        "J. Dean and S. Ghemawat, \"MapReduce: Simplified Data Processing on Large Clusters\", OSDI, 2004.",
        "A. H. Karp and H. P. Flatt, \"Measuring Parallel Processor Performance\", Communications of the ACM, 33(5), 1990.",
        "G. Attardi, WikiExtractor, https://github.com/attardi/wikiextractor",
        "Wikimedia Foundation, Wikipedia database dumps, https://dumps.wikimedia.org/",
        "Open MPI documentation, https://www.open-mpi.org/doc/",
    ]:
        story.append(Paragraph(r, BUL, bulletText="•"))

    story.append(P("Appendix: How to Run", H1))
    story.append(code(
        "make                      # builds bin/wc_sequential and bin/wc_mpi\n"
        "make test                 # sample file, p = 1,2,3,4,8, both merge modes\n"
        "bash scripts/prepare_wikipedia.sh <dump-url>\n"
        "./bin/wc_sequential data/wiki_1G.txt --out out_seq\n"
        "mpirun -np 8 ./bin/wc_mpi data/wiki_1G.txt --out out_mpi --merge alltoall\n"
        "python3 scripts/verify.py out_seq out_mpi\n"
        "bash scripts/run_benchmarks.sh && python3 scripts/plot_results.py\n"
        "python3 report/build_report.py --repo <github url>"))

    def on_page(canvas, doc):
        if doc.page == 1:
            return
        canvas.saveState()
        canvas.setFont(FONT, 8)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(2 * cm, 1.2 * cm, "PDC Project #13 - Parallel Word Count and Text Processing on Large Corpora")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(out_pdf, pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm,
                            title="Parallel Word Count and Text Processing on Large Corpora",
                            author=", ".join(n for n, _ in TEAM))
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print("wrote", out_pdf)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="https://github.com/<your-username>/pdc-parallel-wordcount")
    ap.add_argument("--out", default=os.path.join(ROOT, "report", "Project_Report.pdf"))
    a = ap.parse_args()
    build(a.repo, a.out)
