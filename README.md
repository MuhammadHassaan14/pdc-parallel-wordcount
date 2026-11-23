# Parallel Word Count and Text Processing on Large Corpora (MPI)

**Course:** Parallel and Distributed Computing (PDC) — Project #13
**Instructor:** Sir Minhaal Raza
**Team:** Fozan Javaid (23K-0605) · Muhammad Hassaan (23K-0570) · Ajay Kumar (23K-0514)

Counts word frequencies and bigrams and builds an inverted index (word → document IDs)
over a multi-GB English Wikipedia corpus. A sequential C++ baseline is compared with an
MPI version that splits the file by byte ranges and reads it with MPI-IO.

## Features

- **Sequential baseline** (`bin/wc_sequential`): word counts, bigram counts, inverted index, top-K.
- **MPI version** (`bin/wc_mpi`):
  - Rank 0 gets the file size and broadcasts it; each rank takes `[r·N/P, (r+1)·N/P)`.
  - Each rank reads only its own range with `MPI_File_read_at`, in 64 MB blocks.
  - Boundary handling: a rank owns every line whose first byte is inside its range, so no word,
    bigram or document is split or counted twice.
  - Two merge strategies (`--merge`):
    - `gather` — every rank sends its tables to rank 0 with `MPI_Gatherv`; rank 0 merges.
    - `alltoall` (default) — each word is owned by rank `hash(word) % P`; tables are exchanged with
      `MPI_Alltoallv`, so merging and index writing happen in parallel. Owners then send their local
      top-K to rank 0.
  - Totals with `MPI_Reduce(SUM)`, phase timings (`MPI_Wtime`) with `MPI_Reduce(MAX)`.
- **Verification** (`scripts/verify.py`): checks that MPI output is identical to sequential output.

## Repository layout

```
include/textproc.hpp     tokenizer, counting, chunked line reader (boundary handling), packing
include/output.hpp       writing result files
include/args.hpp         command line options
src/wc_sequential.cpp    sequential baseline
src/wc_mpi.cpp           MPI version
scripts/prepare_wikipedia.sh   download dump + WikiExtractor + convert + subsets
scripts/wiki_to_lines.py       WikiExtractor JSON -> one article per line
scripts/make_subsets.py        cut 256M / 1G / 5G subsets
scripts/generate_synthetic.py  Zipf synthetic corpus for quick tests
scripts/run_tests.sh           correctness tests (sequential vs MPI, p = 1,2,3,4,8)
scripts/verify.py              compares two output folders
scripts/run_benchmarks.sh      all timing experiments -> results/benchmarks.csv
scripts/plot_results.py        graphs + summary table
report/build_report.py         builds the PDF report from the results
data/sample.txt                tiny test file with edge cases
```

## Requirements

Linux (or WSL2 on Windows), `g++` (C++17), OpenMPI or MPICH, Python 3 with `matplotlib` and `numpy`.

```bash
sudo apt install build-essential openmpi-bin libopenmpi-dev python3-pip
pip install matplotlib numpy reportlab
```

## Build and quick test

```bash
make
make test                   # sample file, all process counts, both merge modes
```

## Usage

```bash
./bin/wc_sequential data/wiki_1G.txt --out out_seq
mpirun -np 8 ./bin/wc_mpi data/wiki_1G.txt --out out_mpi --merge alltoall
python3 scripts/verify.py out_seq out_mpi
```

Options: `--topk K`, `--no-bigrams`, `--no-index`, `--dump-counts`, `--block-mb N`,
`--csv FILE`, `--dataset NAME`, `--merge gather|alltoall`.

Output folder:

| File | Content |
|------|---------|
| `top_words.tsv`, `top_bigrams.tsv` | global top-K |
| `index/part-XXXXX.tsv` | `word <TAB> doc_freq <TAB> doc_id,doc_id,...` (one part per owner rank) |
| `summary.tsv` | bytes, documents, tokens, unique words, unique bigrams |
| `counts/part-XXXXX.tsv` | full count tables (only with `--dump-counts`) |

## Dataset

```bash
# full dump (~22 GB download) or pass the URL of a smaller part file
bash scripts/prepare_wikipedia.sh
# no internet / quick test: synthetic Zipf corpus
python3 scripts/generate_synthetic.py --size-mb 500 --out data/synthetic_500MB.txt
```

Corpus format: one article per line, `<article id> TAB <title>. <text>`.

## Reproducing the results

```bash
bash scripts/run_tests.sh data/wiki_256M.txt | tee results/test_results.txt
bash scripts/run_benchmarks.sh          # see variables at the top of the script
python3 scripts/plot_results.py
python3 report/build_report.py          # -> report/Project_Report.pdf
```

Phase timings: **read** (MPI-IO), **compute** (tokenize + count), **comm** (collectives and waiting
for slower ranks), **merge** (packing, unpacking, merging, top-K), **output** (writing files).
Each phase is the maximum over all ranks.
