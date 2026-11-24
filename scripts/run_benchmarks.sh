#!/usr/bin/env bash
# run_benchmarks.sh - runs every experiment in the report and appends the
# timings to results/benchmarks.csv. Then run: python3 scripts/plot_results.py
#
# Everything is configurable through environment variables, e.g.
#   DATASETS="data/wiki_256M.txt data/wiki_1G.txt" PROCS="1 2 4 8" REPEAT=3 bash scripts/run_benchmarks.sh
#
# Experiments:
#   E1 full pipeline (words + bigrams + index), alltoall merge, all DATASETS x PROCS
#   E2 gather vs alltoall merge on COMPARE_DATASET
#   E3 words only (no bigrams, no index) on WORDS_ONLY_DATASET -> lighter compute,
#      shows where disk I/O starts to limit speedup
set -e
DATASETS=${DATASETS:-"data/wiki_256M.txt data/wiki_1G.txt"}
PROCS=${PROCS:-"1 2 4 8 16"}
REPEAT=${REPEAT:-3}
COMPARE_DATASET=${COMPARE_DATASET:-data/wiki_256M.txt}
WORDS_ONLY_DATASET=${WORDS_ONLY_DATASET:-data/wiki_5G.txt}
MPIRUN=${MPIRUN:-mpirun}
# --oversubscribe allows more ranks than physical cores (needed for p=16 on most laptops)
MPI_FLAGS=${MPI_FLAGS:---oversubscribe}
CSV=${CSV:-results/benchmarks.csv}
OUT=results/tmp/bench_out

mkdir -p results "$OUT"
make -s all

# ---- record the machine (goes into the report) ----
{
  echo "date: $(date '+%Y-%m-%d %H:%M')"
  echo "cpu: $(lscpu | grep 'Model name' | sed 's/Model name:\s*//')"
  echo "cores_logical: $(nproc)"
  echo "cores_physical: $(lscpu -p=core,socket | grep -v '^#' | sort -u | wc -l)"
  echo "ram: $(free -h | awk '/Mem:/ {print $2}')"
  echo "os: $(. /etc/os-release && echo "$PRETTY_NAME")"
  echo "compiler: $(g++ --version | head -1)"
  echo "mpi: $($MPIRUN --version 2>&1 | head -1)"
  echo "disk: $(df -h . | awk 'NR==2 {print $1}')"
} > results/machine.txt
cat results/machine.txt

warm_cache() { cat "$1" > /dev/null; }   # same starting point for every run (file in page cache)

run_seq() {  # dataset label flags...
  local f=$1 label=$2; shift 2
  for r in $(seq 1 "$REPEAT"); do
    warm_cache "$f"
    ./bin/wc_sequential "$f" --out "$OUT" --csv "$CSV" --dataset "$label" "$@" | sed -n 2p
  done
}
run_mpi() {  # dataset label procs mode flags...
  local f=$1 label=$2 p=$3 mode=$4; shift 4
  for r in $(seq 1 "$REPEAT"); do
    warm_cache "$f"
    $MPIRUN $MPI_FLAGS -np "$p" ./bin/wc_mpi "$f" --out "$OUT" --merge "$mode" --csv "$CSV" --dataset "$label" "$@" | sed -n 2p
  done
}
label_of() { basename "$1" .txt; }

echo "== E1: full pipeline, alltoall =="
for f in $DATASETS; do
  [ -f "$f" ] || { echo "missing $f, skipping"; continue; }
  L=$(label_of "$f")
  echo "-- $L sequential"; run_seq "$f" "$L"
  for p in $PROCS; do echo "-- $L p=$p"; run_mpi "$f" "$L" "$p" alltoall; done
done

echo "== E2: gather vs alltoall =="
if [ -f "$COMPARE_DATASET" ]; then
  L=$(label_of "$COMPARE_DATASET")
  for p in $PROCS; do
    [ "$p" = 1 ] && continue
    echo "-- $L p=$p gather"; run_mpi "$COMPARE_DATASET" "$L" "$p" gather
  done
fi

echo "== E3: words only =="
if [ -f "$WORDS_ONLY_DATASET" ]; then
  L="$(label_of "$WORDS_ONLY_DATASET")-wordsonly"
  echo "-- $L sequential"; run_seq "$WORDS_ONLY_DATASET" "$L" --no-bigrams --no-index
  for p in $PROCS; do echo "-- $L p=$p"; run_mpi "$WORDS_ONLY_DATASET" "$L" "$p" alltoall --no-bigrams --no-index; done
fi

# keep one real output (top words etc.) for the report
mkdir -p results/sample_output
f=$(echo $DATASETS | awk '{print $1}')
if [ -f "$f" ]; then
  $MPIRUN $MPI_FLAGS -np 4 ./bin/wc_mpi "$f" --out results/tmp/report_out --topk 25 > results/sample_output/console.txt
  cp results/tmp/report_out/top_words.tsv results/tmp/report_out/top_bigrams.tsv results/tmp/report_out/summary.tsv results/sample_output/
  # a few inverted-index lines (posting lists shortened for display)
  : > results/sample_output/index_excerpt.tsv
  for w in parallel computer university karachi pakistan science; do
    grep -h -P "^$w\t" results/tmp/report_out/index/part-*.tsv | awk -F'\t' '{p=$3; if (length(p)>70) p=substr(p,1,70)"..."; print $1"\t"$2"\t"p}' >> results/sample_output/index_excerpt.tsv || true
  done
  if [ ! -s results/sample_output/index_excerpt.tsv ]; then
    sort -t$'\t' -k2,2nr results/tmp/report_out/index/part-00000.tsv | head -6 | awk -F'\t' '{p=$3; if (length(p)>70) p=substr(p,1,70)"..."; print $1"\t"$2"\t"p}' > results/sample_output/index_excerpt.tsv
  fi
fi
echo "done -> $CSV   (now run: python3 scripts/plot_results.py)"
