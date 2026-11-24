#!/usr/bin/env bash
# run_tests.sh - correctness tests: sequential output vs MPI output for
# several process counts and both merge strategies.
#
# Usage: bash scripts/run_tests.sh [corpus.txt]   (default: data/sample.txt)
set -e
INPUT=${1:-data/sample.txt}
MPIRUN=${MPIRUN:-mpirun}
# --oversubscribe lets us test 8 ranks on a laptop with fewer cores
MPI_FLAGS=${MPI_FLAGS:---oversubscribe}
TMP=results/tmp
rm -rf "$TMP"; mkdir -p "$TMP"

# small block size on purpose: forces lines to cross read-block boundaries
# (16 KB blocks, so long articles are split over several blocks)
./bin/wc_sequential "$INPUT" --out "$TMP/seq" --dump-counts ${BLOCK:---block-kb 16} > /dev/null

fail=0
for p in 1 2 3 4 8; do
  for mode in gather alltoall; do
    out="$TMP/mpi_${p}_${mode}"
    $MPIRUN $MPI_FLAGS -np $p ./bin/wc_mpi "$INPUT" --out "$out" --merge $mode --dump-counts ${BLOCK:---block-kb 16} > /dev/null
    if python3 scripts/verify.py "$TMP/seq" "$out" > "$out.verify.txt"; then
      echo "p=$p merge=$mode : IDENTICAL"
    else
      echo "p=$p merge=$mode : MISMATCH (see $out.verify.txt)"; fail=1
    fi
  done
done
exit $fail
