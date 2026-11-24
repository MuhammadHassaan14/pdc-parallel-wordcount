# Parallel Word Count and Text Processing on Large Corpora
CXX      ?= g++
MPICXX   ?= mpic++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
INCLUDES  = -Iinclude
# skip the deprecated MPI C++ bindings (removes noisy OpenMPI warnings)
MPIFLAGS  = -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX
HEADERS   = include/textproc.hpp include/output.hpp include/args.hpp

all: bin/wc_sequential bin/wc_mpi

bin/wc_sequential: src/wc_sequential.cpp $(HEADERS)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

bin/wc_mpi: src/wc_mpi.cpp $(HEADERS)
	@mkdir -p bin
	$(MPICXX) $(CXXFLAGS) $(MPIFLAGS) $(INCLUDES) -o $@ $<

# Quick correctness test on the small sample file (sequential vs 1..4 ranks, both merge modes)
test: all
	bash scripts/run_tests.sh

clean:
	rm -rf bin output out_* results/tmp

.PHONY: all test clean
