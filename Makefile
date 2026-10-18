# PCAP mini-project -- GPU-accelerated exact kNN search, and its comparison
# against approximate indexes (FAISS).
#
#   make            build everything available on this machine
#   make cpu        bin/knn_omp + bin/gendata      (any Linux box, no GPU/MPI)
#   make gpu        bin/knn  (MPI + OpenMP + CUDA, one binary)   needs nvcc+mpicxx
#   make check      build + run correctness checks
#   make check-cpu  CPU checks only (safe without a GPU)
#   make check-gpu  GPU checks (run on the GPU box: Colab, over the ssh tunnel)
#   make compare    brute force vs FAISS: export data, then benchmark ANN
#
# bin/knn is the unified engine: src/knn.cu is ONE program that uses MPI for
# sharding, OpenMP for the host-side/CPU path and CUDA for the device kernels.
# It supersedes the previous knn_omp / knn_mpi_omp / knn_cuda / knn_mpi_cuda
# split (those sources are still in src/ but are no longer built by default;
# see the `legacy` target). bin/knn_omp is kept as the fallback for machines
# with neither MPI nor CUDA, which is what the laptop runs.

SHELL     := /bin/bash
CXX       ?= g++
MPICXX    ?= mpicxx
CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(shell command -v nvcc 2>/dev/null || echo $(CUDA_HOME)/bin/nvcc)
# Mandatory flags are baked in (an env-provided CXXFLAGS from conda toolchains
# must not silently drop -O3 -std=c++17); extensible via EXTRA_CXXFLAGS.
EXTRA_CXXFLAGS ?=
CXXFLAGS  := -O3 -std=c++17 -march=native $(EXTRA_CXXFLAGS)
OMPFLAGS  ?= -fopenmp
CUDA_ARCH_FLAGS ?=
MPIRUN    ?= mpirun
MPIRUN_FLAGS ?=
PYTHON    ?= python3
DATA_DIR  ?= data

# Build for the GPU that is actually visible (T4 = sm_75, P100 = sm_60, ...).
# Falls back to nvcc's own default when no GPU is present (syntax-check only).
COMPUTE_CAP := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
ifeq ($(shell echo '$(COMPUTE_CAP)' | grep -Eq '^[0-9]+$$' && echo numeric),numeric)
# SASS for this session's GPU, plus PTX so a binary from one session still runs
# on a newer GPU model that a later session might hand us.
CUDA_ARCH_FLAGS := -gencode arch=compute_$(COMPUTE_CAP),code=sm_$(COMPUTE_CAP) \
                   -gencode arch=compute_$(COMPUTE_CAP),code=compute_$(COMPUTE_CAP)
else
COMPUTE_CAP :=
endif

# nvcc needs the MPI include paths (mpi.h lives outside the default search path)
MPI_SHOW := $(shell $(MPICXX) -show 2>/dev/null || $(MPICXX) --showme:compile 2>/dev/null || true)
MPI_INC  := $(shell echo '$(MPI_SHOW)' | tr ' ' '\n' | grep '^-I' | tr '\n' ' ')

BIN := bin
SRC := src

.PHONY: all cpu gpu legacy check check-cpu check-gpu check-mpi compare gendata clean help

all: cpu gpu

cpu: $(BIN)/knn_omp $(BIN)/gendata

# One binary, MPI + OpenMP + CUDA. Compiled by nvcc (device code) and linked by
# mpicxx (MPI runtime), exactly as the old knn_mpi_cuda was -- only now it also
# carries the OpenMP CPU path and the MPI sharding in the same source.
gpu: $(BIN)/knn

$(BIN):
	@mkdir -p $(BIN)

# ---- CPU side (works anywhere) ---------------------------------------------
$(BIN)/knn_omp: $(SRC)/knn_omp.cpp $(SRC)/common.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<

$(BIN)/gendata: $(SRC)/gendata.cpp $(SRC)/common.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

# ---- unified engine --------------------------------------------------------
$(BIN)/knn.o: $(SRC)/knn.cu $(SRC)/fused_topk.cuh $(SRC)/common.hpp | $(BIN)
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "nvcc not found (looked at: $(NVCC)); build the GPU targets on the Colab box"; exit 1; }
	$(NVCC) -O3 -std=c++17 $(CUDA_ARCH_FLAGS) -DPCAP_USE_MPI $(MPI_INC) \
		-Xcompiler $(OMPFLAGS) -c -o $@ $(SRC)/knn.cu

$(BIN)/knn: $(BIN)/knn.o
	@command -v $(MPICXX) >/dev/null 2>&1 || { echo "mpicxx not found; install MPI (Colab: apt-get install -y mpich) or use bin/knn_omp"; exit 1; }
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $< -L$(CUDA_HOME)/lib64 -lcudart

# ---- superseded binaries (kept for reference; not built by default) --------
legacy: $(BIN)/knn_cuda $(BIN)/knn_mpi_cuda $(BIN)/knn_mpi_omp

$(BIN)/knn_mpi_omp: $(SRC)/knn_mpi_omp.cpp $(SRC)/common.hpp | $(BIN)
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<

$(BIN)/knn_cuda: $(SRC)/knn_cuda.cu $(SRC)/fused_topk.cuh $(SRC)/common.hpp | $(BIN)
	$(NVCC) -O3 -std=c++17 $(CUDA_ARCH_FLAGS) -o $@ $(SRC)/knn_cuda.cu

$(BIN)/knn_mpi_cuda.o: $(SRC)/knn_mpi_cuda.cu $(SRC)/fused_topk.cuh $(SRC)/common.hpp | $(BIN)
	$(NVCC) -O3 -std=c++17 $(CUDA_ARCH_FLAGS) $(MPI_INC) -c -o $@ $(SRC)/knn_mpi_cuda.cu

$(BIN)/knn_mpi_cuda: $(BIN)/knn_mpi_cuda.o
	$(MPICXX) $(CXXFLAGS) -o $@ $< -L$(CUDA_HOME)/lib64 -lcudart

# ---- checks ----------------------------------------------------------------
check: check-cpu
	@if [ -x "$(NVCC)" ] || command -v $(NVCC) >/dev/null 2>&1; then \
		$(MAKE) check-gpu; \
	else \
		echo "[check] nvcc not found: skipping GPU checks (expected on a laptop; run 'make check-gpu' on the GPU box: Colab)"; \
	fi

check-cpu: cpu
	./$(BIN)/knn_omp --n 200000 --d 128 --q 8 --k 10 --verify 2
	@echo "[ok  ] bin/gendata exports the shared dataset for the FAISS comparison"

# bin/knn with -np 1 must reproduce the old knn_cuda numbers; -np 2 must merge
# back to exactly the same verified result.
check-gpu: gpu
	./$(BIN)/knn --n 1000000 --d 128 --q 16 --k 10 --verify 2 --twopass 1
	@if command -v $(MPIRUN) >/dev/null 2>&1; then \
		$(MPIRUN) -np 2 $(MPIRUN_FLAGS) ./$(BIN)/knn --n 1000000 --d 128 --q 16 --k 10 --verify 2; \
	else \
		echo "[check] mpirun not found: skipping the sharded run"; \
	fi

# Objective 3: brute force vs the established approximate index (FAISS).
# Runs on the GPU box; faiss-gpu is installed by remote/colab_bootstrap.py.
compare: $(BIN)/gendata
	./$(BIN)/gendata --n 1000000 --d 128 --q 1000 --k 10 --seed 42 --out $(DATA_DIR)
	$(PYTHON) bench/faiss_compare.py --data $(DATA_DIR) --k 10 --json $(DATA_DIR)/faiss.json

clean:
	rm -rf $(BIN)

help:
	@echo "make cpu        -- bin/knn_omp + bin/gendata (any Linux box)"
	@echo "make gpu        -- bin/knn (MPI + OpenMP + CUDA in one binary; needs nvcc + mpicxx)"
	@echo "make check-cpu  -- run small correctness checks (no GPU needed)"
	@echo "make check-gpu  -- run verified checks on the GPU box (Colab)"
	@echo "make compare    -- export data, then brute force vs FAISS (ANN)"
	@echo "make legacy     -- build the superseded knn_cuda/knn_mpi_* binaries"
	@echo "make clean"
	@echo
	@echo "Examples:"
	@echo "  mpirun -np 1 ./bin/knn --n 1000000 --d 128 --q 32 --k 10"
	@echo "  mpirun -np 4 ./bin/knn --n 1000000 --d 128 --q 32 --k 10   # one GPU per rank"
	@echo "  ./bin/knn --device cpu --threads 8 --n 200000 --d 128 --q 8 --k 10"
