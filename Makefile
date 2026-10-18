# PCAP mini-project -- GPU-accelerated exact kNN search
# Targets: CUDA (primary), OpenMP CPU baselines, MPI sharding (CPU + GPU).
#
#   make            build everything that is available on this machine
#   make cpu        OpenMP + MPI/OpenMP baselines   (works on any Linux box)
#   make gpu        CUDA + MPI/CUDA binaries        (needs nvcc)
#   make check      build + run small correctness checks
#   make check-cpu  only the CPU checks (safe without a GPU)
#   make check-gpu  only the GPU checks (run this on the GPU box: Colab/Kaggle, over the tunnel)

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

# Build for the GPU that is actually visible (P100 = sm_60, T4 = sm_75, ...).
# Falls back to nvcc's own default when no GPU is present (syntax-check only).
# Some drivers print "N/A"; keep the value only when it is purely numeric.
COMPUTE_CAP := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
ifeq ($(shell echo '$(COMPUTE_CAP)' | grep -Eq '^[0-9]+$$' && echo numeric),numeric)
# SASS for this session's GPU, plus PTX so CUDA can JIT the binary for a
# newer GPU model that a later session might hand us.
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

.PHONY: all cpu gpu check check-cpu check-gpu clean help

all: cpu gpu

cpu: $(BIN)/knn_omp $(BIN)/knn_mpi_omp

gpu: $(BIN)/knn_cuda $(BIN)/knn_mpi_cuda

$(BIN):
	@mkdir -p $(BIN)

$(BIN)/knn_omp: $(SRC)/knn_omp.cpp $(SRC)/common.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<

$(BIN)/knn_mpi_omp: $(SRC)/knn_mpi_omp.cpp $(SRC)/common.hpp | $(BIN)
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<

$(BIN)/knn_cuda: $(SRC)/knn_cuda.cu $(SRC)/fused_topk.cuh $(SRC)/common.hpp | $(BIN)
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "nvcc not found (looked at: $(NVCC)); build the GPU targets on the Kaggle box"; exit 1; }
	$(NVCC) -O3 -std=c++17 $(CUDA_ARCH_FLAGS) -o $@ $(SRC)/knn_cuda.cu

$(BIN)/knn_mpi_cuda.o: $(SRC)/knn_mpi_cuda.cu $(SRC)/fused_topk.cuh $(SRC)/common.hpp | $(BIN)
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "nvcc not found (looked at: $(NVCC))"; exit 1; }
	$(NVCC) -O3 -std=c++17 $(CUDA_ARCH_FLAGS) $(MPI_INC) -c -o $@ $(SRC)/knn_mpi_cuda.cu

$(BIN)/knn_mpi_cuda: $(BIN)/knn_mpi_cuda.o
	$(MPICXX) $(CXXFLAGS) -o $@ $< -L$(CUDA_HOME)/lib64 -lcudart

# ---------------------------------------------------------------------------
check: check-cpu
	@if [ -x "$(NVCC)" ] || command -v $(NVCC) >/dev/null 2>&1; then \
		$(MAKE) check-gpu; \
	else \
		echo "[check] nvcc not found: skipping GPU checks (expected on a laptop; run 'make check-gpu' on the GPU box: Colab or Kaggle)"; \
	fi

check-cpu: cpu
	./$(BIN)/knn_omp --n 200000 --d 128 --q 8 --k 10 --verify 2
	@if command -v $(MPIRUN) >/dev/null 2>&1; then \
		$(MPIRUN) -np 2 $(MPIRUN_FLAGS) ./$(BIN)/knn_mpi_omp --n 200000 --d 128 --q 8 --k 10 --verify 2; \
	else \
		echo "[check] mpirun not found: skipping knn_mpi_omp"; \
	fi

check-gpu: gpu
	./$(BIN)/knn_cuda --n 1000000 --d 128 --q 16 --k 10 --verify 2
	@if command -v $(MPIRUN) >/dev/null 2>&1; then \
		$(MPIRUN) -np 2 $(MPIRUN_FLAGS) ./$(BIN)/knn_mpi_cuda --n 1000000 --d 128 --q 16 --k 10 --verify 2; \
	else \
		echo "[check] mpirun not found: skipping knn_mpi_cuda"; \
	fi

clean:
	rm -rf $(BIN)

help:
	@echo "make cpu        -- build OpenMP + MPI/OpenMP CPU baselines"
	@echo "make gpu        -- build CUDA + MPI/CUDA binaries (needs nvcc)"
	@echo "make check-cpu  -- run small correctness checks on CPU"
	@echo "make check-gpu  -- run small correctness checks on GPU (Colab/Kaggle)"
	@echo "make clean"
	@echo
	@echo "Examples:"
	@echo "  ./bin/knn_cuda      --n 1000000 --d 128 --q 32 --k 10"
	@echo "  $(MPIRUN) -np 2 ./bin/knn_mpi_cuda --n 1000000 --d 128 --q 32 --k 10"
