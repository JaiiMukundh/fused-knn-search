# fused-knn-search

**GPU-Accelerated Exact Vector Similarity Search** — a fused distance–selection
CUDA kernel and multi-GPU sharding for large-scale embedding retrieval, with a
web frontend to drive it on a free Colab GPU.

**PCAP mini-project — team: Samarth Sukumar · Shlok Sharma · Jaii Mukundh**

> The fused kernel never materialises the n×q distance matrix — each thread
> streams the database keeping its own k-best in registers, then blocks merge
> in shared memory. The web UI below runs the real binaries on a free GPU and
> shows the fused-vs-two-pass gap, verified against a double-precision CPU
> reference.

## Objectives → code

| # | Objective | Implementation |
|---|---|---|
| 1 | **Fused distance + top-k selection** kernel vs. the two-pass alternative (distances, then reduction) | `src/fused_topk.cuh` (kernel + K-specialized instantiations), `src/knn_cuda.cu` (times both variants) |
| 2 | **Multi-GPU sharding**: rank-local top-k then a merge to global top-k; scaling across ranks/GPUs | `src/knn_mpi_cuda.cu` (MPI + CUDA, one GPU per rank), `src/knn_mpi_omp.cpp` (CPU analog) |
| 3 | **CPU baselines** (serial/OpenMP) and **ground-truth verification** of every result | `src/knn_omp.cpp`, double-precision reference + `verify_topk()` in `src/common.hpp` |

Datasets: SIFT1M / GIST1M (TEXMEX), ANN-Benchmarks sets (GloVe, NYTimes-256,
Fashion-MNIST-784) and synthetic vectors up to ~10M×768. All binaries generate
synthetic data inline today (`--seed`); loading the `.fvecs`/`.bvecs` benchmark
files is a small extension on top of the same interface.

## Quick start on a free Colab GPU

```bash
# 1. Colab: New notebook -> Runtime -> Change runtime type -> T4 GPU
# 2. one cell -- paste remote/colab_bootstrap.py, or this loader:
#      import urllib.request as u; exec(u.urlopen(
#        'https://raw.githubusercontent.com/JaiiMukundh/fused-knn-search/main/remote/colab_bootstrap.py').read())
# 3. it prints, within the first minute, the ssh tunnel + URLs:
#      ssh tunnel     -> ssh -p <port> root@bore.pub -i ~/.ssh/colab_pcap
#                        (terminal, rsync, VS Code Remote-SSH)
#      web frontend   -> search / sweeps / timings + top-k in the browser
#      shared terminal-> sshx link for your teammates
```

Full walkthrough, sharing and troubleshooting: **[`remote/COLAB_SETUP.md`](remote/COLAB_SETUP.md)**.

Locally (no GPU needed for the CPU side):

```bash
make check-cpu     # builds knn_omp + knn_mpi_omp, runs verified checks
```

## Web frontend (`frontend/`)

A single-page UI + JSON API served from the GPU box:

- **Search** — one exact kNN run: fused kernel ms, throughput (GB/s, GFLOP/s),
  two-pass ms + speedup, PASS/FAIL exactness check, and the top-k table
  (indices + distances) for the first queries (`--dump`).
- **Sweep** — fused vs two-pass across database sizes, chart + table.
- **Build** — `make gpu` / `make cpu` on the box, straight from the browser.
- Engine switch: `knn_cuda` on GPU, `knn_omp` as CPU fallback; every number
  comes from executing the compiled binaries and parsing their stdout.

It runs anywhere Python + Flask exist (`python3 frontend/server.py --port 8080`);
the Colab bootstrap serves it through a Cloudflare quick tunnel.

## Builds and checks

| Command | What it does |
|---|---|
| `make cpu` / `make gpu` | OpenMP + MPI/OpenMP binaries / CUDA + MPI/CUDA binaries |
| `make check-cpu` | builds CPU side, runs verified checks locally |
| `make check-gpu` | builds GPU side, runs verified checks (GPU box) |
| `make check` | both, skipping whatever this machine can't build |

`make gpu` targets the compute capability of the GPU present at build time
(T4 → sm_75, P100 → sm_60) and embeds matching SASS plus PTX (so a binary
from one session still runs on a *newer* GPU model later).

## Binaries

All binaries share the same flags:

```
--n N          database size            (default 1048576)
--d D          dimensionality           (default 128)
--q Q          number of queries        (default 32)
--k K          neighbours               (default 10; CUDA supports 1,4,8,10,16,32)
--seed S       dataset seed             (default 42)
--threads T    OpenMP threads           (0 = all, default 0)
--verify V     queries checked vs CPU reference (default 4, 0 = off)
--twopass 0|1  knn_cuda: also time two-pass variant (default 1)
--dump M       print top-k of the first M queries as [topk] lines (default 0)
```

```bash
./bin/knn_omp                      # OpenMP CPU baseline
mpirun -np 4 ./bin/knn_mpi_omp     # MPI + OpenMP shards the database
./bin/knn_cuda                     # single-GPU fused vs two-pass
mpirun -np 2 ./bin/knn_mpi_cuda    # multi-GPU: rank r uses device r % ndev
```

Every run prints `[run]` parameters, `[time]` results and, when verifying,
`[verify] … PASS/FAIL` lines. **Verification policy:** the top-1 neighbour
index must match the double-precision CPU reference exactly, and the k-th
distance must agree within a 1e-3 relative tolerance (float vs double).
Timings from a FAIL-ing configuration are meaningless.

## Layout

```
src/common.hpp        shared: data generation, CPU double reference, verify, args
src/fused_topk.cuh    fused distance+top-k CUDA kernel, two-pass variant, K dispatch
src/knn_cuda.cu       single-GPU driver (fused vs two-pass timing, --dump)
src/knn_mpi_cuda.cu   MPI multi-GPU sharding (local top-k -> merge)
src/knn_omp.cpp       OpenMP CPU baseline
src/knn_mpi_omp.cpp   MPI + OpenMP CPU baseline
frontend/server.py    Flask API driving the binaries (/api/search, /benchmark, /build)
frontend/index.html   single-page UI (search, sweeps, build, how-it-works)
remote/colab_bootstrap.py   Colab cell: ssh (bore) + Cloudflare tunnels + clone + build
remote/COLAB_SETUP.md       guide for the Colab route
Makefile              build targets (cpu / gpu / check*)
```

## Notes

- Timing methodology: warm-up iterations are excluded; `knn_cuda` reports
  kernel time plus end-to-end and compares fused vs two-pass on identical
  inputs. CPU baselines report the same metrics for the same `--n/--d/--q/--k`.
- The MPI sharding is exact: each rank searches its share of the database and
  the merge scans all ranks' candidates; results equal the single-GPU result
  (checked via `--verify`).
- Free-GPU realities: Colab runtimes recycle (keep the tab open; ~12 h max,
  ~90 min idle). Keep sweeps lean and log results to `out/`.
- Local compile check (no GPU needed): with a CUDA toolchain on the laptop
  (conda `cuda-nvcc` + `cuda-cudart-dev` work), `make gpu NVCC=<env>/bin/nvcc
  CUDA_HOME=<env> CUDA_ARCH_FLAGS='-arch=sm_75'` compiles and links both CUDA
  binaries — catches API breakage (e.g. CUDA 13 removing
  `cudaDeviceProp::memoryClockRate`) before burning GPU quota.
