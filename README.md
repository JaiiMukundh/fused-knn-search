# fused-knn-search

**GPU-Accelerated Exact Vector Similarity Search** — a fused distance–selection
CUDA kernel, MPI multi-GPU sharding and OpenMP baselines in one binary, plus a
measured comparison against FAISS approximate indexes, with a web frontend to
drive it on a free Colab GPU.

**PCAP mini-project — team: Samarth Sukumar · Shlok Sharma · Jaii Mukundh**

> The fused kernel never materialises the n×q distance matrix — each thread
> streams the database keeping its own k-best in registers, then blocks merge
> in shared memory. The web UI below runs the real binaries on a free GPU and
> shows the fused-vs-two-pass gap, verified against a double-precision CPU
> reference, and the brute-force-vs-ANN recall/latency trade-off.

## Objectives → code

| # | Objective | Implementation |
|---|---|---|
| 1 | **Fused distance + top-k selection** kernel vs. the two-pass alternative (distances, then reduction) | `src/fused_topk.cuh` (kernel + K-specialized instantiations); `src/knn.cu` times both variants |
| 2 | **Multi-GPU sharding**: rank-local top-k then a merge to global top-k; scaling across ranks/GPUs | `src/knn.cu` (MPI + OpenMP + CUDA in one binary; rank r owns rows `[r·n/size, (r+1)·n/size)`) |
| 3 | **CPU baselines** (serial/OpenMP), **ground-truth verification** of every result, and the **comparison against established approximate indexes** | `src/knn.cu --device cpu` (OpenMP baseline) + double-precision reference/`verify_topk()` in `src/common.hpp`; `bench/faiss_compare.py` benchmarks FAISS IVF-PQ and HNSW on byte-identical vectors and measures recall@k + latency against the exact ground truth |

Datasets: SIFT1M / GIST1M (TEXMEX), ANN-Benchmarks sets (GloVe, NYTimes-256,
Fashion-MNIST-784) and synthetic vectors up to ~10M×768. All binaries generate
synthetic data inline today (`--seed`); loading the `.fvecs`/`.bvecs` benchmark
files is a small extension on top of the same interface.

## The comparison (brute force vs ANN)

`bin/knn` is the brute-force side: a fused exact kernel that returns every true
neighbour (recall = 1 by construction, and `--verify` proves it). The
approximate side is FAISS on the *same* vectors:

```
./bin/gendata --n 1000000 --d 128 --q 1000 --k 10 --out data   # export + ground truth
python3 bench/faiss_compare.py --data data --k 10 \
        --nprobe 1,4,16,64 --efsearch 16,64,256 --json data/faiss.json
# or, in one step:  make compare
```

`bin/gendata` writes `db.f32`, `queries.f32` and the double-precision
`gt_idx.i32`/`gt_dist.f32`, so FAISS and our kernel see byte-identical data and
recall@k is measured against the same oracle. The harness sweeps `nprobe`
(IVF-PQ) and `efSearch` (HNSW) and reports recall@k vs latency; the frontend
draws the recall-vs-latency curve.

## Quick start on a free Colab GPU

```bash
# 1. Colab: New notebook -> Runtime -> Change runtime type -> T4 GPU
# 2. one cell -- paste remote/colab_bootstrap.py, or this loader:
#      import urllib.request as u; exec(u.urlopen(
#        'https://raw.githubusercontent.com/JaiiMukundh/fused-knn-search/main/remote/colab_bootstrap.py').read())
# 3. it prints, within the first minute, the ssh tunnel + URLs:
#      ssh tunnel     -> ssh -p <port> root@bore.pub -i ~/.ssh/colab_pcap
#                        (terminal, rsync, VS Code Remote-SSH)
#      web frontend   -> search / sweeps / brute-force-vs-ANN in the browser
#      shared terminal-> sshx link for your teammates
```

The bootstrap also installs FAISS (`faiss-gpu-cu12`, falling back to
`faiss-cpu`), runs one brute-force-vs-ANN comparison and prints the table.

Full walkthrough, sharing and troubleshooting: **[`remote/COLAB_SETUP.md`](remote/COLAB_SETUP.md)**.

Locally (no GPU needed for the CPU side):

```bash
make check-cpu     # builds knn_omp + gendata, runs verified checks
```

## Web frontend (`frontend/`)

A single-page UI + JSON API served from the GPU box:

- **Search** — one exact kNN run: fused kernel ms, throughput (GB/s, GFLOP/s),
  two-pass ms + speedup, PASS/FAIL exactness check, and the top-k table
  (indices + distances) for the first queries (`--dump`).
- **Sweep** — fused vs two-pass across database sizes, chart + table.
- **Compare** — brute force (`bin/knn`) vs ANN (FAISS): `nprobe`/`efSearch`
  sweeps, recall@k per configuration, ms/query, speedup vs exact, and the
  recall-vs-latency curve (Objective 3).
- **Build** — `make gpu` / `make cpu` on the box, straight from the browser.
- Engine switch: GPU (`knn --device cuda`) or CPU (`knn --device cpu`); every
  number comes from executing the compiled binaries and parsing their stdout.
- Availability is *probed*, not guessed: the CUDA engine is declared ready only
  after it has actually executed a tiny search, so a present-but-unusable
  binary reports `cuda unavailable: <reason>` instead of an empty results grid.

It runs anywhere Python + Flask exist (`python3 frontend/server.py --port 8080`);
the Colab bootstrap serves it through a Cloudflare quick tunnel.

## Builds and checks

| Command | What it does |
|---|---|
| `make cpu` | builds `bin/knn_omp` (OpenMP baseline) + `bin/gendata` |
| `make gpu` | builds `bin/knn` — MPI + OpenMP + CUDA in one binary (needs `nvcc` + `mpicxx`) |
| `make compare` | exports the dataset, then brute force vs FAISS (ANN) |
| `make check-cpu` | builds the CPU side, runs verified checks locally |
| `make check-gpu` | builds `bin/knn`, runs verified checks (`-np 1` and `-np 2`) on the GPU box |
| `make check` | both, skipping whatever this machine can't build |
| `make legacy` | the superseded `knn_cuda` / `knn_mpi_*` binaries |

`make gpu` targets the compute capability of the GPU present at build time
(T4 → sm_75, P100 → sm_60) and embeds matching SASS plus PTX (so a binary
from one session still runs on a *newer* GPU model later).

## Binaries

`bin/knn` is one program: MPI for sharding, OpenMP for the host-side/CPU path,
CUDA for the device kernels. `--device` selects the search path; MPI is
optional at compile time (`-DPCAP_USE_MPI`) and at run time (`-np 1` behaves
exactly like a single-process run).

```
--n N          database size            (default 1048576)
--d D          dimensionality           (default 128)
--q Q          number of queries        (default 32)
--k K          neighbours               (default 10; CUDA supports 1,4,8,10,16,32)
--seed S       dataset seed             (default 42)
--device P     cuda | cpu | auto        (default auto)
--threads T    OpenMP threads           (0 = all, default 0)
--verify V     queries checked vs CPU reference (default 4, 0 = off)
--twopass 0|1  also time the two-pass variant (CUDA path, default 1)
--dump M       print top-k of the first M queries as [topk] lines (default 0)
```

```bash
./bin/knn --device cpu --threads 8          # OpenMP CPU baseline
./bin/knn --device cuda --twopass 1          # single GPU: fused vs two-pass
mpirun -np 4 ./bin/knn --device cuda         # multi-GPU: rank r uses device r % ndev
./bin/gendata --n 1000000 --d 128 --q 1000 --out data   # export for FAISS
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
src/knn.cu            the unified engine: MPI sharding + OpenMP + CUDA kernels
src/gendata.cpp       exports db/queries/ground-truth for the FAISS comparison
src/knn_omp.cpp       standalone OpenMP CPU baseline (laptop fallback, no CUDA)
src/knn_cuda.cu       superseded single-GPU driver (kept for reference)
src/knn_mpi_cuda.cu   superseded MPI+CUDA driver (kept for reference)
src/knn_mpi_omp.cpp   superseded MPI+OpenMP baseline (kept for reference)
bench/faiss_compare.py  brute force vs FAISS: IVF-PQ + HNSW, recall@k, latency, JSON
frontend/server.py    Flask API driving the binaries
                      (/api/search, /benchmark, /compare, /build, /state)
frontend/index.html   single-page UI (search, sweeps, compare, build, how-it-works)
remote/colab_bootstrap.py   Colab cell: ssh (bore) + Cloudflare tunnels + clone + build + FAISS
remote/COLAB_SETUP.md       guide for the Colab route
Makefile              build targets (cpu / gpu / compare / check*)
```

## Notes

- Timing methodology: warm-up iterations are excluded; `bin/knn --device cuda`
  reports kernel time plus end-to-end and compares fused vs two-pass on
  identical inputs. The CPU path reports the same metrics for the same
  `--n/--d/--q/--k`.
- The MPI sharding is exact: each rank searches its share of the database and
  the merge scans all ranks' candidates; results equal the single-GPU result
  (checked via `--verify`, `-np 1` vs `-np 2`).
- The ANN side is approximate by design: IVF-PQ and HNSW return most of the
  true neighbours, never all of them, which is what the recall@k column
  quantifies against the exact oracle. The exact side is our kernel, whose
  recall is 1.000 because it evaluates every vector.
- Free-GPU realities: Colab runtimes recycle (keep the tab open; ~12 h max,
  ~90 min idle). Keep sweeps lean and log results to `out/`.
- Local compile check (no GPU needed): with a CUDA toolchain on the laptop
  (conda `cuda-nvcc` + `cuda-cudart-dev` work), `make gpu NVCC=<env>/bin/nvcc
  CUDA_HOME=<env> CUDA_ARCH_FLAGS='-arch=sm_75'` compiles and links both CUDA
  binaries — catches API breakage (e.g. CUDA 13 removing
  `cudaDeviceProp::memoryClockRate`) before burning GPU quota.
