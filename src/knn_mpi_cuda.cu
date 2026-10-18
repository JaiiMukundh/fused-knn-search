// knn_mpi_cuda.cu -- multi-GPU exact kNN: MPI rank per GPU + fused CUDA kernel.
//
//   mpirun -np 2 ./knn_mpi_cuda --n 1000000 --d 128 --q 32 --k 10
//
// Sharding and merging are identical to knn_mpi_omp (see that file for the
// scheme); the difference is that each rank's local search runs on a GPU with
// fused_topk_kernel from fused_topk.cuh. Run it with -np 2 on a box with two
// visible GPUs; each rank calls cudaSetDevice(rank % ndevices),
// so ranks land on different GPUs when more than one is visible.
//
// The communication volume stays O(size * q * k) per rank, so this is the
// strong/weak-scaling vehicle for the report.

#include <cuda_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "common.hpp"
#include "fused_topk.cuh"

using namespace pcap;

static constexpr int kDevNameLen = 256;  // matches cudaDeviceProp::name

static void ck(cudaError_t e, int rank, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "rank %d: CUDA error at %s: %s\n", rank, what, cudaGetErrorString(e));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const Args a = parse_args(argc, argv);

    if (!cuda_k_supported(a.k)) {
        if (rank == 0) std::fprintf(stderr, "CUDA build supports --k in {1,4,8,10,16,32}\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    if (a.n > (size_t)INT_MAX || a.q * (size_t)a.k > (size_t)INT_MAX) {
        if (rank == 0)
            std::fprintf(stderr, "--n and q*k must be <= INT_MAX for the CUDA kernels\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    int ndev = 0;
    const cudaError_t dev_err = cudaGetDeviceCount(&ndev);
    if (dev_err != cudaSuccess || ndev < 1) {
        std::fprintf(stderr, "rank %d: no CUDA device (%s)\n", rank, cudaGetErrorString(dev_err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int mydev = rank % ndev;
    if (cudaSetDevice(mydev) != cudaSuccess) {
        std::fprintf(stderr, "rank %d: cudaSetDevice(%d) failed\n", rank, mydev);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaDeviceProp myprop{};
    ck(cudaGetDeviceProperties(&myprop, mydev), rank, "cudaGetDeviceProperties");

    // ---- local shard --------------------------------------------------------
    const size_t r0 = (size_t)rank * a.n / (size_t)size;
    const size_t r1 = (size_t)(rank + 1) * a.n / (size_t)size;
    const size_t ln = r1 - r0;
    const size_t ln_alloc = ln > 0 ? ln : 1;

    std::vector<float> h_shard(ln_alloc * (size_t)a.d);
    for (size_t j = 0; j < ln; ++j)
        for (int t = 0; t < a.d; ++t) h_shard[j * a.d + t] = db_value(r0 + j, t, a.seed);
    std::vector<float> qs(a.q * (size_t)a.d);
    for (size_t r = 0; r < a.q; ++r)
        for (int t = 0; t < a.d; ++t) qs[r * a.d + t] = query_value(r, t, a.seed);

    // ---- device buffers -----------------------------------------------------
    float *d_db = nullptr, *d_q = nullptr, *d_outd = nullptr;
    int* d_outi = nullptr;
    ck(cudaMalloc(&d_db, ln_alloc * (size_t)a.d * sizeof(float)), rank, "cudaMalloc db");
    ck(cudaMalloc(&d_q, a.q * (size_t)a.d * sizeof(float)), rank, "cudaMalloc queries");
    ck(cudaMalloc(&d_outd, a.q * (size_t)a.k * sizeof(float)), rank, "cudaMalloc outd");
    ck(cudaMalloc(&d_outi, a.q * (size_t)a.k * sizeof(int)), rank, "cudaMalloc outi");

    cudaEvent_t ev0, ev1;
    ck(cudaEventCreate(&ev0), rank, "event");
    ck(cudaEventCreate(&ev1), rank, "event");

    ck(cudaEventRecord(ev0), rank, "record");
    ck(cudaMemcpy(d_db, h_shard.data(), ln_alloc * (size_t)a.d * sizeof(float),
                  cudaMemcpyHostToDevice), rank, "H2D db");
    ck(cudaMemcpy(d_q, qs.data(), a.q * (size_t)a.d * sizeof(float), cudaMemcpyHostToDevice),
       rank, "H2D queries");
    ck(cudaEventRecord(ev1), rank, "record");
    ck(cudaEventSynchronize(ev1), rank, "sync");
    float h2d_ms = 0.f;
    ck(cudaEventElapsedTime(&h2d_ms, ev0, ev1), rank, "elapsed");

    // ---- fused search on this rank's shard ----------------------------------
    std::string err;
    if (ln > 0 && !launch_fused(d_db, d_q, (int)ln, a.d, (int)a.q, a.k, d_outd, d_outi, err)) {
        std::fprintf(stderr, "rank %d: fused launch failed: %s\n", rank, err.c_str());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    ck(cudaDeviceSynchronize(), rank, "warmup sync");

    MPI_Barrier(MPI_COMM_WORLD);
    ck(cudaEventRecord(ev0), rank, "record");
    if (ln > 0 && !launch_fused(d_db, d_q, (int)ln, a.d, (int)a.q, a.k, d_outd, d_outi, err)) {
        std::fprintf(stderr, "rank %d: fused launch failed: %s\n", rank, err.c_str());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    ck(cudaEventRecord(ev1), rank, "record");
    ck(cudaEventSynchronize(ev1), rank, "sync");
    float kernel_ms = 0.f;
    ck(cudaEventElapsedTime(&kernel_ms, ev0, ev1), rank, "elapsed");

    const int per_rank = (int)(a.q * (size_t)a.k);
    std::vector<float> l_outd((size_t)per_rank, FLT_MAX);
    std::vector<int> l_outi((size_t)per_rank, -1);
    if (ln > 0) {  // empty shard keeps the FLT_MAX/-1 sentinels: never merge stale device memory
        ck(cudaMemcpy(l_outd.data(), d_outd, (size_t)per_rank * sizeof(float),
                      cudaMemcpyDeviceToHost), rank, "D2H outd");
        ck(cudaMemcpy(l_outi.data(), d_outi, (size_t)per_rank * sizeof(int),
                      cudaMemcpyDeviceToHost), rank, "D2H outi");
    }

    // ---- gather local results and stats to rank 0 ---------------------------
    double t_stats[2] = {kernel_ms, h2d_ms};  // ms
    std::vector<double> stats;
    std::vector<int> devs;
    if (rank == 0) {
        stats.resize((size_t)size * 2);
        devs.resize((size_t)size);
    }
    MPI_Gather(t_stats, 2, MPI_DOUBLE, rank == 0 ? stats.data() : nullptr, 2, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);
    MPI_Gather(&mydev, 1, MPI_INT, rank == 0 ? devs.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<char> devnames;  // each rank reports the name of the GPU it used
    if (rank == 0) devnames.resize((size_t)size * kDevNameLen);
    MPI_Gather(myprop.name, kDevNameLen, MPI_CHAR, rank == 0 ? devnames.data() : nullptr,
               kDevNameLen, MPI_CHAR, 0, MPI_COMM_WORLD);

    std::vector<float> gd;
    std::vector<int> gi;
    if (rank == 0) {
        gd.resize((size_t)size * per_rank);
        gi.resize((size_t)size * per_rank);
    }
    MPI_Gather(l_outd.data(), per_rank, MPI_FLOAT, rank == 0 ? gd.data() : nullptr, per_rank,
               MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(l_outi.data(), per_rank, MPI_INT, rank == 0 ? gi.data() : nullptr, per_rank,
               MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        print_run_header("knn_mpi_cuda", a);
        std::printf("[mpi ] ranks=%d  all ranks use the fused kernel\n", size);
        for (int rr = 0; rr < size; ++rr) {
            const double kms = stats[(size_t)rr * 2 + 0];
            const double hms = stats[(size_t)rr * 2 + 1];
            const size_t rr0 = (size_t)rr * a.n / (size_t)size;
            const size_t rr1 = (size_t)(rr + 1) * a.n / (size_t)size;
            const double bytes = (double)a.q * (double)(rr1 - rr0) * a.d * sizeof(float);
            std::printf("[rank] %2d  gpu%d %-18s  kernel %7.3f ms  %7.1f GB/s  (H2D %6.2f ms)\n",
                        rr, devs[(size_t)rr], &devnames[(size_t)rr * kDevNameLen], kms,
                        gbytes_per_s(bytes, kms / 1e3), hms);
        }
        if (size > ndev)
            std::printf("[warn] %d ranks share %d GPU(s); use -np <= %d for real multi-GPU scaling\n",
                        size, ndev, ndev);

        // ---- merge: only size*k candidates per query ------------------------
        const double m0 = wall_time();
        std::vector<float> outd(a.q * (size_t)a.k);
        std::vector<int> outi(a.q * (size_t)a.k);
        for (size_t qq = 0; qq < a.q; ++qq) {
            std::vector<float> bd((size_t)a.k, FLT_MAX);
            std::vector<int> bi((size_t)a.k, -1);
            for (int rr = 0; rr < size; ++rr) {
                const size_t rr0 = (size_t)rr * a.n / (size_t)size;  // rank rr's first global row
                const float* drow = &gd[((size_t)rr * a.q + qq) * a.k];
                const int* irow = &gi[((size_t)rr * a.q + qq) * a.k];
                for (int t = 0; t < a.k; ++t) {
                    const int lidx = irow[t];  // shard-local: translate to a global row
                    const int gidx = lidx < 0 ? -1 : (int)(rr0 + (size_t)lidx);
                    insert_topk_host(drow[t], gidx, bd.data(), bi.data(), a.k);
                }
            }
            for (int t = 0; t < a.k; ++t) {
                outd[qq * a.k + t] = bd[(size_t)t];
                outi[qq * a.k + t] = bi[(size_t)t];
            }
        }
        const double merge_ms = (wall_time() - m0) * 1e3;
        std::printf("[time] merge (rank0, %d x %zu candidates) %.3f ms\n", size, a.q * (size_t)a.k,
                    merge_ms);

        if (a.verify > 0)
            print_verify(verify_topk(outd.data(), outi.data(), qs.data(), a.n, a.d, a.k, a.seed,
                                     std::min<int>(a.verify, (int)a.q)));
    }

    ck(cudaFree(d_db), rank, "free");
    ck(cudaFree(d_q), rank, "free");
    ck(cudaFree(d_outd), rank, "free");
    ck(cudaFree(d_outi), rank, "free");
    ck(cudaEventDestroy(ev0), rank, "destroy");
    ck(cudaEventDestroy(ev1), rank, "destroy");
    MPI_Finalize();
    return 0;
}
