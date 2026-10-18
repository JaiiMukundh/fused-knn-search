// knn.cu -- the unified exact kNN engine: MPI + OpenMP + CUDA in one program.
//
//   mpirun -np 1 ./bin/knn --n 1000000 --d 128 --q 32 --k 10
//   mpirun -np 4 ./bin/knn --n 1000000 --d 128 --q 32 --k 10     # one shard per rank
//   ./bin/knn --device cpu --threads 8 --n 200000 --d 128 --q 8 --k 10
//
// This replaces the previous split of knn_omp / knn_mpi_omp / knn_cuda /
// knn_mpi_cuda: sharding, the CPU (OpenMP) search and the GPU (CUDA) kernels
// now live in ONE binary with one argument parser and one output format.
//
//   * MPI  -- rank r owns rows [r*n/size, (r+1)*n/size) of the database and
//             searches them; only q*k candidates per rank are communicated, so
//             the merge volume is O(size*q*k), independent of n.
//   * OpenMP -- the host-side work: shard generation, and the whole CPU search
//             path when --device cpu (one thread per query).
//   * CUDA -- the fused distance+top-k kernel (and the two-pass variant for the
//             fusion comparison) when --device cuda.
//
// Output lines are the ones frontend/server.py parses:
//   [run] / [gpu ] / [H2D ] / [time] fused|twopass|cpu|merge / [rank] / [verify] / [topk]
//
// With -np 1 the behaviour matches the old knn_cuda (cuda) or knn_omp (cpu), so
// the single-GPU numbers and the laptop baseline are unchanged.

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef PCAP_USE_MPI
#include <mpi.h>
#endif

#include <cuda_runtime.h>

#include "common.hpp"
#include "fused_topk.cuh"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace pcap;

// ---------------------------------------------------------------------------
// MPI may be compiled out so the same source also builds on a box without it
// (that binary then behaves exactly like -np 1). Everything below goes through
// these few wrappers instead of calling MPI_* directly.
// ---------------------------------------------------------------------------
namespace mpi {

int g_rank = 0;
int g_size = 1;

inline void init(int* argc, char*** argv) {
#ifdef PCAP_USE_MPI
    MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
#else
    (void)argc;
    (void)argv;
#endif
}

inline int rank() { return g_rank; }
inline int size() { return g_size; }

inline void barrier() {
#ifdef PCAP_USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
}

// Gather `count` doubles from every rank into rank 0's out (null elsewhere).
inline void gather_d(const double* in, int count, double* out) {
#ifdef PCAP_USE_MPI
    MPI_Gather(const_cast<double*>(in), count, MPI_DOUBLE, out, count, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);
#else
    if (out) for (int i = 0; i < count; ++i) out[i] = in[i];
#endif
}

inline void gather_i(const int* in, int count, int* out) {
#ifdef PCAP_USE_MPI
    MPI_Gather(const_cast<int*>(in), count, MPI_INT, out, count, MPI_INT, 0, MPI_COMM_WORLD);
#else
    if (out) for (int i = 0; i < count; ++i) out[i] = in[i];
#endif
}

inline void gather_f(const float* in, int count, float* out) {
#ifdef PCAP_USE_MPI
    MPI_Gather(const_cast<float*>(in), count, MPI_FLOAT, out, count, MPI_FLOAT, 0, MPI_COMM_WORLD);
#else
    if (out) for (int i = 0; i < count; ++i) out[i] = in[i];
#endif
}

inline void gather_chars(const char* in, int count, char* out) {
#ifdef PCAP_USE_MPI
    MPI_Gather(const_cast<char*>(in), count, MPI_CHAR, out, count, MPI_CHAR, 0, MPI_COMM_WORLD);
#else
    if (out) for (int i = 0; i < count; ++i) out[i] = in[i];
#endif
}

inline void finalize() {
#ifdef PCAP_USE_MPI
    MPI_Finalize();
#endif
}

[[noreturn]] inline void abort(int code) {
#ifdef PCAP_USE_MPI
    MPI_Abort(MPI_COMM_WORLD, code);
#endif
    std::exit(code);
}

}  // namespace mpi

namespace {

constexpr int kDevNameLen = 256;  // matches cudaDeviceProp::name

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "rank %d: CUDA error at %s: %s\n", mpi::rank(), what,
                     cudaGetErrorString(e));
        mpi::abort(1);
    }
}

void set_threads(int threads) {
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#else
    (void)threads;
#endif
}

int max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// ---------------------------------------------------------------------------
// The two search paths. Both fill l_outd / l_outi (q*k each) with *global* row
// indices, so rank 0's merge does not have to know which path produced them.
// ---------------------------------------------------------------------------

// CUDA path: fused kernel (+ optional two-pass) on this rank's shard.
// Returns {kernel_ms, h2d_ms, twopass_ms or -1}.
struct GpuTimes {
    float kernel_ms = 0.f;
    float h2d_ms = 0.f;
    float twopass_ms = -1.f;
    std::string device_name;
};

GpuTimes search_gpu(const std::vector<float>& h_shard, size_t ln_alloc, size_t ln,
                    const std::vector<float>& qs, const Args& a, size_t r0, bool twopass,
                    std::vector<float>& l_outd, std::vector<int>& l_outi) {
    GpuTimes t;
    const int mydev = mpi::rank();
    int ndev = 0;
    const cudaError_t dev_err = cudaGetDeviceCount(&ndev);
    if (dev_err != cudaSuccess || ndev < 1) {
        std::fprintf(stderr, "rank %d: no CUDA device (%s)\n", mpi::rank(),
                     cudaGetErrorString(dev_err));
        mpi::abort(1);
    }
    const int use_dev = mydev % ndev;
    if (cudaSetDevice(use_dev) != cudaSuccess) {
        std::fprintf(stderr, "rank %d: cudaSetDevice(%d) failed\n", mpi::rank(), use_dev);
        mpi::abort(1);
    }
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, use_dev), "cudaGetDeviceProperties");
    t.device_name = prop.name;

    float *d_db = nullptr, *d_q = nullptr, *d_outd = nullptr;
    int* d_outi = nullptr;
    ck(cudaMalloc(&d_db, ln_alloc * (size_t)a.d * sizeof(float)), "cudaMalloc db");
    ck(cudaMalloc(&d_q, a.q * (size_t)a.d * sizeof(float)), "cudaMalloc queries");
    ck(cudaMalloc(&d_outd, a.q * (size_t)a.k * sizeof(float)), "cudaMalloc outd");
    ck(cudaMalloc(&d_outi, a.q * (size_t)a.k * sizeof(int)), "cudaMalloc outi");

    cudaEvent_t ev0, ev1;
    ck(cudaEventCreate(&ev0), "event");
    ck(cudaEventCreate(&ev1), "event");

    ck(cudaEventRecord(ev0), "record");
    ck(cudaMemcpy(d_db, h_shard.data(), ln_alloc * (size_t)a.d * sizeof(float),
                  cudaMemcpyHostToDevice), "H2D db");
    ck(cudaMemcpy(d_q, qs.data(), a.q * (size_t)a.d * sizeof(float), cudaMemcpyHostToDevice),
       "H2D queries");
    ck(cudaEventRecord(ev1), "record");
    ck(cudaEventSynchronize(ev1), "sync");
    ck(cudaEventElapsedTime(&t.h2d_ms, ev0, ev1), "elapsed");

    std::string err;
    if (ln > 0 && !launch_fused(d_db, d_q, (int)ln, a.d, (int)a.q, a.k, d_outd, d_outi, err)) {
        std::fprintf(stderr, "rank %d: fused launch failed: %s\n", mpi::rank(), err.c_str());
        mpi::abort(1);
    }
    ck(cudaDeviceSynchronize(), "warmup sync");

    mpi::barrier();
    ck(cudaEventRecord(ev0), "record");
    if (ln > 0 && !launch_fused(d_db, d_q, (int)ln, a.d, (int)a.q, a.k, d_outd, d_outi, err)) {
        std::fprintf(stderr, "rank %d: fused launch failed: %s\n", mpi::rank(), err.c_str());
        mpi::abort(1);
    }
    ck(cudaEventRecord(ev1), "record");
    ck(cudaEventSynchronize(ev1), "sync");
    ck(cudaEventElapsedTime(&t.kernel_ms, ev0, ev1), "elapsed");

    if (twopass && ln > 0) {
        const size_t rows_bytes = (size_t)a.q * ln * sizeof(float);
        size_t free_mem = 0, total_mem = 0;
        ck(cudaMemGetInfo(&free_mem, &total_mem), "cudaMemGetInfo");
        if (rows_bytes > free_mem / 2) {
            if (mpi::rank() == 0)
                std::printf("[skip] two-pass needs %.1f GB for the distance array, "
                            "only %.1f GB free\n", rows_bytes / 1e9, free_mem / 1e9);
        } else {
            float* d_rows = nullptr;
            float* d_outd2 = nullptr;
            int* d_outi2 = nullptr;
            ck(cudaMalloc(&d_rows, rows_bytes), "cudaMalloc rows");
            ck(cudaMalloc(&d_outd2, a.q * (size_t)a.k * sizeof(float)), "cudaMalloc outd2");
            ck(cudaMalloc(&d_outi2, a.q * (size_t)a.k * sizeof(int)), "cudaMalloc outi2");

            const unsigned chunks = (unsigned)std::min<size_t>((ln + 255) / 256, 4096);
            const dim3 dgrid((unsigned)a.q, chunks);

            ck(cudaEventRecord(ev0), "record");
            distances_kernel<<<dgrid, 256>>>(d_db, d_q, (int)ln, a.d, d_rows);
            ck(cudaGetLastError(), "distances_kernel");
            ck(cudaEventRecord(ev1), "record");
            ck(cudaEventSynchronize(ev1), "sync");
            float dist_ms = 0.f;
            ck(cudaEventElapsedTime(&dist_ms, ev0, ev1), "elapsed");

            if (!launch_select(d_rows, (int)ln, (int)a.q, a.k, d_outd2, d_outi2, err)) {
                std::fprintf(stderr, "rank %d: select launch failed: %s\n", mpi::rank(),
                             err.c_str());
                mpi::abort(1);
            }
            ck(cudaEventRecord(ev0), "record");
            if (!launch_select(d_rows, (int)ln, (int)a.q, a.k, d_outd2, d_outi2, err)) {
                std::fprintf(stderr, "rank %d: select launch failed: %s\n", mpi::rank(),
                             err.c_str());
                mpi::abort(1);
            }
            ck(cudaEventRecord(ev1), "record");
            ck(cudaEventSynchronize(ev1), "sync");
            float select_ms = 0.f;
            ck(cudaEventElapsedTime(&select_ms, ev0, ev1), "elapsed");

            t.twopass_ms = dist_ms + select_ms;
            if (mpi::rank() == 0 && mpi::size() == 1) {
                const float two = t.twopass_ms;
                std::printf("[time] twopass %7.3f ms  (dist %.3f + select %.3f)  "
                            "writes %.1f MB of distance array; fused is %.2fx faster\n",
                            two, dist_ms, select_ms, rows_bytes / 1e6,
                            two > 0.f ? two / t.kernel_ms : 0.0);
                std::printf("[time] twopass extra traffic vs fused: %.1f MB written + "
                            "%.1f MB re-read\n", rows_bytes / 1e6, rows_bytes / 1e6);
            }
            ck(cudaFree(d_rows), "free");
            ck(cudaFree(d_outd2), "free");
            ck(cudaFree(d_outi2), "free");
        }
    }

    const int per_rank = (int)(a.q * (size_t)a.k);
    if (ln > 0) {
        ck(cudaMemcpy(l_outd.data(), d_outd, (size_t)per_rank * sizeof(float),
                      cudaMemcpyDeviceToHost), "D2H outd");
        ck(cudaMemcpy(l_outi.data(), d_outi, (size_t)per_rank * sizeof(int),
                      cudaMemcpyDeviceToHost), "D2H outi");
    }
    // Translate shard-local indices to global rows now, so the merge is uniform.
    for (size_t i = 0; i < (size_t)per_rank; ++i)
        if (l_outi[i] >= 0) l_outi[i] = (int)(r0 + (size_t)l_outi[i]);

    ck(cudaFree(d_db), "free");
    ck(cudaFree(d_q), "free");
    ck(cudaFree(d_outd), "free");
    ck(cudaFree(d_outi), "free");
    ck(cudaEventDestroy(ev0), "destroy");
    ck(cudaEventDestroy(ev1), "destroy");
    return t;
}

// OpenMP path: this rank's shard is scanned on the host, one thread per query.
double search_cpu(const std::vector<float>& h_shard, size_t ln, const std::vector<float>& qs,
                  const Args& a, size_t r0, std::vector<float>& l_outd,
                  std::vector<int>& l_outi) {
    const double t0 = wall_time();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (long long qq = 0; qq < (long long)a.q; ++qq) {
        float* bd = &l_outd[(size_t)qq * a.k];
        int* bi = &l_outi[(size_t)qq * a.k];
        for (int t = 0; t < a.k; ++t) {
            bd[t] = FLT_MAX;
            bi[t] = -1;
        }
        const float* qv = &qs[(size_t)qq * a.d];
        for (size_t j = 0; j < ln; ++j) {
            const float* v = &h_shard[j * a.d];
            float acc = 0.0f;
            for (int t = 0; t < a.d; ++t) {
                const float diff = qv[t] - v[t];
                acc = fmaf(diff, diff, acc);
            }
            insert_topk_host(acc, (int)(r0 + j), bd, bi, a.k);  // global index
        }
    }
    return wall_time() - t0;
}

}  // namespace

int main(int argc, char** argv) {
    mpi::init(&argc, &argv);
    const int rank = mpi::rank();
    const int size = mpi::size();

    // --device is ours; common.hpp's parser must only see the shared flags.
    std::string device = "auto";
    std::vector<char*> rest;
    rest.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string f = argv[i];
        if (f == "--device") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--device needs cuda|cpu|auto\n");
                mpi::abort(2);
            }
            device = argv[++i];
        } else {
            rest.push_back(argv[i]);
        }
    }
    const Args a = parse_args((int)rest.size(), rest.data());
    set_threads(a.threads);

    if (device != "auto" && device != "cuda" && device != "cpu") {
        std::fprintf(stderr, "--device must be cuda, cpu or auto\n");
        mpi::abort(2);
    }
    bool want_gpu = (device == "cuda");
    if (device == "auto") {
        int ndev = 0;
        want_gpu = (cudaGetDeviceCount(&ndev) == cudaSuccess && ndev >= 1);
    }
    if (want_gpu && !cuda_k_supported(a.k)) {
        if (rank == 0)
            std::fprintf(stderr, "CUDA path supports --k in {1,4,8,10,16,32}; "
                                 "use --device cpu for other k\n");
        mpi::abort(2);
    }
    if (want_gpu && (a.n > (size_t)INT_MAX || a.q * (size_t)a.k > (size_t)INT_MAX)) {
        if (rank == 0)
            std::fprintf(stderr, "--n and q*k must be <= INT_MAX for the CUDA kernels\n");
        mpi::abort(2);
    }

    // ---- this rank's shard of the one shared database ------------------------
    const size_t r0 = (size_t)rank * a.n / (size_t)size;
    const size_t r1 = (size_t)(rank + 1) * a.n / (size_t)size;
    const size_t ln = r1 - r0;
    const size_t ln_alloc = ln > 0 ? ln : 1;

    std::vector<float> h_shard(ln_alloc * (size_t)a.d);
    std::vector<float> qs(a.q * (size_t)a.d);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long j = 0; j < (long long)ln; ++j)
        for (int t = 0; t < a.d; ++t) h_shard[(size_t)j * a.d + t] = db_value(r0 + (size_t)j, t, a.seed);
    for (size_t qq = 0; qq < a.q; ++qq)
        for (int t = 0; t < a.d; ++t) qs[qq * a.d + t] = query_value(qq, t, a.seed);

    const int per_rank = (int)(a.q * (size_t)a.k);
    std::vector<float> l_outd((size_t)per_rank, FLT_MAX);
    std::vector<int> l_outi((size_t)per_rank, -1);

    // ---- search --------------------------------------------------------------
    GpuTimes gt;
    double cpu_s = 0.0;
    if (want_gpu) {
        gt = search_gpu(h_shard, ln_alloc, ln, qs, a, r0, a.twopass != 0, l_outd, l_outi);
    } else {
        cpu_s = search_cpu(h_shard, ln, qs, a, r0, l_outd, l_outi);
    }

    // ---- gather stats + candidates, then merge on rank 0 ---------------------
    double t_stats[3] = {gt.kernel_ms, gt.h2d_ms, cpu_s};
    std::vector<double> stats;
    std::vector<int> devs;
    std::vector<char> devnames;
    if (rank == 0) {
        stats.resize((size_t)size * 3);
        devs.resize((size_t)size);
        devnames.resize((size_t)size * kDevNameLen);
    }
    mpi::gather_d(t_stats, 3, rank == 0 ? stats.data() : nullptr);

    int ndev = 0;
    if (want_gpu) cudaGetDeviceCount(&ndev);
    const int mydev = (want_gpu && ndev > 0) ? (rank % ndev) : -1;
    mpi::gather_i(&mydev, 1, rank == 0 ? devs.data() : nullptr);
    mpi::gather_chars(gt.device_name.c_str(), kDevNameLen, rank == 0 ? devnames.data() : nullptr);

    std::vector<float> gd;
    std::vector<int> gi;
    if (rank == 0) {
        gd.resize((size_t)size * per_rank);
        gi.resize((size_t)size * per_rank);
    }
    mpi::gather_f(l_outd.data(), per_rank, rank == 0 ? gd.data() : nullptr);
    mpi::gather_i(l_outi.data(), per_rank, rank == 0 ? gi.data() : nullptr);

    if (rank == 0) {
        print_run_header("knn", a);
        if (size > 1) {
            std::printf("[mpi ] ranks=%d  shard=%zu rows/rank  device=%s  threads=%d\n", size,
                        ln, want_gpu ? "cuda" : "cpu", max_threads());
        }
        if (want_gpu) {
            std::printf("[gpu ] %s\n", gt.device_name.c_str());
            if (size == 1) {
                const double read_bytes = (double)a.q * (double)a.n * a.d * sizeof(float);
                std::printf("[H2D ] upload %.1f MB in %.2f ms (%.1f GB/s)\n",
                            (double)ln * a.d * sizeof(float) / 1e6, gt.h2d_ms,
                            gbytes_per_s((double)ln * a.d * sizeof(float), gt.h2d_ms / 1e3));
                std::printf("[time] fused   %8.3f ms  (%.3f ms/query)  %7.1f GB/s db reads  "
                            "%7.1f GFLOP/s\n",
                            gt.kernel_ms, gt.kernel_ms / (double)a.q,
                            gbytes_per_s(read_bytes, gt.kernel_ms / 1e3),
                            gflop_per_s((double)a.q * (double)a.n * a.d * 2.0, 1.0,
                                        gt.kernel_ms / 1e3));
            }
        } else {
            const double read_bytes = (double)a.q * (double)a.n * a.d * sizeof(float);
            std::printf("[time] cpu     %8.3f s   (%.3f ms/query)  %.3g pairs/s  %.1f GB/s db reads  "
                        "%.1f GFLOP/s\n",
                        cpu_s, cpu_s * 1e3 / (double)a.q,
                        (double)a.q * (double)a.n / (cpu_s > 0 ? cpu_s : 1.0),
                        gbytes_per_s(read_bytes, cpu_s),
                        gflop_per_s((double)a.q * (double)a.n * a.d * 2.0, 1.0, cpu_s));
        }

        if (size > 1) {
            for (int rr = 0; rr < size; ++rr) {
                const double kms = stats[(size_t)rr * 3 + 0];
                const double hms = stats[(size_t)rr * 3 + 1];
                const double cms = stats[(size_t)rr * 3 + 2];
                const size_t rr0 = (size_t)rr * a.n / (size_t)size;
                const size_t rr1 = (size_t)(rr + 1) * a.n / (size_t)size;
                const double bytes = (double)a.q * (double)(rr1 - rr0) * a.d * sizeof(float);
                if (want_gpu)
                    std::printf("[rank] %2d  gpu%d %-18s  kernel %7.3f ms  %7.1f GB/s  (H2D %6.2f ms)\n",
                                rr, devs[(size_t)rr], &devnames[(size_t)rr * kDevNameLen], kms,
                                gbytes_per_s(bytes, kms / 1e3), hms);
                else
                    std::printf("[rank] %2d  cpu  %-18s  shard %7.3f s  %7.1f GB/s  (%zu rows)\n",
                                rr, "", cms, gbytes_per_s(bytes, cms), rr1 - rr0);
            }
        }

        // ---- merge: size*k candidates per query, O(size*q*k) -----------------
        const double m0 = wall_time();
        std::vector<float> outd(a.q * (size_t)a.k);
        std::vector<int> outi(a.q * (size_t)a.k);
        for (size_t qq = 0; qq < a.q; ++qq) {
            std::vector<float> bd((size_t)a.k, FLT_MAX);
            std::vector<int> bi((size_t)a.k, -1);
            for (int rr = 0; rr < size; ++rr) {
                const float* drow = &gd[((size_t)rr * a.q + qq) * a.k];
                const int* irow = &gi[((size_t)rr * a.q + qq) * a.k];
                for (int t = 0; t < a.k; ++t)
                    insert_topk_host(drow[t], irow[t], bd.data(), bi.data(), a.k);
            }
            for (int t = 0; t < a.k; ++t) {
                outd[qq * a.k + t] = bd[(size_t)t];
                outi[qq * a.k + t] = bi[(size_t)t];
            }
        }
        const double merge_ms = (wall_time() - m0) * 1e3;
        if (size > 1)
            std::printf("[time] merge (rank0, %d x %zu candidates) %.3f ms\n", size,
                        a.q * (size_t)a.k, merge_ms);

        if (a.verify > 0)
            print_verify(verify_topk(outd.data(), outi.data(), qs.data(), a.n, a.d, a.k, a.seed,
                                     std::min<int>(a.verify, (int)a.q)));
        if (a.dump > 0) {
            const size_t nq = std::min<size_t>((size_t)a.dump, a.q);
            for (size_t qq = 0; qq < nq; ++qq)
                for (int t = 0; t < a.k; ++t)
                    std::printf("[topk] q=%zu r=%d idx=%d dist=%.6f\n", qq, t,
                                outi[qq * a.k + t], outd[qq * a.k + t]);
        }
    }

    mpi::finalize();
    return 0;
}
