// knn_cuda.cu -- single-GPU exact kNN search, fused distance + top-k.
//
//   ./knn_cuda --n 1000000 --d 128 --q 32 --k 10 --verify 4
//
// Times (per stage, CUDA events):
//   1. H2D upload of database + queries
//   2. the fused kernel       (the project's core contribution)
//   3. a two-pass equivalent  (distance kernel writes all N distances to
//      global memory, selection kernel reads them back) for comparison
//
// Verifies its output against the double-precision CPU reference in common.hpp.

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common.hpp"
#include "fused_topk.cuh"

using namespace pcap;

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

static void fill_db(std::vector<float>& db, size_t n, int d, uint64_t seed) {
    for (size_t r = 0; r < n; ++r)
        for (int t = 0; t < d; ++t) db[r * d + t] = db_value(r, t, seed);
}

static void fill_queries(std::vector<float>& qs, size_t nq, int d, uint64_t seed) {
    for (size_t r = 0; r < nq; ++r)
        for (int t = 0; t < d; ++t) qs[r * d + t] = query_value(r, t, seed);
}

int main(int argc, char** argv) {
    const Args a = parse_args(argc, argv);
    if (a.n > (size_t)INT_MAX || a.q > (size_t)INT_MAX) {
        std::fprintf(stderr, "--n/--q must be <= INT_MAX for the CUDA kernels\n");
        return 2;
    }
    if (!cuda_k_supported(a.k)) {
        std::fprintf(stderr, "CUDA build supports --k in {1,4,8,10,16,32} (got %d)\n", a.k);
        return 2;
    }

    int ndev = 0;
    ck(cudaGetDeviceCount(&ndev), "cudaGetDeviceCount");
    if (ndev < 1) {
        std::fprintf(stderr, "no CUDA device visible; run this on the GPU box (Colab)\n");
        return 1;
    }
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    if (!a.n) return 2;

    print_run_header("knn_cuda", a);
#if CUDART_VERSION >= 13000
    // CUDA 13 removed memoryClockRate/memoryBusWidth from cudaDeviceProp.
    std::printf("[gpu ] %s (sm_%d%d, %.1f GB)\n", prop.name,
                prop.major, prop.minor, prop.totalGlobalMem / 1073741824.0);
#else
    std::printf("[gpu ] %s (sm_%d%d, %.1f GB, %.1f GB/s peak)\n", prop.name,
                prop.major, prop.minor, prop.totalGlobalMem / 1073741824.0,
                2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
#endif

    const size_t db_bytes = a.n * (size_t)a.d * sizeof(float);
    const size_t q_bytes = a.q * (size_t)a.d * sizeof(float);

    std::vector<float> h_db(a.n * (size_t)a.d);
    std::vector<float> h_q(a.q * (size_t)a.d);
    fill_db(h_db, a.n, a.d, a.seed);
    fill_queries(h_q, a.q, a.d, a.seed);

    float *d_db = nullptr, *d_q = nullptr, *d_outd = nullptr;
    int* d_outi = nullptr;
    ck(cudaMalloc(&d_db, db_bytes), "cudaMalloc db");
    ck(cudaMalloc(&d_q, q_bytes), "cudaMalloc queries");
    ck(cudaMalloc(&d_outd, a.q * a.k * sizeof(float)), "cudaMalloc outd");
    ck(cudaMalloc(&d_outi, a.q * a.k * sizeof(int)), "cudaMalloc outi");

    cudaEvent_t ev0, ev1;
    ck(cudaEventCreate(&ev0), "event");
    ck(cudaEventCreate(&ev1), "event");

    ck(cudaEventRecord(ev0), "record");
    ck(cudaMemcpy(d_db, h_db.data(), db_bytes, cudaMemcpyHostToDevice), "H2D db");
    ck(cudaMemcpy(d_q, h_q.data(), q_bytes, cudaMemcpyHostToDevice), "H2D queries");
    ck(cudaEventRecord(ev1), "record");
    ck(cudaEventSynchronize(ev1), "sync");
    float h2d_ms = 0.f;
    ck(cudaEventElapsedTime(&h2d_ms, ev0, ev1), "elapsed");
    std::printf("[H2D ] upload %.1f MB in %.2f ms (%.1f GB/s)\n",
                (db_bytes + q_bytes) / 1e6, h2d_ms, gbytes_per_s((double)(db_bytes + q_bytes), h2d_ms / 1e3));

    // --- fused run (one warm-up, then timed) ---------------------------------
    std::string err;
    if (!launch_fused(d_db, d_q, (int)a.n, a.d, (int)a.q, a.k, d_outd, d_outi, err)) {
        std::fprintf(stderr, "fused launch failed: %s\n", err.c_str());
        return 1;
    }
    ck(cudaDeviceSynchronize(), "warmup sync");

    ck(cudaEventRecord(ev0), "record");
    if (!launch_fused(d_db, d_q, (int)a.n, a.d, (int)a.q, a.k, d_outd, d_outi, err)) {
        std::fprintf(stderr, "fused launch failed: %s\n", err.c_str());
        return 1;
    }
    ck(cudaEventRecord(ev1), "record");
    ck(cudaEventSynchronize(ev1), "sync");
    float fused_ms = 0.f;
    ck(cudaEventElapsedTime(&fused_ms, ev0, ev1), "elapsed");

    const double read_bytes = (double)a.q * (double)a.n * a.d * sizeof(float);
    std::printf("[time] fused   %8.3f ms  (%.3f ms/query)  %7.1f GB/s db reads  %7.1f GFLOP/s\n",
                fused_ms, fused_ms / (double)a.q,
                gbytes_per_s(read_bytes, fused_ms / 1e3),
                gflop_per_s((double)a.q * (double)a.n * a.d * 2.0, 1.0, fused_ms / 1e3));

    std::vector<float> h_outd(a.q * a.k);
    std::vector<int> h_outi(a.q * a.k);
    ck(cudaMemcpy(h_outd.data(), d_outd, a.q * a.k * sizeof(float), cudaMemcpyDeviceToHost), "D2H outd");
    ck(cudaMemcpy(h_outi.data(), d_outi, a.q * a.k * sizeof(int), cudaMemcpyDeviceToHost), "D2H outi");
    if (a.verify > 0)
        print_verify(verify_topk(h_outd.data(), h_outi.data(), h_q.data(), a.n, a.d, a.k, a.seed,
                                 std::min<int>(a.verify, (int)a.q)));
    if (a.dump > 0) {
        const size_t nq = std::min<size_t>((size_t)a.dump, a.q);
        for (size_t qq = 0; qq < nq; ++qq)
            for (int t = 0; t < a.k; ++t)
                std::printf("[topk] q=%zu r=%d idx=%d dist=%.6f\n", qq, t,
                            h_outi[qq * a.k + t], h_outd[qq * a.k + t]);
    }

    // --- two-pass run (for the fusion comparison) ----------------------------
    if (a.twopass) {
        const size_t rows_bytes = (size_t)a.q * a.n * sizeof(float);
        size_t free_mem = 0, total_mem = 0;
        ck(cudaMemGetInfo(&free_mem, &total_mem), "cudaMemGetInfo");
        if (rows_bytes > free_mem / 2) {
            std::printf("[skip] two-pass needs %.1f GB for the distance array, only %.1f GB free\n",
                        rows_bytes / 1e9, free_mem / 1e9);
        } else {
            float* d_rows = nullptr;
            float* d_outd2 = nullptr;
            int* d_outi2 = nullptr;
            ck(cudaMalloc(&d_rows, rows_bytes), "cudaMalloc rows");
            ck(cudaMalloc(&d_outd2, a.q * a.k * sizeof(float)), "cudaMalloc outd2");
            ck(cudaMalloc(&d_outi2, a.q * a.k * sizeof(int)), "cudaMalloc outi2");

            const unsigned chunks =
                (unsigned)std::min<size_t>((a.n + 255) / 256, 4096);  // grid y cap
            const dim3 dgrid((unsigned)a.q, chunks);

            ck(cudaEventRecord(ev0), "record");
            distances_kernel<<<dgrid, 256>>>(d_db, d_q, (int)a.n, a.d, d_rows);
            ck(cudaGetLastError(), "distances_kernel");
            ck(cudaEventRecord(ev1), "record");
            ck(cudaEventSynchronize(ev1), "sync");
            float dist_ms = 0.f;
            ck(cudaEventElapsedTime(&dist_ms, ev0, ev1), "elapsed");

            if (!launch_select(d_rows, (int)a.n, (int)a.q, a.k, d_outd2, d_outi2, err)) {
                std::fprintf(stderr, "select launch failed: %s\n", err.c_str());
                return 1;
            }
            ck(cudaEventRecord(ev0), "record");
            if (!launch_select(d_rows, (int)a.n, (int)a.q, a.k, d_outd2, d_outi2, err)) {
                std::fprintf(stderr, "select launch failed: %s\n", err.c_str());
                return 1;
            }
            ck(cudaEventRecord(ev1), "record");
            ck(cudaEventSynchronize(ev1), "sync");
            float select_ms = 0.f;
            ck(cudaEventElapsedTime(&select_ms, ev0, ev1), "elapsed");

            const float two_ms = dist_ms + select_ms;
            const double write_bytes = (double)rows_bytes;
            std::printf("[time] twopass %7.3f ms  (dist %.3f + select %.3f)  writes %.1f MB "
                        "of distance array; fused is %.2fx faster\n",
                        two_ms, dist_ms, select_ms, rows_bytes / 1e6,
                        two_ms > 0.f ? two_ms / fused_ms : 0.0);
            std::printf("[time] twopass extra traffic vs fused: %.1f MB written + %.1f MB re-read\n",
                        write_bytes / 1e6, write_bytes / 1e6);

            if (a.verify > 0) {
                std::vector<float> h2d2(a.q * a.k);
                std::vector<int> h2i2(a.q * a.k);
                ck(cudaMemcpy(h2d2.data(), d_outd2, a.q * a.k * sizeof(float), cudaMemcpyDeviceToHost), "D2H2");
                ck(cudaMemcpy(h2i2.data(), d_outi2, a.q * a.k * sizeof(int), cudaMemcpyDeviceToHost), "D2H2");
                std::printf("[verify] two-pass: ");
                const VerifyReport rep2 = verify_topk(h2d2.data(), h2i2.data(), h_q.data(), a.n, a.d,
                                                      a.k, a.seed, std::min<int>(a.verify, (int)a.q));
                std::printf("top1 %d/%d rel=%.3g -> %s\n", rep2.top1_hits, rep2.checked,
                            rep2.max_kth_rel, rep2.pass ? "PASS" : "FAIL");
            }

            ck(cudaFree(d_rows), "free");
            ck(cudaFree(d_outd2), "free");
            ck(cudaFree(d_outi2), "free");
        }
    }

    ck(cudaFree(d_db), "free");
    ck(cudaFree(d_q), "free");
    ck(cudaFree(d_outd), "free");
    ck(cudaFree(d_outi), "free");
    ck(cudaEventDestroy(ev0), "destroy");
    ck(cudaEventDestroy(ev1), "destroy");
    return 0;
}
