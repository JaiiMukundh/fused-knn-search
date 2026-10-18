// knn_omp.cpp -- multicore CPU exact kNN baseline (OpenMP).
//
//   OMP_NUM_THREADS=4 ./knn_omp --n 1000000 --d 128 --q 32 --k 10 --verify 4
//   ./knn_omp --threads 1 ...            # serial baseline for the report
//
// Parallelism is over queries (each query is independent); the inner distance
// loop is additionally vectorised with `omp simd`. This is the "multicore CPU
// baseline" side of the evaluation; the same program with --threads 1 is the
// serial baseline.

#include <omp.h>

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <vector>

#include "common.hpp"

using namespace pcap;

int main(int argc, char** argv) {
    const Args a = parse_args(argc, argv);
    if (a.k > 1024) {
        std::fprintf(stderr, "k too large for the CPU baseline (max 1024)\n");
        return 2;
    }
    if (a.threads > 0) omp_set_num_threads(a.threads);

    int max_threads = 1;
#pragma omp parallel
    {
#pragma omp master
        max_threads = omp_get_num_threads();
    }

    print_run_header("knn_omp", a);
    std::printf("[cpu ] threads=%d\n", max_threads);

    // Materialise the database and queries (single allocation, generated with
    // the shared deterministic generator).
    std::vector<float> db(a.n * (size_t)a.d);
    std::vector<float> qs(a.q * (size_t)a.d);
#pragma omp parallel for schedule(static)
    for (long long r = 0; r < (long long)a.n; ++r)
        for (int t = 0; t < a.d; ++t) db[(size_t)r * a.d + t] = db_value((size_t)r, t, a.seed);
#pragma omp parallel for schedule(static)
    for (long long r = 0; r < (long long)a.q; ++r)
        for (int t = 0; t < a.d; ++t) qs[(size_t)r * a.d + t] = query_value((size_t)r, t, a.seed);

    std::vector<float> outd(a.q * (size_t)a.k, FLT_MAX);
    std::vector<int> outi(a.q * (size_t)a.k, -1);

    const double t0 = wall_time();
#pragma omp parallel
    {
        std::vector<float> bd((size_t)a.k);
        std::vector<int> bi((size_t)a.k);
#pragma omp for schedule(dynamic, 1)
        for (long long qq = 0; qq < (long long)a.q; ++qq) {
            for (int t = 0; t < a.k; ++t) {
                bd[(size_t)t] = FLT_MAX;
                bi[(size_t)t] = -1;
            }
            const float* qv = &qs[(size_t)qq * a.d];
            for (size_t j = 0; j < a.n; ++j) {
                const float* v = &db[j * (size_t)a.d];
                float acc = 0.0f;
#pragma omp simd reduction(+ : acc)
                for (int t = 0; t < a.d; ++t) {
                    const float diff = qv[t] - v[t];
                    acc += diff * diff;
                }
                insert_topk_host(acc, (int)j, bd.data(), bi.data(), a.k);
            }
            for (int t = 0; t < a.k; ++t) {
                outd[(size_t)qq * a.k + t] = bd[(size_t)t];
                outi[(size_t)qq * a.k + t] = bi[(size_t)t];
            }
        }
    }
    const double dt = wall_time() - t0;

    const double pairs = (double)a.q * (double)a.n;
    std::printf("[time] cpu     %8.3f s   (%.3f ms/query)  %.3g pairs/s  %.1f GB/s db reads  %.1f GFLOP/s\n",
                dt, dt * 1e3 / (double)a.q, pairs / dt,
                gbytes_per_s((double)a.q * (double)a.n * a.d * sizeof(float), dt),
                gflop_per_s(pairs * a.d * 2.0, 1.0, dt));

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
    return 0;
}
