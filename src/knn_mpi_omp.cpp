// knn_mpi_omp.cpp -- MPI sharding + OpenMP threads, CPU only.
//
//   mpirun -np 2 ./knn_mpi_omp --n 1000000 --d 128 --q 32 --k 10
//   mpirun -np 4 ./knn_mpi_omp ...
//
// Deliberately runs on CPUs: it lets the whole MPI sharding + merge scheme
// (local-top-k then merge) be developed, tested and measured on any laptop,
// including one without a GPU, and it gives the report its
// "MPI + OpenMP hybrid" scaling point. knn_mpi_cuda is the GPU counterpart
// with the exact same sharding and merge.
//
// Scheme: rank r owns rows [r*n/size, (r+1)*n/size). Each rank runs the same
// top-k search over its shard for ALL queries, then only q*k (dist, idx)
// pairs per rank are gathered, and rank 0 merges them. Communication volume is
// O(size * q * k), independent of n and d -- the property the project
// statement argues makes near-linear scaling plausible.

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <numeric>
#include <vector>

#include "common.hpp"

using namespace pcap;

static void shard_bounds(int rank, int size, size_t n, size_t& r0, size_t& r1) {
    r0 = (size_t)rank * n / (size_t)size;
    r1 = (size_t)(rank + 1) * n / (size_t)size;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const Args a = parse_args(argc, argv);  // every rank sees the same argv

    size_t r0 = 0, r1 = 0;
    shard_bounds(rank, size, a.n, r0, r1);
    const size_t ln = r1 - r0;  // local rows

    // ---- local shard of the database ---------------------------------------
    std::vector<float> shard(ln * (size_t)a.d);
    for (size_t j = 0; j < ln; ++j)
        for (int t = 0; t < a.d; ++t) shard[j * a.d + t] = db_value(r0 + j, t, a.seed);

    std::vector<float> qs(a.q * (size_t)a.d);
    for (size_t r = 0; r < a.q; ++r)
        for (int t = 0; t < a.d; ++t) qs[r * a.d + t] = query_value(r, t, a.seed);

    // ---- local exact top-k for every query over the local shard ------------
    std::vector<float> l_outd(a.q * (size_t)a.k, FLT_MAX);
    std::vector<int> l_outi(a.q * (size_t)a.k, -1);

    MPI_Barrier(MPI_COMM_WORLD);
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
            for (size_t j = 0; j < ln; ++j) {
                const float* v = &shard[j * a.d];
                float acc = 0.0f;
#pragma omp simd reduction(+ : acc)
                for (int t = 0; t < a.d; ++t) {
                    const float diff = qv[t] - v[t];
                    acc += diff * diff;
                }
                insert_topk_host(acc, (int)(r0 + j), bd.data(), bi.data(), a.k);
            }
            for (int t = 0; t < a.k; ++t) {
                l_outd[(size_t)qq * a.k + t] = bd[(size_t)t];
                l_outi[(size_t)qq * a.k + t] = bi[(size_t)t];
            }
        }
    }
    const double t_local = wall_time() - t0;

    // ---- collect stats and the local top-k lists ---------------------------
    double t_local_max = 0.0;
    MPI_Reduce(&t_local, &t_local_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> t_all;
    if (rank == 0) t_all.resize((size_t)size);
    MPI_Gather(&t_local, 1, MPI_DOUBLE, rank == 0 ? t_all.data() : nullptr, 1, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

    const unsigned long long ln_ull = (unsigned long long)ln;
    std::vector<unsigned long long> rows_all;
    if (rank == 0) rows_all.resize((size_t)size);
    MPI_Gather(&ln_ull, 1, MPI_UNSIGNED_LONG_LONG, rank == 0 ? rows_all.data() : nullptr, 1,
               MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    const int per_rank = (int)(a.q * (size_t)a.k);
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

    // ---- rank 0: merge the local top-k lists (only size*k candidates/query)
    double t_merge = 0.0, t_total = 0.0;
    if (rank == 0) {
        const double tt0 = wall_time();
        const double m0 = wall_time();
        std::vector<float> outd(a.q * (size_t)a.k);
        std::vector<int> outi(a.q * (size_t)a.k);
#pragma omp parallel for schedule(dynamic, 1)
        for (long long qq = 0; qq < (long long)a.q; ++qq) {
            std::vector<float> bd((size_t)a.k, FLT_MAX);
            std::vector<int> bi((size_t)a.k, -1);
            for (int rr = 0; rr < size; ++rr) {
                const float* drow = &gd[((size_t)rr * a.q + (size_t)qq) * a.k];
                const int* irow = &gi[((size_t)rr * a.q + (size_t)qq) * a.k];
                for (int t = 0; t < a.k; ++t)
                    insert_topk_host(drow[t], irow[t], bd.data(), bi.data(), a.k);
            }
            for (int t = 0; t < a.k; ++t) {
                outd[(size_t)qq * a.k + t] = bd[(size_t)t];
                outi[(size_t)qq * a.k + t] = bi[(size_t)t];
            }
        }
        t_merge = wall_time() - m0;
        t_total = wall_time() - tt0;

        print_run_header("knn_mpi_omp", a);
        std::printf("[mpi ] ranks=%d\n", size);
        for (int rr = 0; rr < size; ++rr)
            std::printf("[rank] %2d  local search %.3f s  rows=%llu\n", rr, t_all[(size_t)rr],
                        rows_all[(size_t)rr]);
        const double pairs = (double)a.q * (double)a.n;
        const double t_sum = std::accumulate(t_all.begin(), t_all.end(), 0.0);
        std::printf("[time] local_max %.3f s  merge %.3f ms  (rank0 wall incl. gather+merge %.3f ms)\n",
                    t_local_max, t_merge * 1e3, t_total * 1e3);
        std::printf("[time] aggregate: %.3g pairs/s  %.1f GB/s db reads  %.1f GFLOP/s  "
                    "(sum(rank time)/critical path = %.2fx)\n",
                    pairs / t_local_max,
                    gbytes_per_s((double)a.q * (double)a.n * a.d * sizeof(float), t_local_max),
                    gflop_per_s(pairs * a.d * 2.0, 1.0, t_local_max),
                    t_sum / t_local_max);

        if (a.verify > 0)
            print_verify(verify_topk(outd.data(), outi.data(), qs.data(), a.n, a.d, a.k, a.seed,
                                     std::min<int>(a.verify, (int)a.q)));
    }

    MPI_Finalize();
    return 0;
}
