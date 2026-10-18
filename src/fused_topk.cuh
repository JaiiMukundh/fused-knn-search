// fused_topk.cuh -- the core CUDA piece of the project (v0, correctness-first).
//
// fused_topk_kernel: one thread block per query. Each thread scans its strided
// share of the database and keeps its own top-k in registers; the per-thread
// lists are then merged through shared memory by thread 0, and only the final
// k results are written to global memory. The N-element distance array is
// never materialised -- this is the kernel-fusion property the project is
// about.
//
// distances_kernel + select_kernel: the conventional two-pass alternative
// (write all N distances, then select), kept so knn_cuda can quantify the
// memory traffic that fusion avoids.
//
// Supported k at compile time: 1, 4, 8, 10, 16, 32 (see launch_fused()).
#pragma once

#include <cfloat>
#include <cstdio>
#include <string>

#include <cuda_runtime.h>

namespace pcap {

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool dbeats(float d0, int i0, float d1, int i1) {
    return (d0 < d1) || (d0 == d1 && i0 < i1);
}

// Insert (d, idx) into the sorted ascending top-k list if it belongs there.
template <int K>
__device__ __forceinline__ void dinsert_topk(float d, int idx, float (&bd)[K], int (&bi)[K]) {
    if (!dbeats(d, idx, bd[K - 1], bi[K - 1])) return;
    int p = K - 1;
    while (p > 0 && dbeats(d, idx, bd[p - 1], bi[p - 1])) {
        bd[p] = bd[p - 1];
        bi[p] = bi[p - 1];
        --p;
    }
    bd[p] = d;
    bi[p] = idx;
}

template <int K>
__device__ __forceinline__ void dinit_topk(float (&bd)[K], int (&bi)[K]) {
#pragma unroll
    for (int t = 0; t < K; ++t) {
        bd[t] = FLT_MAX;
        bi[t] = -1;
    }
}

// Block-level merge: every thread drops its private list into shared memory,
// thread 0 merges all blockDim.x * K entries and writes the final k results.
// `smem` must provide blockDim.x * K * (sizeof(float) + sizeof(int)) bytes.
template <int K>
__device__ __forceinline__ void dmerge_write(float (&bd)[K], int (&bi)[K], unsigned char* smem,
                                             float* outd_row, int* outi_row) {
    float* s_d = reinterpret_cast<float*>(smem);
    int* s_i = reinterpret_cast<int*>(s_d + blockDim.x * K);

#pragma unroll
    for (int t = 0; t < K; ++t) {
        s_d[threadIdx.x * K + t] = bd[t];
        s_i[threadIdx.x * K + t] = bi[t];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float rd[K];
        int ri[K];
        dinit_topk(rd, ri);
        const int total = blockDim.x * K;
        for (int e = 0; e < total; ++e) dinsert_topk<K>(s_d[e], s_i[e], rd, ri);
#pragma unroll
        for (int t = 0; t < K; ++t) {
            outd_row[t] = rd[t];
            outi_row[t] = ri[t];
        }
    }
}

// ---------------------------------------------------------------------------
// Fused kernel: one block per query, distance + selection in one pass
// ---------------------------------------------------------------------------
template <int K>
__global__ void fused_topk_kernel(const float* __restrict__ db,
                                  const float* __restrict__ queries,
                                  int n, int d,
                                  float* __restrict__ outd,
                                  int* __restrict__ outi) {
    const int q = blockIdx.x;
    const float* qv = queries + (size_t)q * d;

    float bd[K];
    int bi[K];
    dinit_topk(bd, bi);

    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const float* v = db + (size_t)j * d;
        float acc = 0.0f;
        for (int t = 0; t < d; ++t) {
            const float diff = qv[t] - v[t];
            acc = fmaf(diff, diff, acc);
        }
        dinsert_topk<K>(acc, j, bd, bi);
    }

    extern __shared__ unsigned char smem[];
    dmerge_write<K>(bd, bi, smem, outd + (size_t)q * K, outi + (size_t)q * K);
}

// ---------------------------------------------------------------------------
// Two-pass reference variant
// ---------------------------------------------------------------------------
__global__ void distances_kernel(const float* __restrict__ db,
                                 const float* __restrict__ queries,
                                 int n, int d,
                                 float* __restrict__ dist_rows) {
    const int q = blockIdx.x;
    const float* qv = queries + (size_t)q * d;
    float* row = dist_rows + (size_t)q * (size_t)n;
    const size_t stride = (size_t)gridDim.y * blockDim.x;
    for (size_t j = (size_t)blockIdx.y * blockDim.x + threadIdx.x; j < (size_t)n; j += stride) {
        const float* v = db + j * d;
        float acc = 0.0f;
        for (int t = 0; t < d; ++t) {
            const float diff = qv[t] - v[t];
            acc = fmaf(diff, diff, acc);
        }
        row[j] = acc;
    }
}

template <int K>
__global__ void select_kernel(const float* __restrict__ dist_rows, int n,
                              float* __restrict__ outd, int* __restrict__ outi) {
    const int q = blockIdx.x;
    const float* row = dist_rows + (size_t)q * (size_t)n;

    float bd[K];
    int bi[K];
    dinit_topk(bd, bi);
    for (int j = threadIdx.x; j < n; j += blockDim.x) dinsert_topk<K>(row[j], j, bd, bi);

    extern __shared__ unsigned char smem[];
    dmerge_write<K>(bd, bi, smem, outd + (size_t)q * K, outi + (size_t)q * K);
}

// ---------------------------------------------------------------------------
// Launchers / dispatch on k
// ---------------------------------------------------------------------------
template <int K>
inline int fused_block_size() { return (K > 16) ? 128 : 256; }

template <int K>
inline size_t fused_smem_bytes() {
    const int block = fused_block_size<K>();
    return (size_t)block * K * (sizeof(float) + sizeof(int));
}

template <int K>
inline cudaError_t launch_fused_t(const float* d_db, const float* d_q, int n, int d, int nq,
                                  float* d_outd, int* d_outi) {
    const int block = fused_block_size<K>();
    fused_topk_kernel<K><<<nq, block, fused_smem_bytes<K>()>>>(d_db, d_q, n, d, d_outd, d_outi);
    return cudaGetLastError();
}

template <int K>
inline cudaError_t launch_select_t(const float* d_rows, int n, int nq,
                                   float* d_outd, int* d_outi) {
    const int block = fused_block_size<K>();
    select_kernel<K><<<nq, block, fused_smem_bytes<K>()>>>(d_rows, n, d_outd, d_outi);
    return cudaGetLastError();
}

#define PCAP_FUSED_CASE(K)                                                                   \
    case K: {                                                                                \
        const cudaError_t e = launch_fused_t<K>(d_db, d_q, n, d, nq, d_outd, d_outi);        \
        if (e != cudaSuccess) { err = cudaGetErrorString(e); return false; }                 \
        return true;                                                                         \
    }

inline bool launch_fused(const float* d_db, const float* d_q, int n, int d, int nq, int k,
                         float* d_outd, int* d_outi, std::string& err) {
    switch (k) {
        PCAP_FUSED_CASE(1)
        PCAP_FUSED_CASE(4)
        PCAP_FUSED_CASE(8)
        PCAP_FUSED_CASE(10)
        PCAP_FUSED_CASE(16)
        PCAP_FUSED_CASE(32)
        default:
            err = "CUDA build supports k in {1,4,8,10,16,32}";
            return false;
    }
}

#define PCAP_SELECT_CASE(K)                                                                \
    case K: {                                                                              \
        const cudaError_t e = launch_select_t<K>(d_rows, n, nq, d_outd, d_outi);           \
        if (e != cudaSuccess) { err = cudaGetErrorString(e); return false; }               \
        return true;                                                                       \
    }

inline bool launch_select(const float* d_rows, int n, int nq, int k,
                          float* d_outd, int* d_outi, std::string& err) {
    switch (k) {
        PCAP_SELECT_CASE(1)
        PCAP_SELECT_CASE(4)
        PCAP_SELECT_CASE(8)
        PCAP_SELECT_CASE(10)
        PCAP_SELECT_CASE(16)
        PCAP_SELECT_CASE(32)
        default:
            err = "CUDA build supports k in {1,4,8,10,16,32}";
            return false;
    }
}

inline bool cuda_k_supported(int k) {
    return k == 1 || k == 4 || k == 8 || k == 10 || k == 16 || k == 32;
}

}  // namespace pcap
