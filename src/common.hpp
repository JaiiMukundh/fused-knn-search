// common.hpp -- shared helpers for the PCAP mini-project
// "GPU-Accelerated Exact Vector Similarity Search"
//
// Everything here is deliberately deterministic and stateless so that:
//   * every MPI rank materialises only its own shard of the database while
//     still agreeing on one single global dataset (see db_value()), and
//   * the CPU reference (double precision) can check any program's output.
//
// Ordering convention used everywhere: candidate (dist, idx) beats
// (dist2, idx2) iff dist < dist2, or dist == dist2 and idx < idx2.
#pragma once

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <string>
#include <vector>

namespace pcap {

// ---------------------------------------------------------------------------
// Args
// ---------------------------------------------------------------------------
struct Args {
    size_t n = 1u << 20;   // database size
    int d = 128;           // dimensionality
    size_t q = 32;         // number of queries
    int k = 10;            // neighbours to return
    uint64_t seed = 42;    // dataset seed
    int threads = 0;       // OpenMP threads (0 = use all)
    int verify = 4;        // queries checked against the CPU reference (0 = off)
    int twopass = 1;       // knn_cuda only: also time the two-pass variant
    int dump = 0;          // print top-k of the first M queries as [topk] lines
};

inline void print_usage(const char* prog) {
    std::printf(
        "Usage: %s [options]\n"
        "  --n N          database size            (default 1048576)\n"
        "  --d D          dimensionality           (default 128)\n"
        "  --q Q          number of queries        (default 32)\n"
        "  --k K          neighbours               (default 10; CUDA supports 1,4,8,10,16,32)\n"
        "  --seed S       dataset seed             (default 42)\n"
        "  --threads T    OpenMP threads           (0 = all, default 0)\n"
        "  --verify V     queries checked vs CPU reference (default 4, 0 = off)\n"
        "  --twopass 0|1  knn_cuda: also time two-pass variant (default 1)\n"
        "  --dump M       print top-k of the first M queries as [topk] lines (default 0)\n"
        "  --help\n", prog);
}

inline Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string f = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", what); std::exit(2); }
            return argv[++i];
        };
        if (f == "--n")            a.n = (size_t)std::strtoull(next("--n").c_str(), nullptr, 10);
        else if (f == "--d")       a.d = std::atoi(next("--d").c_str());
        else if (f == "--q")       a.q = (size_t)std::strtoull(next("--q").c_str(), nullptr, 10);
        else if (f == "--k")       a.k = std::atoi(next("--k").c_str());
        else if (f == "--seed")    a.seed = std::strtoull(next("--seed").c_str(), nullptr, 10);
        else if (f == "--threads") a.threads = std::atoi(next("--threads").c_str());
        else if (f == "--verify")  a.verify = std::atoi(next("--verify").c_str());
        else if (f == "--twopass") a.twopass = std::atoi(next("--twopass").c_str());
        else if (f == "--dump")    a.dump = std::atoi(next("--dump").c_str());
        else if (f == "--help" || f == "-h") { print_usage(argv[0]); std::exit(0); }
        else { std::fprintf(stderr, "unknown option: %s\n", f.c_str()); print_usage(argv[0]); std::exit(2); }
    }
    if (a.n < 1 || a.d < 1 || a.q < 1 || a.k < 1 || a.k > 1024) {
        std::fprintf(stderr, "invalid sizes (need n,d,q,k >= 1 and k <= 1024)\n");
        std::exit(2);
    }
    if (a.n < (size_t)a.k) {
        std::fprintf(stderr, "n must be >= k\n");
        std::exit(2);
    }
    return a;
}

inline void print_run_header(const char* prog, const Args& a) {
    std::printf("[run] %s  n=%zu d=%d q=%zu k=%d seed=%llu\n",
                prog, a.n, a.d, a.q, a.k, (unsigned long long)a.seed);
}

// ---------------------------------------------------------------------------
// Deterministic dataset: db_value(row, dim) is the *global* value, so a rank
// computing rows [r0, r1) generates exactly the same numbers as any other
// program that generates the full database.
// ---------------------------------------------------------------------------
inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

inline float unit01(uint64_t x) {                 // [0, 1)
    return (float)((x >> 40) * 0x1.0p-24);        // top 24 bits -> exactly representable
}

inline float db_value(size_t row, int dim, uint64_t seed) {
    const uint64_t h = (uint64_t)row * 0x9E3779B97F4A7C15ull ^
                       (uint64_t)(uint32_t)dim * 0xBF58476D1CE4E5B9ull ^
                       seed;
    return unit01(splitmix64(h)) * 2.0f - 1.0f;   // uniform in [-1, 1)
}

inline float query_value(size_t q, int dim, uint64_t seed) {
    return db_value(q, dim, seed ^ 0x51ED270B0DDC0DEull);
}

// ---------------------------------------------------------------------------
// CPU reference top-k: double precision, exact ordering, O(n log k) memory.
// Streams the database through db_value() so it never materialises it.
// ---------------------------------------------------------------------------
struct Hit {
    double dist = 0.0;
    int idx = -1;
};

inline bool hit_less(const Hit& a, const Hit& b) {
    return a.dist < b.dist || (a.dist == b.dist && a.idx < b.idx);
}

// priority_queue "less-than": a comes before b iff a is *better* than b.
// priority_queue keeps the LAST element of that ordering on top, i.e. top()
// is the WORST kept hit -- which is what we want to test candidates against.
struct HitWorstFirst {
    bool operator()(const Hit& a, const Hit& b) const { return hit_less(a, b); }
};

inline std::vector<Hit> cpu_topk_from_fn(const float* query, size_t n, int d, int k, uint64_t seed) {
    std::priority_queue<Hit, std::vector<Hit>, HitWorstFirst> heap;
    for (size_t j = 0; j < n; ++j) {
        double acc = 0.0;
        for (int t = 0; t < d; ++t) {
            const double diff = (double)query[t] - (double)db_value(j, t, seed);
            acc += diff * diff;
        }
        const Hit h{acc, (int)j};
        if ((int)heap.size() < k) {
            heap.push(h);
        } else if (hit_less(h, heap.top())) {
            heap.pop();
            heap.push(h);
        }
    }
    std::vector<Hit> out(heap.size());
    for (size_t i = out.size(); i-- > 0;) { out[i] = heap.top(); heap.pop(); }  // ascending
    return out;
}

// ---------------------------------------------------------------------------
// Top-k bookkeeping shared by the OpenMP baselines (the CUDA kernels have the
// device equivalent in fused_topk.cuh; the ordering rule is the same).
// ---------------------------------------------------------------------------
inline bool beats(float d0, int i0, float d1, int i1) {
    return (d0 < d1) || (d0 == d1 && i0 < i1);
}

// bd/bi hold the current k best as a sorted list (ascending); cost is O(k) per
// insertion but only for candidates that beat bd[k-1], which is rare.
inline void insert_topk_host(float d, int idx, float* bd, int* bi, int k) {
    if (!beats(d, idx, bd[k - 1], bi[k - 1])) return;
    int p = k - 1;
    while (p > 0 && beats(d, idx, bd[p - 1], bi[p - 1])) {
        bd[p] = bd[p - 1];
        bi[p] = bi[p - 1];
        --p;
    }
    bd[p] = d;
    bi[p] = idx;
}

// ---------------------------------------------------------------------------
// Verification against the CPU reference.
//
// The GPU accumulates in float while the reference uses double, so the k-th
// distance is compared with a tolerance instead of bit-for-bit. Top-1 must
// match exactly on every checked query; the k-th distance must agree to
// ~1e-3 relative, which is generous for float32 but still fails on any real
// selection bug.
// ---------------------------------------------------------------------------
struct VerifyReport {
    int checked = 0;
    int top1_hits = 0;
    double max_kth_abs = 0.0;
    double max_kth_rel = 0.0;
    bool pass = false;
};

inline VerifyReport verify_topk(const float* outd, const int* outi, const float* queries,
                                size_t n, int d, int k, uint64_t seed, int ncheck) {
    VerifyReport rep;
    for (int qq = 0; qq < ncheck; ++qq) {
        const std::vector<Hit> ref = cpu_topk_from_fn(queries + (size_t)qq * d, n, d, k, seed);
        const float* gd = outd + (size_t)qq * k;
        const int* gi = outi + (size_t)qq * k;
        if (!ref.empty() && gi[0] == ref[0].idx) ++rep.top1_hits;
        const double gk = (double)gd[k - 1];
        const double rk = ref.empty() ? 0.0 : ref[(size_t)k - 1].dist;
        const double absd = std::fabs(gk - rk);
        rep.max_kth_abs = std::max(rep.max_kth_abs, absd);
        rep.max_kth_rel = std::max(rep.max_kth_rel, absd / (1.0 + rk));
    }
    rep.checked = ncheck;
    rep.pass = (rep.top1_hits == rep.checked) && (rep.max_kth_rel < 1e-3);
    return rep;
}

inline void print_verify(const VerifyReport& rep) {
    if (rep.checked == 0) { std::printf("[verify] skipped\n"); return; }
    std::printf("[verify] top1 %d/%d  max |kth dist diff| abs=%.3g rel=%.3g  -> %s\n",
                rep.top1_hits, rep.checked, rep.max_kth_abs, rep.max_kth_rel,
                rep.pass ? "PASS" : "FAIL");
}

// ---------------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------------
inline double wall_time() {   // seconds, monotonic
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

inline double gbytes_per_s(double bytes, double seconds) {
    return seconds > 0.0 ? bytes / seconds / 1e9 : 0.0;
}

inline double gflop_per_s(double pairs, double flops_per_pair, double seconds) {
    return seconds > 0.0 ? pairs * flops_per_pair / seconds / 1e9 : 0.0;
}

}  // namespace pcap
