// gendata.cpp -- export the deterministic dataset so an external index (FAISS)
// can be benchmarked on exactly the same vectors our CUDA kernel search.
//
//   ./bin/gendata --n 1000000 --d 128 --q 1000 --k 10 --seed 42 --out data
//
// Writes into <out>/:
//   db.f32        n*d float32, row-major, little-endian
//   queries.f32   q*d float32, row-major, little-endian
//   gt_idx.i32    q*k int32, row-major, ascending by distance   (optional)
//   gt_dist.f32   q*k float32, same order                       (optional)
//   meta.json     {"n","d","q","k","seed","dtype","order","gt":bool}
//
// The generator is pcap::db_value() from common.hpp -- the same function every
// other binary uses -- so these files are bit-identical to what knn_cuda /
// knn_omp / knn_mpi_* generate internally. That is the whole point: FAISS and
// our kernel must not be fed subtly different data.
//
// No CUDA, no MPI: builds and runs on any machine with g++.
//
// Ground truth (gt_*) is the double-precision CPU reference from common.hpp,
// computed one query per thread. It is O(q*n*d) and therefore only computed
// when that fits --gt-budget; when it does not, the stale gt_* files are
// removed and meta.json records "gt": false so the benchmark computes an exact
// ground truth with FAISS IndexFlat instead (see bench/faiss_compare.py).

#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "common.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace pcap;

namespace {

// gt_idx.i32 / gt_dist.f32 are optional outputs, and they must never be read
// back at a shape they were not written at: a leftover gt file from an earlier
// --q (say 200 rows when this run wants 1000) silently makes the ANN benchmark
// compare against the wrong ground truth. Start every run by removing them;
// they are rewritten below only when the reference is actually computed.
void remove_stale_gt(const std::string& gi, const std::string& gd) {
    if (std::remove(gi.c_str()) == 0) std::printf("[clean] removed stale %s\n", gi.c_str());
    if (std::remove(gd.c_str()) == 0) std::printf("[clean] removed stale %s\n", gd.c_str());
}

bool write_all(FILE* f, const void* p, size_t bytes) {
    return bytes == 0 || std::fwrite(p, 1, bytes, f) == bytes;
}

// Stream the database in row blocks: n*d can be many GB and there is no reason
// to hold it in host memory just to copy it to a file.
bool write_db(const std::string& path, size_t n, int d, uint64_t seed) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "cannot open %s: %s\n", path.c_str(), std::strerror(errno));
        return false;
    }
    constexpr size_t kRows = 4096;
    std::vector<float> buf(kRows * (size_t)d);
    for (size_t r0 = 0; r0 < n; r0 += kRows) {
        const size_t rows = std::min(kRows, n - r0);
        for (size_t r = 0; r < rows; ++r)
            for (int t = 0; t < d; ++t) buf[r * d + t] = db_value(r0 + r, t, seed);
        if (!write_all(f, buf.data(), rows * (size_t)d * sizeof(float))) {
            std::fprintf(stderr, "short write to %s\n", path.c_str());
            std::fclose(f);
            return false;
        }
    }
    return std::fclose(f) == 0;
}

bool write_queries(const std::string& path, size_t q, int d, uint64_t seed) {
    std::vector<float> buf(q * (size_t)d);
    for (size_t r = 0; r < q; ++r)
        for (int t = 0; t < d; ++t) buf[r * d + t] = query_value(r, t, seed);
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "cannot open %s: %s\n", path.c_str(), std::strerror(errno));
        return false;
    }
    const bool ok = write_all(f, buf.data(), buf.size() * sizeof(float));
    if (!ok) std::fprintf(stderr, "short write to %s\n", path.c_str());
    return std::fclose(f) == 0 && ok;
}

bool write_meta(const std::string& path, const Args& a, bool have_gt) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "cannot open %s: %s\n", path.c_str(), std::strerror(errno));
        return false;
    }
    std::fprintf(f,
                 "{\n"
                 "  \"n\": %zu,\n  \"d\": %d,\n  \"q\": %zu,\n  \"k\": %d,\n"
                 "  \"seed\": %" PRIu64 ",\n"
                 "  \"dtype\": \"float32\",\n  \"order\": \"row-major\",\n"
                 "  \"metric\": \"l2-squared\",\n  \"gt\": %s\n}\n",
                 a.n, a.d, a.q, a.k, a.seed, have_gt ? "true" : "false");
    return std::fclose(f) == 0;
}

}  // namespace

int main(int argc, char** argv) {
    // Pull our own flags out first: common.hpp's parse_args() rejects anything
    // it does not know, so it must only ever see the shared flags.
    std::string out = "data";
    double gt_budget = 2e11;   // max q*n*d ops for the in-process reference
    std::vector<char*> rest;
    rest.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string f = argv[i];
        if (f == "--out" || f == "--gt-budget") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", f.c_str());
                return 2;
            }
            const std::string v = argv[++i];
            if (f == "--out") out = v;
            else gt_budget = std::strtod(v.c_str(), nullptr);
        } else {
            rest.push_back(argv[i]);
        }
    }
    Args a = parse_args((int)rest.size(), rest.data());
    if (out.empty()) { std::fprintf(stderr, "--out must not be empty\n"); return 2; }
#if defined(_OPENMP)
    if (a.threads > 0) omp_set_num_threads(a.threads);
#endif

    const std::string cmd = "mkdir -p '" + out + "'";
    if (std::system(cmd.c_str()) != 0) {
        std::fprintf(stderr, "cannot create %s\n", out.c_str());
        return 2;
    }

    print_run_header("gendata", a);
    std::printf("[out ] %s/\n", out.c_str());

    const std::string db_path = out + "/db.f32";
    const std::string q_path = out + "/queries.f32";
    const std::string gi_path = out + "/gt_idx.i32";
    const std::string gd_path = out + "/gt_dist.f32";

    if (!write_db(db_path, a.n, a.d, a.seed)) return 1;
    std::printf("[write] db.f32       %zu x %d float32  (%.1f MB)\n", a.n, a.d,
                (double)a.n * a.d * sizeof(float) / 1e6);
    if (!write_queries(q_path, a.q, a.d, a.seed)) return 1;
    std::printf("[write] queries.f32  %zu x %d float32  (%.1f MB)\n", a.q, a.d,
                (double)a.q * a.d * sizeof(float) / 1e6);

    // Exact ground truth, when the double-precision scan is affordable.
    // The stale-file cleanup happens in both branches: see remove_stale_gt().
    const double ops = (double)a.q * (double)a.n * (double)a.d;
    const bool want_gt = ops <= gt_budget;
    remove_stale_gt(gi_path, gd_path);

    if (want_gt) {
        std::vector<int> idx(a.q * (size_t)a.k);
        std::vector<float> dist(a.q * (size_t)a.k);
        const double t0 = wall_time();
        // One query per thread: cpu_topk_from_fn() is stateless, so this is
        // embarrassingly parallel, and q*n*d is the whole cost of the program.
#if defined(_OPENMP)
        const int nthreads = omp_get_max_threads();
#else
        const int nthreads = 1;
#endif
        std::printf("[gt  ] double-precision reference over %zu queries on %d thread%s ...\n",
                    a.q, nthreads, nthreads == 1 ? "" : "s");
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
        for (long long qq = 0; qq < (long long)a.q; ++qq) {
            // The query vector is regenerated per query so q*d never has to sit
            // in memory next to the streamed database (and each thread owns it).
            std::vector<float> query((size_t)a.d);
            for (int t = 0; t < a.d; ++t) query[t] = query_value((size_t)qq, t, a.seed);
            const std::vector<Hit> ref = cpu_topk_from_fn(query.data(), a.n, a.d, a.k, a.seed);
            for (int j = 0; j < a.k && j < (int)ref.size(); ++j) {
                idx[(size_t)qq * a.k + j] = ref[j].idx;
                dist[(size_t)qq * a.k + j] = (float)ref[j].dist;
            }
        }
        FILE* fi = std::fopen(gi_path.c_str(), "wb");
        FILE* fd = std::fopen(gd_path.c_str(), "wb");
        if (!fi || !fd) {
            std::fprintf(stderr, "cannot open gt outputs: %s\n", std::strerror(errno));
            if (fi) std::fclose(fi);
            if (fd) std::fclose(fd);
            return 1;
        }
        const bool ok = write_all(fi, idx.data(), idx.size() * sizeof(int)) &&
                        write_all(fd, dist.data(), dist.size() * sizeof(float));
        std::fclose(fi);
        std::fclose(fd);
        if (!ok) { std::fprintf(stderr, "short write to gt outputs\n"); return 1; }
        std::printf("[write] gt_idx.i32   %zu x %d int32     (%.3f MB)\n", a.q, a.k,
                    (double)a.q * a.k * sizeof(int) / 1e6);
        std::printf("[gt  ] double-precision reference in %.1f s\n", wall_time() - t0);
    } else {
        std::printf("[gt  ] skipped: q*n*d = %.3g > --gt-budget %.3g\n", ops, gt_budget);
        std::printf("[gt  ] no gt_*.i32 written to %s/; bench/faiss_compare.py will compute\n", out.c_str());
        std::printf("[gt  ] an exact ground truth with FAISS IndexFlat instead. Pass\n");
        std::printf("[gt  ] --gt-budget %.3g to build the double-precision one here.\n", ops);
    }

    if (!write_meta(out + "/meta.json", a, want_gt)) return 1;
    std::printf("[write] meta.json    gt=%s\n", want_gt ? "true" : "false");
    return 0;
}
