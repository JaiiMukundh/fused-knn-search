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
// Ground truth (gt_*) is the double-precision CPU reference from common.hpp.
// It is O(q*n*d) and therefore only computed when that fits --gt-budget;
// on the GPU box you would instead compute the exact ground truth with
// FAISS IndexFlatL2 / the fused kernel (see bench/faiss_compare.py).

#include <cerrno>
#include <cinttypes>
#include <cstring>
#include <string>
#include <vector>

#include "common.hpp"

using namespace pcap;

namespace {

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
    double gt_budget = 2e10;   // max q*n*d operations for the in-process reference
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
    const double ops = (double)a.q * (double)a.n * (double)a.d;
    if (ops <= gt_budget) {
        std::vector<int> idx(a.q * (size_t)a.k);
        std::vector<float> dist(a.q * (size_t)a.k);
        const double t0 = wall_time();
        // The reference needs the query vector itself; regenerate it here so we
        // never have to keep q*d in memory alongside the streamed database.
        std::vector<float> query((size_t)a.d);
        for (size_t qq = 0; qq < a.q; ++qq) {
            for (int t = 0; t < a.d; ++t) query[t] = query_value(qq, t, a.seed);
            const std::vector<Hit> ref = cpu_topk_from_fn(query.data(), a.n, a.d, a.k, a.seed);
            for (int j = 0; j < a.k && j < (int)ref.size(); ++j) {
                idx[qq * a.k + j] = ref[j].idx;
                dist[qq * a.k + j] = (float)ref[j].dist;
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
        std::printf("[gt  ] compute ground truth on the GPU box instead "
                    "(bench/faiss_compare.py --make-gt)\n");
    }

    if (!write_meta(out + "/meta.json", a, ops <= gt_budget)) return 1;
    std::printf("[write] meta.json\n");
    return 0;
}
