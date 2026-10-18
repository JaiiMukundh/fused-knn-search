#!/usr/bin/env python3
"""Brute force vs ANN: benchmark our exact GPU search against FAISS indexes.

Objective 3 of the project asks when exact search is fast enough that
approximation becomes unnecessary. To answer that you need an approximate
index to compare against, and the problem statement asks for *established*
ones -- reference [5] is exactly the FAISS paper. So: FAISS provides the ANN
side (IVF-PQ, and HNSW where FAISS supports it), our CUDA kernel provides the
exact side, and both run on the identical vectors exported by bin/gendata.

    # 1. on any box (fast, no CUDA): export the shared dataset
    ./bin/gendata --n 1000000 --d 128 --q 1000 --k 10 --seed 42 --out data

    # 2. on the GPU box: build ground truth + benchmark the ANN indexes
    python3 bench/faiss_compare.py --data data --k 10 --nprobe 1,2,4,8,16,32 --json out/faiss.json

The ground truth is the exact neighbour set. It is taken from an existing
gt_idx.i32 if gendata produced one, otherwise computed here with an exact
FAISS IndexFlatL2 (the standard approach: measure ANN recall against a
brute-force oracle, which is also what ANN-Benchmarks does).

recall@k is the mean, over queries, of |ANN top-k  INTERSECT  exact top-k| / k.
"""

import argparse
import json
import os
import sys
import time

import numpy as np


# --------------------------------------------------------------- loading ----
def load_meta(data_dir):
    path = os.path.join(data_dir, "meta.json")
    if not os.path.isfile(path):
        sys.exit(f"missing {path} -- run ./bin/gendata first")
    with open(path) as f:
        return json.load(f)


def load_f32(path, rows, cols):
    a = np.fromfile(path, dtype="<f4")
    if a.size != rows * cols:
        sys.exit(f"{path}: expected {rows * cols} floats, found {a.size}")
    return a.reshape(rows, cols)


def load_i32(path, rows, cols):
    a = np.fromfile(path, dtype="<i4")
    if a.size != rows * cols:
        sys.exit(f"{path}: expected {rows * cols} ints, found {a.size}")
    return a.reshape(rows, cols)


# ----------------------------------------------------------------- faiss ----
def import_faiss(want_gpu):
    try:
        import faiss
    except ImportError:
        sys.exit("faiss is not installed.\n"
                 "  GPU box:  pip install faiss-gpu    (or  pip install faiss-cpu)\n"
                 "  laptop :  pip install faiss-cpu")
    n_gpu = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0
    use_gpu = bool(want_gpu and n_gpu > 0)
    return faiss, use_gpu, n_gpu


def to_gpu(faiss, index, resources, use_gpu):
    if not use_gpu:
        return index
    return faiss.index_cpu_to_gpu(resources, 0, index)


def topk_of(faiss, index, queries, k):
    """Return (distances, labels) for the k nearest neighbours of each query."""
    D, I = index.search(queries, k)
    return D, I


def recall_at_k(ann_labels, gt_labels, k):
    """Mean |ann INTERSECT exact| / k over all queries."""
    hits = 0
    for i in range(ann_labels.shape[0]):
        hits += len(set(ann_labels[i, :k].tolist()) & set(gt_labels[i, :k].tolist()))
    return hits / (ann_labels.shape[0] * k)


def timed_search(faiss, index, queries, k, repeats=1):
    """Best-of-`repeats` search latency. One untimed warm-up first."""
    index.search(queries[: min(1, len(queries))], k)   # warm-up
    best = None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        _, I = index.search(queries, k)
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return best, I


def pick_m(d, requested):
    """PQ sub-quantizer count must divide d; fall back to the largest divisor."""
    if requested and d % requested == 0:
        return requested
    for m in (16, 8, 4, 2, 1):
        if d % m == 0:
            return m
    return 1


# --------------------------------------------------------------- indexes ----
def build_flat(faiss, db, use_gpu, resources):
    index = faiss.IndexFlatL2(db.shape[1])
    index = to_gpu(faiss, index, resources, use_gpu)
    index.add(db)
    return index, {"index": "flat", "device": "gpu" if use_gpu else "cpu"}


def build_ivfpq(faiss, db, use_gpu, resources, nlist, m, nbits):
    d = db.shape[1]
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
    index = to_gpu(faiss, index, resources, use_gpu)
    index.train(db)
    index.add(db)
    return index, {"index": "ivfpq", "device": "gpu" if use_gpu else "cpu",
                   "nlist": nlist, "m": m, "nbits": nbits}


def build_hnsw(faiss, db, M, ef_construction):
    """FAISS has no GPU HNSW: this one is always CPU. Reported as such."""
    index = faiss.IndexHNSWFlat(db.shape[1], M)
    index.hnsw.efConstruction = ef_construction
    index.add(db)
    return index, {"index": "hnsw", "device": "cpu", "M": M,
                   "efConstruction": ef_construction}


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser(description="brute force (exact) vs FAISS ANN")
    ap.add_argument("--data", default="data", help="directory written by bin/gendata")
    ap.add_argument("--k", type=int, default=None, help="default: from meta.json")
    ap.add_argument("--nprobe", default="1,2,4,8,16,32",
                    help="IVF-PQ probe sweep (comma separated)")
    ap.add_argument("--efsearch", default="16,32,64,128",
                    help="HNSW efSearch sweep (comma separated)")
    ap.add_argument("--nlist", type=int, default=0, help="IVF cells (default ~4*sqrt(n))")
    ap.add_argument("--m", type=int, default=0, help="PQ sub-quantizers (default: divisor of d)")
    ap.add_argument("--nbits", type=int, default=8)
    ap.add_argument("--M", type=int, default=32, help="HNSW graph degree")
    ap.add_argument("--efconstruction", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3, help="best-of-N per measurement")
    ap.add_argument("--cpu", action="store_true", help="force CPU even if a GPU exists")
    ap.add_argument("--no-hnsw", action="store_true", help="skip HNSW (it is CPU-only)")
    ap.add_argument("--skip-flat", action="store_true", help="skip the exact FAISS baseline")
    ap.add_argument("--json", default="", help="write the result JSON here")
    args = ap.parse_args()

    meta = load_meta(args.data)
    n, d, q = meta["n"], meta["d"], meta["q"]
    k = args.k if args.k else meta["k"]
    float_bytes = 4

    print(f"[data] n={n} d={d} q={q} k={k} seed={meta['seed']}  ({args.data}/)", flush=True)
    db = load_f32(os.path.join(args.data, "db.f32"), n, d)
    queries = load_f32(os.path.join(args.data, "queries.f32"), q, d)

    faiss, use_gpu, n_gpu = import_faiss(not args.cpu)
    print(f"[faiss] {faiss.__version__}  gpus visible={n_gpu}  using={'gpu' if use_gpu else 'cpu'}",
          flush=True)
    resources = faiss.StandardGpuResources() if use_gpu else None

    result = {
        "meta": meta, "k": k,
        "faiss_version": getattr(faiss, "__version__", "?"),
        "gpus_visible": n_gpu,
        "device": "gpu" if use_gpu else "cpu",
        "ann": [],
    }

    # ---- ground truth ------------------------------------------------------
    gt_path = os.path.join(args.data, "gt_idx.i32")
    if os.path.isfile(gt_path) and not args.skip_flat:
        gt_labels = load_i32(gt_path, q, k)
        gt_source = "gendata (double-precision CPU reference)"
        print(f"[gt  ] loaded {gt_path}", flush=True)
    else:
        flat, finfo = build_flat(faiss, db, use_gpu, resources)
        _, gt_labels = topk_of(faiss, flat, queries, k)
        gt_source = f"faiss IndexFlatL2 ({finfo['device']})"
        print(f"[gt  ] computed exactly with {gt_source}", flush=True)
    result["gt_source"] = gt_source

    # ---- the exact baseline ------------------------------------------------
    if not args.skip_flat:
        flat, finfo = build_flat(faiss, db, use_gpu, resources)
        ms, _ = timed_search(faiss, flat, queries, k, args.repeats)
        exact = dict(finfo)
        exact.update(total_ms=ms * 1e3, ms_per_query=ms * 1e3 / q, recall_at_k=1.0)
        # an exact index has no recall knob; state the traffic it must move
        exact["db_read_gb"] = n * d * float_bytes * q / 1e9
        result["exact"] = exact
        print(f"[exact] flat {finfo['device']:3s}  {ms * 1e3 / q:8.3f} ms/query  "
              f"({exact['db_read_gb']:.2f} GB db reads)", flush=True)

    # ---- IVF-PQ (approximate, the FAISS workhorse) -------------------------
    nlist = args.nlist or max(1, int(4 * np.sqrt(n)))
    m = pick_m(d, args.m)
    probes = [int(x) for x in args.nprobe.split(",") if x.strip()]
    t0 = time.perf_counter()
    ivfpq, iinfo = build_ivfpq(faiss, db, use_gpu, resources, nlist, m, args.nbits)
    build_s = time.perf_counter() - t0
    print(f"[ivfpq] nlist={nlist} m={m} nbits={args.nbits} built in {build_s:.1f}s", flush=True)
    for nprobe in probes:
        if nprobe > nlist:
            continue
        ivfpq.nprobe = nprobe
        ms, labels = timed_search(faiss, ivfpq, queries, k, args.repeats)
        rec = recall_at_k(labels, gt_labels, k)
        row = dict(iinfo)
        row.update(nprobe=nprobe, build_s=build_s, recall_at_k=rec,
                   total_ms=ms * 1e3, ms_per_query=ms * 1e3 / q)
        result["ann"].append(row)
        print(f"[ivfpq] nprobe={nprobe:4d}  recall@{k}={rec:6.4f}  "
              f"{ms * 1e3 / q:8.4f} ms/query", flush=True)

    # ---- HNSW (approximate, graph-based; CPU in FAISS) ---------------------
    if not args.no_hnsw:
        t0 = time.perf_counter()
        hnsw, hinfo = build_hnsw(faiss, db, args.M, args.efconstruction)
        build_s = time.perf_counter() - t0
        print(f"[hnsw ] M={args.M} efC={args.efconstruction} built in {build_s:.1f}s", flush=True)
        for ef in [int(x) for x in args.efsearch.split(",") if x.strip()]:
            hnsw.hnsw.efSearch = ef
            ms, labels = timed_search(faiss, hnsw, queries, k, args.repeats)
            rec = recall_at_k(labels, gt_labels, k)
            row = dict(hinfo)
            row.update(efSearch=ef, build_s=build_s, recall_at_k=rec,
                       total_ms=ms * 1e3, ms_per_query=ms * 1e3 / q)
            result["ann"].append(row)
            print(f"[hnsw ] efSearch={ef:4d}  recall@{k}={rec:6.4f}  "
                  f"{ms * 1e3 / q:8.4f} ms/query", flush=True)

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[json ] wrote {args.json}", flush=True)

    # A one-line verdict for the report.
    if result["ann"] and "exact" in result:
        base = result["exact"]["ms_per_query"]
        good = [r for r in result["ann"] if r["recall_at_k"] >= 0.95]
        if good:
            best = min(good, key=lambda r: r["ms_per_query"])
            print(f"[verdict] exact {base:.4f} ms/q vs fastest ANN with recall>=0.95: "
                  f"{best['index']} {best['ms_per_query']:.4f} ms/q "
                  f"({base / best['ms_per_query']:.1f}x faster, recall={best['recall_at_k']:.4f})",
                  flush=True)
        else:
            best = max(result["ann"], key=lambda r: r["recall_at_k"])
            print(f"[verdict] exact {base:.4f} ms/q; no ANN config reached recall>=0.95 "
                  f"(best was {best['index']} at recall={best['recall_at_k']:.4f}, "
                  f"{best['ms_per_query']:.4f} ms/q -- widen the nprobe/efSearch sweep)",
                  flush=True)


if __name__ == "__main__":
    main()
