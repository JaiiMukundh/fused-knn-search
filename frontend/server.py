#!/usr/bin/env python3
"""Web frontend + JSON API for the fused-kNN project.

Runs on the GPU box (Colab / any Linux host). Serves one HTML page and
a small API that shells out to the compiled binaries in <repo>/bin and parses
their [run] / [time] / [verify] / [topk] output lines.

    python3 frontend/server.py --port 8080

Endpoints
    GET  /                  the single-page UI
    GET  /api/state         GPU / engines / limits / git info
    GET  /api/health        liveness probe (used by the tunnel test)
    POST /api/search        run one exact kNN search, return timings + top-k
    POST /api/benchmark     run a size sweep (fused vs two-pass), return rows
    POST /api/build         run `make gpu` (or cpu), return the log tail

Only one heavy job runs at a time; concurrent requests get HTTP 409.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time

from flask import Flask, jsonify, request, send_from_directory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "bin")
FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))

K_CUDA = (1, 4, 8, 10, 16, 32)

# Colab free tier has ~12.7 GB RAM: keep the host-side database allocation
# safely below that (n * d * 4 bytes).
MAX_ELEMENTS = 2_000_000_000
MAX_SWEEP = 6
RUN_TIMEOUT = 300
BUILD_TIMEOUT = 900

JOBS = threading.Lock()
LAST_BUILD = {"output": "", "when": None, "target": None}

app = Flask(__name__, static_folder=None)


# ---------------------------------------------------------------- engines --
def engine_bin(engine):
    return os.path.join(BIN, {"cuda": "knn_cuda", "omp": "knn_omp"}.get(engine, ""))


def engine_available(engine):
    p = engine_bin(engine)
    return bool(p) and os.path.isfile(p) and os.access(p, os.X_OK)


def gpu_name():
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        names = [l.strip() for l in (p.stdout or "").splitlines() if l.strip()]
        return ", ".join(dict.fromkeys(names))  # dedupe, keep order
    except Exception:
        return None


def git_rev():
    try:
        p = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return (p.stdout or "").strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------- parsing --
RX = {
    "gpu": re.compile(r"\[gpu \] (.+)"),
    "fused": re.compile(r"\[time\] fused\s+([\d.]+) ms\s+\(([\d.]+) ms/query\)\s+"
                        r"([\d.]+) GB/s.*?([\d.]+) GFLOP/s"),
    "cpu": re.compile(r"\[time\] cpu\s+([\d.]+) s\s+\(([\d.]+) ms/query\)\s+"
                      r"([\d.eE+]+) pairs/s\s+([\d.]+) GB/s.*?([\d.]+) GFLOP/s"),
    "twopass": re.compile(r"\[time\] twopass\s+([\d.]+) ms.*fused is ([\d.]+)x faster"),
    "verify": re.compile(r"\[verify\].*->\s*(PASS|FAIL)"),
    "topk": re.compile(r"\[topk\] q=(\d+) r=(\d+) idx=(-?\d+) dist=([\d.eE+-]+)"),
}


def parse_output(out):
    res = {"device": None, "fused_ms": None, "ms_per_query": None, "gbps": None,
           "gflops": None, "twopass_ms": None, "speedup": None, "cpu_s": None,
           "verify": None, "topk": {}}
    for line in out.splitlines():
        m = RX["gpu"].search(line)
        if m:
            res["device"] = m.group(1)
            continue
        m = RX["fused"].search(line)
        if m:
            res.update(fused_ms=float(m.group(1)), ms_per_query=float(m.group(2)),
                       gbps=float(m.group(3)), gflops=float(m.group(4)))
            continue
        m = RX["cpu"].search(line)
        if m:
            res.update(cpu_s=float(m.group(1)), ms_per_query=float(m.group(2)),
                       gbps=float(m.group(4)), gflops=float(m.group(5)))
            continue
        m = RX["twopass"].search(line)
        if m:
            res.update(twopass_ms=float(m.group(1)), speedup=float(m.group(2)))
            continue
        m = RX["verify"].search(line)
        if m:
            res["verify"] = m.group(1)
            continue
        m = RX["topk"].search(line)
        if m:
            qq, rank, idx, dist = int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))
            res["topk"].setdefault(qq, []).append({"rank": rank, "idx": idx, "dist": dist})
    return res


# ------------------------------------------------------------------ runner --
def clamp_int(payload, key, lo, hi, default):
    try:
        v = int(payload.get(key, default))
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be an integer")
    if v < lo or v > hi:
        raise ValueError(f"{key} must be in [{lo}, {hi}]")
    return v


def run_params(payload, engine):
    p = {}
    p["n"] = clamp_int(payload, "n", 1_000, MAX_ELEMENTS, 1_000_000)
    p["d"] = clamp_int(payload, "d", 8, 1024, 128)
    p["q"] = clamp_int(payload, "q", 1, 1024, 32)
    p["k"] = clamp_int(payload, "k", 1, 1024, 10)
    p["seed"] = clamp_int(payload, "seed", 0, 2**31 - 1, 42)
    p["verify"] = clamp_int(payload, "verify", 0, 64, 4)
    p["dump"] = clamp_int(payload, "dump", 0, 16, 4)
    p["twopass"] = bool(payload.get("twopass", True))
    if p["n"] * p["d"] > MAX_ELEMENTS:
        raise ValueError(f"n*d must be <= {MAX_ELEMENTS} (GPU-box RAM); got {p['n'] * p['d']}")
    if engine == "cuda" and p["k"] not in K_CUDA:
        raise ValueError(f"cuda engine supports k in {list(K_CUDA)}")
    return p


def run_binary(engine, p, timeout=RUN_TIMEOUT):
    cmd = [engine_bin(engine), "--n", str(p["n"]), "--d", str(p["d"]), "--q", str(p["q"]),
           "--k", str(p["k"]), "--seed", str(p["seed"]), "--verify", str(p["verify"]),
           "--dump", str(p["dump"])]
    if engine == "cuda":
        cmd += ["--twopass", "1" if p["twopass"] else "0"]
    else:
        cmd += ["--threads", "0"]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    out = (proc.stdout or "") + (proc.stderr or "")
    res = parse_output(out)
    res.update(ok=(proc.returncode == 0), rc=proc.returncode, engine=engine,
               wall_s=round(time.time() - t0, 3), cmd=" ".join(cmd), stdout=out)
    return res


# ------------------------------------------------------------------ routes --
@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/favicon.ico")
def favicon():
    return "", 204


@app.get("/api/health")
def health():
    return jsonify(ok=True, service="fused-knn-frontend", rev=git_rev())


@app.get("/api/state")
def state():
    return jsonify(
        root=ROOT if os.path.isdir(BIN) else "not built yet",
        git=git_rev(),
        python=sys.version.split()[0],
        gpu=gpu_name(),
        engines={e: {"available": engine_available(e), "binary": engine_bin(e)}
                 for e in ("cuda", "omp")},
        limits={"max_elements": MAX_ELEMENTS, "max_sweep": MAX_SWEEP,
                "run_timeout_s": RUN_TIMEOUT},
        last_build=LAST_BUILD,
    )


@app.post("/api/search")
def search():
    payload = request.get_json(force=True, silent=True) or {}
    engine = payload.get("engine", "cuda")
    if engine not in ("cuda", "omp"):
        return jsonify(ok=False, error="engine must be cuda or omp"), 400
    if not engine_available(engine):
        return jsonify(ok=False, error=f"{engine} binary not built (run Build first)"), 409
    try:
        p = run_params(payload, engine)
    except ValueError as e:
        return jsonify(ok=False, error=str(e)), 400
    if not JOBS.acquire(blocking=False):
        return jsonify(ok=False, error="a job is already running; try again shortly"), 409
    try:
        res = run_binary(engine, p)
    except subprocess.TimeoutExpired:
        return jsonify(ok=False, error=f"run exceeded {RUN_TIMEOUT}s"), 504
    finally:
        JOBS.release()
    return jsonify(res)


@app.post("/api/benchmark")
def benchmark():
    payload = request.get_json(force=True, silent=True) or {}
    engine = payload.get("engine", "cuda")
    if engine not in ("cuda", "omp"):
        return jsonify(ok=False, error="engine must be cuda or omp"), 400
    if not engine_available(engine):
        return jsonify(ok=False, error=f"{engine} binary not built (run Build first)"), 409
    sizes = payload.get("sizes")
    if not isinstance(sizes, list) or not sizes:
        return jsonify(ok=False, error="sizes must be a non-empty list"), 400
    if len(sizes) > MAX_SWEEP:
        sizes = sizes[:MAX_SWEEP]
    rows = []
    try:
        base = run_params({**payload, "sizes": None}, engine)
    except ValueError as e:
        return jsonify(ok=False, error=str(e)), 400
    if not JOBS.acquire(blocking=False):
        return jsonify(ok=False, error="a job is already running; try again shortly"), 409
    try:
        for n in sizes:
            try:
                p = dict(base)
                p["n"] = int(n)
                if p["n"] < 1_000 or p["n"] * p["d"] > MAX_ELEMENTS:
                    rows.append({"n": n, "error": "out of range"})
                    continue
                p["twopass"] = True
                r = run_binary(engine, p)
                rows.append({"n": p["n"], "fused_ms": r["fused_ms"], "cpu_s": r["cpu_s"],
                             "twopass_ms": r["twopass_ms"], "speedup": r["speedup"],
                             "gbps": r["gbps"], "gflops": r["gflops"],
                             "verify": r["verify"], "ok": r["ok"]})
            except subprocess.TimeoutExpired:
                rows.append({"n": n, "error": f"timed out after {RUN_TIMEOUT}s"})
    finally:
        JOBS.release()
    return jsonify(ok=True, engine=engine, rows=rows)


@app.post("/api/build")
def build():
    payload = request.get_json(force=True, silent=True) or {}
    target = payload.get("target", "gpu")
    if target not in ("gpu", "cpu", "all"):
        return jsonify(ok=False, error="target must be gpu, cpu or all"), 400
    if not JOBS.acquire(blocking=False):
        return jsonify(ok=False, error="a job is already running; try again shortly"), 409
    try:
        proc = subprocess.run(["make", target], cwd=ROOT, capture_output=True,
                              text=True, timeout=BUILD_TIMEOUT)
        out = (proc.stdout or "") + (proc.stderr or "")
        LAST_BUILD.update(output="\n".join(out.strip().splitlines()[-60:]),
                          when=time.strftime("%H:%M:%S"), target=target)
        return jsonify(ok=(proc.returncode == 0), rc=proc.returncode, target=target,
                       output=LAST_BUILD["output"], engines={
                           e: engine_available(e) for e in ("cuda", "omp")})
    except subprocess.TimeoutExpired:
        return jsonify(ok=False, error=f"build exceeded {BUILD_TIMEOUT}s"), 504
    except FileNotFoundError:
        return jsonify(ok=False, error="make not found on this host"), 500
    finally:
        JOBS.release()


def main():
    ap = argparse.ArgumentParser(description="fused-kNN web frontend")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    print(f"[frontend] repo root : {ROOT}", flush=True)
    print(f"[frontend] listening : http://{args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
