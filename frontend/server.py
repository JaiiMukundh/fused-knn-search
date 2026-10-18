#!/usr/bin/env python3
"""Web frontend + JSON API for the fused-kNN project.

Runs on the GPU box (Colab / any Linux host). Serves one HTML page and a small
API that shells out to the compiled binaries in <repo>/bin and parses their
[run] / [time] / [verify] / [topk] output lines.

    python3 frontend/server.py --port 8080

Endpoints
    GET  /                  the single-page UI
    GET  /api/state         GPU / engines / limits / git info
    GET  /api/health        liveness probe (used by the tunnel test)
    POST /api/search        run one exact kNN search, return timings + top-k
    POST /api/benchmark     run a size sweep (fused vs two-pass), return rows
    POST /api/compare       brute force vs ANN: exact search against FAISS
    POST /api/build         run `make gpu` (or cpu), return the log tail

Availability is *probed*, not guessed. The old version reported a CUDA engine as
"ready" whenever bin/knn_cuda was merely present, so on a laptop with a copy of
the binary every run died silently and the UI rendered a grid of dashes. Here
engine_available() actually executes the binary on a tiny workload and caches
the verdict, so "cuda: unavailable" means it will really fail, and the UI can
say why.

Only one heavy job runs at a time; concurrent requests get HTTP 409.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time

from flask import Flask, jsonify, request, send_from_directory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "bin")
FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
BENCH = os.path.join(ROOT, "frontend", "..", "bench")
FAISS_HARNESS = os.path.abspath(os.path.join(ROOT, "bench", "faiss_compare.py"))

K_CUDA = (1, 4, 8, 10, 16, 32)

# Colab free tier has ~12.7 GB RAM: keep the host-side database allocation
# safely below that (n * d * 4 bytes).
MAX_ELEMENTS = 2_000_000_000
MAX_SWEEP = 6
RUN_TIMEOUT = 600
COMPARE_TIMEOUT = 3600
BUILD_TIMEOUT = 900

JOBS = threading.Lock()
LAST_BUILD = {"output": "", "when": None, "target": None}

app = Flask(__name__, static_folder=None)


# ---------------------------------------------------------------- engines --
def _is_exe(path):
    return bool(path) and os.path.isfile(path) and os.access(path, os.X_OK)


def unified_bin():
    """The MPI + OpenMP + CUDA engine, if this box built it."""
    p = os.path.join(BIN, "knn")
    return p if _is_exe(p) else None


def cpu_bin():
    """Fallback exact CPU baseline for boxes with neither MPI nor CUDA."""
    p = os.path.join(BIN, "knn_omp")
    return p if _is_exe(p) else None


def gendata_bin():
    p = os.path.join(BIN, "gendata")
    return p if _is_exe(p) else None


_PROBE = {"t": 0.0, "data": None}
_PROBE_TTL = 60.0


def cuda_probe(force=False):
    """Really run the CUDA path once on a tiny workload and cache the verdict.

    A present-but-unusable binary (no driver, or a driver too old for the CUDA
    runtime it was built against) is the exact failure that used to render as an
    empty results table. Executing it is the only honest check.
    """
    now = time.time()
    if not force and _PROBE["data"] and now - _PROBE["t"] < _PROBE_TTL:
        return _PROBE["data"]

    info = {"available": False, "name": gpu_name(), "reason": None}
    binary = unified_bin()
    if not binary:
        info["reason"] = "bin/knn not built (MPI+OpenMP+CUDA target; run Build)"
        _PROBE.update(t=now, data=info)
        return info
    try:
        p = subprocess.run([binary, "--device", "cuda", "--n", "2048", "--d", "8",
                            "--q", "1", "--k", "4", "--verify", "0"],
                           cwd=ROOT, capture_output=True, text=True, timeout=120)
        out = (p.stdout or "") + (p.stderr or "")
        if p.returncode == 0 and "[gpu ]" in out:
            info["available"] = True
            m = re.search(r"\[gpu \] (.+)", out)
            if m:
                info["name"] = m.group(1).strip()
        else:
            tail = "\n".join(out.strip().splitlines()[-4:])
            info["reason"] = tail or f"exited with rc={p.returncode}"
    except subprocess.TimeoutExpired:
        info["reason"] = "probe timed out"
    except Exception as e:  # pragma: no cover
        info["reason"] = f"probe failed: {e}"
    _PROBE.update(t=now, data=info)
    return info


def gpu_name():
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        names = [l.strip() for l in (p.stdout or "").splitlines() if l.strip()]
        return ", ".join(dict.fromkeys(names))  # dedupe, keep order
    except Exception:
        return None


def engine_status():
    """What can this box actually run, with a reason when it cannot."""
    gpu = cuda_probe()
    uni, cpu = unified_bin(), cpu_bin()
    return {
        "cuda": {
            "available": bool(uni) and gpu["available"],
            "binary": uni or "",
            "device": gpu["name"],
            "reason": gpu["reason"] if uni else "bin/knn not built",
        },
        "cpu": {
            "available": bool(uni) or bool(cpu),
            "binary": uni or cpu or "",
            "device": "host (OpenMP)",
            "reason": None if (uni or cpu) else "no CPU binary built",
        },
        "gendata": {"available": bool(gendata_bin()), "binary": gendata_bin() or "",
                    "reason": None if gendata_bin() else "bin/gendata not built"},
        "faiss": faiss_status(),
    }


def faiss_status():
    try:
        p = subprocess.run([sys.executable, "-c", "import faiss, sys; "
                            "print(faiss.__version__, getattr(faiss,'get_num_gpus',lambda:0)())"],
                           capture_output=True, text=True, timeout=60)
        if p.returncode == 0:
            ver = (p.stdout or "").strip().split()
            return {"available": True, "version": ver[0] if ver else "?",
                    "gpus": int(ver[1]) if len(ver) > 1 and ver[1].isdigit() else 0,
                    "reason": None}
    except Exception:
        pass
    return {"available": False, "version": None, "gpus": 0,
            "reason": "faiss not importable (pip install faiss-gpu)"}


def resolve_engine(engine):
    """-> (argv_prefix, label) or raise ValueError with a human reason."""
    st = engine_status()
    if engine == "cuda":
        if not st["cuda"]["available"]:
            raise ValueError(f"cuda engine unavailable: {st['cuda']['reason']}")
        return [unified_bin(), "--device", "cuda"], "cuda"
    if engine in ("cpu", "omp"):
        if not st["cpu"]["available"]:
            raise ValueError(f"cpu engine unavailable: {st['cpu']['reason']}")
        if unified_bin():
            return [unified_bin(), "--device", "cpu"], "cpu"
        return [cpu_bin()], "cpu"
    raise ValueError("engine must be cuda or cpu")


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
    prefix, label = resolve_engine(engine)
    cmd = prefix + ["--n", str(p["n"]), "--d", str(p["d"]), "--q", str(p["q"]),
                    "--k", str(p["k"]), "--seed", str(p["seed"]), "--verify", str(p["verify"]),
                    "--dump", str(p["dump"])]
    if label == "cuda":
        cmd += ["--twopass", "1" if p["twopass"] else "0"]
    else:
        cmd += ["--threads", "0"]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    out = (proc.stdout or "") + (proc.stderr or "")
    res = parse_output(out)
    res.update(ok=(proc.returncode == 0), rc=proc.returncode, engine=label,
               wall_s=round(time.time() - t0, 3), cmd=" ".join(cmd), stdout=out)
    return res


def ensure_dataset(n, d, q, k, seed, force=False):
    """Make data/ match the requested parameters; run bin/gendata if needed."""
    gd = gendata_bin()
    if not gd:
        raise ValueError("bin/gendata not built -- run Build (make cpu) first")
    meta_path = os.path.join(DATA_DIR, "meta.json")
    want = {"n": n, "d": d, "q": q, "k": k, "seed": seed}
    if not force and os.path.isfile(meta_path):
        try:
            with open(meta_path) as f:
                got = json.load(f)
            if all(got.get(key) == val for key, val in want.items()):
                return DATA_DIR, False
        except Exception:
            pass
    os.makedirs(DATA_DIR, exist_ok=True)
    cmd = [gd, "--n", str(n), "--d", str(d), "--q", str(q), "--k", str(k),
           "--seed", str(seed), "--out", DATA_DIR]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=RUN_TIMEOUT)
    if proc.returncode != 0:
        raise ValueError("gendata failed: " + "\n".join(
            ((proc.stdout or "") + (proc.stderr or "")).strip().splitlines()[-6:]))
    return DATA_DIR, True


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


def git_rev():
    try:
        p = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return (p.stdout or "").strip() or None
    except Exception:
        return None


@app.get("/api/state")
def state():
    return jsonify(
        root=ROOT if os.path.isdir(BIN) else "not built yet",
        git=git_rev(),
        python=sys.version.split()[0],
        gpu=gpu_name(),
        engines=engine_status(),
        limits={"max_elements": MAX_ELEMENTS, "max_sweep": MAX_SWEEP,
                "run_timeout_s": RUN_TIMEOUT, "compare_timeout_s": COMPARE_TIMEOUT},
        last_build=LAST_BUILD,
    )


@app.post("/api/search")
def search():
    payload = request.get_json(force=True, silent=True) or {}
    engine = payload.get("engine", "cuda")
    if engine not in ("cuda", "cpu", "omp"):
        return jsonify(ok=False, error="engine must be cuda or cpu"), 400
    if not JOBS.acquire(blocking=False):
        return jsonify(ok=False, error="a job is already running; try again shortly"), 409
    try:
        try:
            p = run_params(payload, "cuda" if engine == "cuda" else "cpu")
            res = run_binary(engine, p)
        except ValueError as e:
            return jsonify(ok=False, error=str(e)), 400
        except subprocess.TimeoutExpired:
            return jsonify(ok=False, error=f"run exceeded {RUN_TIMEOUT}s"), 504
    finally:
        JOBS.release()
    return jsonify(res)


@app.post("/api/benchmark")
def benchmark():
    payload = request.get_json(force=True, silent=True) or {}
    engine = payload.get("engine", "cuda")
    if engine not in ("cuda", "cpu", "omp"):
        return jsonify(ok=False, error="engine must be cuda or cpu"), 400
    sizes = payload.get("sizes")
    if not isinstance(sizes, list) or not sizes:
        return jsonify(ok=False, error="sizes must be a non-empty list"), 400
    if len(sizes) > MAX_SWEEP:
        sizes = sizes[:MAX_SWEEP]
    try:
        base = run_params({**payload, "sizes": None}, "cuda" if engine == "cuda" else "cpu")
    except ValueError as e:
        return jsonify(ok=False, error=str(e)), 400
    if not JOBS.acquire(blocking=False):
        return jsonify(ok=False, error="a job is already running; try again shortly"), 409
    rows = []
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
                             "verify": r["verify"], "ok": r["ok"],
                             "error": None if r["ok"] else _first_error(r["stdout"])})
            except subprocess.TimeoutExpired:
                rows.append({"n": n, "error": f"timed out after {RUN_TIMEOUT}s"})
            except ValueError as e:
                rows.append({"n": n, "error": str(e)})
    finally:
        JOBS.release()
    return jsonify(ok=True, engine=engine, rows=rows)


def _first_error(out):
    for line in (out or "").splitlines():
        if "error" in line.lower() or "failed" in line.lower():
            return line.strip()[:300]
    tail = (out or "").strip().splitlines()
    return tail[-1].strip()[:300] if tail else "run failed"


def _csv(payload, key, default):
    v = payload.get(key, default)
    if isinstance(v, str):
        return [int(x) for x in v.split(",") if x.strip()]
    if isinstance(v, list) and v:
        return [int(x) for x in v]
    return [int(x) for x in str(default).split(",")]


@app.post("/api/compare")
def compare():
    """Objective 3: exact brute force vs the established approximate index.

    Exports the shared dataset (bin/gendata), runs our own exact engine for
    reference, then benchmarks FAISS IVF-PQ / HNSW on the identical vectors and
    returns recall@k against the exact ground truth.
    """
    payload = request.get_json(force=True, silent=True) or {}
    if not faiss_status()["available"]:
        return jsonify(ok=False, error=faiss_status()["reason"]), 409
    if not os.path.isfile(FAISS_HARNESS):
        return jsonify(ok=False, error=f"missing {FAISS_HARNESS}"), 500
    try:
        n = clamp_int(payload, "n", 10_000, MAX_ELEMENTS, 1_000_000)
        d = clamp_int(payload, "d", 8, 1024, 128)
        q = clamp_int(payload, "q", 1, 2048, 1000)
        k = clamp_int(payload, "k", 1, 128, 10)
        seed = clamp_int(payload, "seed", 0, 2**31 - 1, 42)
    except ValueError as e:
        return jsonify(ok=False, error=str(e)), 400
    if n * d > MAX_ELEMENTS:
        return jsonify(ok=False, error=f"n*d must be <= {MAX_ELEMENTS}; got {n * d}"), 400

    probes = _csv(payload, "nprobe", [1, 4, 16, 64])
    efs = _csv(payload, "efsearch", [16, 64, 256])
    if not JOBS.acquire(blocking=False):
        return jsonify(ok=False, error="a job is already running; try again shortly"), 409
    try:
        try:
            data_dir, exported = ensure_dataset(n, d, q, k, seed,
                                                force=bool(payload.get("refresh_data")))
        except ValueError as e:
            return jsonify(ok=False, error=str(e)), 400

        fd, json_path = tempfile.mkstemp(suffix=".json", prefix="faiss-", dir="/tmp")
        os.close(fd)
        cmd = [sys.executable, FAISS_HARNESS, "--data", data_dir, "--k", str(k),
               "--nprobe", ",".join(map(str, probes)),
               "--efsearch", ",".join(map(str, efs)),
               "--repeats", str(clamp_int(payload, "repeats", 1, 10, 3)),
               "--json", json_path]
        if payload.get("force_cpu"):
            cmd.append("--cpu")
        if payload.get("no_hnsw"):
            cmd.append("--no-hnsw")
        t0 = time.time()
        fd, log_path = tempfile.mkstemp(suffix=".log", prefix="faiss-", dir="/tmp")
        os.close(fd)
        try:
            with open(log_path, "w") as logf:
                proc = subprocess.run(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT,
                                      text=True, timeout=COMPARE_TIMEOUT)
            with open(log_path) as f:
                log = f.read()
            if proc.returncode != 0:
                return jsonify(ok=False, error="faiss_compare failed: " +
                               "\n".join(log.strip().splitlines()[-6:])), 500
            with open(json_path) as f:
                result = json.load(f)
        finally:
            for pth in (json_path, log_path):
                try:
                    os.unlink(pth)
                except OSError:
                    pass

        result["exported_data"] = exported
        result["data_dir"] = data_dir
        result["wall_s"] = round(time.time() - t0, 2)
        result["log"] = log

        # our own exact engine, on the same vectors, for the honest head-to-head
        ours = None
        if payload.get("run_ours", True):
            try:
                p = {"n": n, "d": d, "q": min(q, 256), "k": k if k in K_CUDA else 10,
                     "seed": seed, "verify": 2, "dump": 0, "twopass": False}
                ours = run_binary("cuda", p, timeout=RUN_TIMEOUT) if \
                    engine_status()["cuda"]["available"] else None
            except Exception as e:
                ours = {"ok": False, "error": str(e)}
        result["ours"] = ours
        return jsonify(result)
    except subprocess.TimeoutExpired:
        return jsonify(ok=False, error=f"comparison exceeded {COMPARE_TIMEOUT}s"), 504
    finally:
        JOBS.release()


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
        cuda_probe(force=True)   # the binaries just changed: re-probe
        return jsonify(ok=(proc.returncode == 0), rc=proc.returncode, target=target,
                       output=LAST_BUILD["output"], engines=engine_status())
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
