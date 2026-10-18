# Run the project on a free Colab GPU (Cloudflare tunnels)

Colab gives you a free GPU box (usually a T4) and — unlike raw SSH setups —
the clean way out to the internet is a **Cloudflare quick tunnel**: no
account, no domain, no credentials. Quick tunnels are HTTP(S)-only, so this
setup exposes *services* instead of sshd:

- a **web frontend** (`frontend/`) to run searches/sweeps and see the
  timings + top-k results, and
- **VS Code in the browser** (code-server) with a full terminal on the same
  machine, and/or an `sshx` shared terminal.

That replaces what the SSH tunnel did: edit, build (`make gpu`), run, and
watch results — all from a browser tab, on the GPU box.

> Prefer raw ssh from your laptop? `remote/TUNNEL_SETUP.md` keeps that route
> on Kaggle (bore) alive as an alternative.

---

## TL;DR

```
1. Colab:  Runtime -> Change runtime type -> GPU (T4)
2. paste remote/colab_bootstrap.py into one cell, run it  (~2 min first time)
3. open the URLs it prints:
     web frontend     https://<random>.trycloudflare.com
     browser VS Code  https://<random>.trycloudflare.com   (+password)
     shared terminal  https://sshx.io/...
```

The cell clones the GitHub repo, installs mpich, compiles the CUDA/OpenMP/MPI
binaries, starts the services, and smoke-tests the toolchain.

---

## How it works

```
 your browser                                Colab GPU runtime
 +-------------+                            +-------------------------------+
 | web UI      |   https://xxx.trycloudflare|  cloudflared (2 instances)    |
 | VS Code     |<----------.com ------------|    :8080  frontend/server.py  |
 | sshx term   |                            |    :8443  code-server         |
 +-------------+                            |  /content/pcap  (git clone)   |
                                            |  nvcc, mpicxx, mpirun         |
                                            +-------------------------------+
```

`cloudflared tunnel --url http://127.0.0.1:PORT` dials out to Cloudflare's
edge and gets a random `*.trycloudflare.com` hostname that proxies back in.
Nothing listens on a public port; every connection arrives through the tunnel
process, and both the tunnel and the services die with the Colab runtime.

Three services, one per purpose:

| Service | What you do with it | Port |
|---|---|---|
| **Web frontend** | run an exact kNN search, watch fused vs two-pass timings, sweep database sizes, build binaries | 8080 |
| **code-server** | VS Code in the browser: edit files, open a terminal, `git pull`, `make gpu`, run `mpirun` | 8443 |
| **sshx** | a shared browser terminal — paste the link to teammates, everyone sees the same shell | out |

---

## 1. Colab: start a session

1. <https://colab.research.google.com> → **New notebook**.
2. **Runtime → Change runtime type → T4 GPU** (free tier).
3. Paste all of `remote/colab_bootstrap.py` into one cell, run it (▶).
   First run: installs + downloads + build ≈ 2–3 min.

The cell ends with:

```
========================================================================
 COLAB GPU BOX READY (Cloudflare quick tunnels)
========================================================================
 web frontend   : https://quiet-river-1234.trycloudflare.com
 browser VS Code: https://lucky-forest-5678.trycloudflare.com   (password: ...)
 shared terminal: https://sshx.io/#abc123
 work dir       : /content/pcap
```

**Keep the tab open while you work.** Idle Colab runtimes get recycled
(GPU interactive sessions disconnect after ~90 min of inactivity; 12 h max
regardless), and everything above dies with the runtime.

**URLs are new every run.** Re-running the cell (or a fresh runtime) restarts
the tunnels with new random hostnames — re-share them with the team.

---

## 2. The web frontend

Open the frontend URL and you get one page with four panels:

- **Search** — pick n / d / q / k / seed, engine (`knn_cuda` or the CPU
  `knn_omp` fallback), run: shows the fused kernel time, db throughput,
  GFLOP/s, the two-pass time + speedup factor, a PASS/FAIL exactness check
  against the double-precision CPU reference, and the actual top-k table
  (indices + distances) for the first few queries.
- **Sweep** — comma-separated database sizes; runs each with fused and
  two-pass and renders the comparison chart (log-scale ms) + table. This is
  the fused-vs-two-pass story of the report in one click.
- **Build** — `make gpu` / `make cpu` on the box, output shown live-ish.
  Handy after editing files in code-server.
- **How it works** — the fused kernel vs two-pass explanation for anyone
  looking over your shoulder.

Notes

- The page *drives the real binaries* (`bin/knn_cuda`, `bin/knn_omp`) — it is
  not a mock; every number comes from an actual GPU run with verification.
- Runs are capped (n·d ≤ 2e9 ≈ 8 GB host RAM; ≤ 6 sweep points) so nobody
  can accidentally OOM the runtime, and one job runs at a time.
- The URL is unlisted but public — anyone who has it can run jobs. Treat it
  like a draft demo, don't post it anywhere permanent, and re-run the cell to
  rotate the URL when you're done.

## 3. Browser VS Code (code-server)

Open the code-server URL, log in with the printed password. You get:

- the repo at `/content/pcap` in an editor (CUDA sources included),
- a terminal where `make gpu`, `./bin/knn_cuda ...`, and
  `mpirun -np 2 ./bin/knn_mpi_cuda ...` work exactly like on any box,
- everything running on the GPU machine — your laptop is just the browser.

Typical loop: edit in browser VS Code → run `make gpu` in its terminal (or
hit Build in the web UI) → look at numbers in the web UI → commit/push from
your laptop later (see below).

## 4. Editing from the laptop + GitHub flow

The repo on Colab is a plain `git clone` of
`github.com/JaiiMukundh/fused-knn-search`:

```bash
# laptop: edit + push as usual
git add -A && git commit -m "..." && git push

# Colab: get the changes (either re-run the bootstrap cell, which pulls,
# or in the code-server terminal:)
cd /content/pcap && git pull --ff-only && make gpu
```

Pushing *from* Colab needs credentials; the simple flow is: laptop pushes,
Colab pulls.

## 5. Team sharing

- **Watch mode:** send the web frontend URL — anyone can run searches/sweeps
  and see results in their browser.
- **Pair-debugging:** send the sshx link — a shared, end-to-end-encrypted
  terminal on the GPU box.
- **Co-editing:** code-server password + URL; note all editors act as the
  same user, and the password is per-run.
- Everyone sees the *same* numbers because datasets are deterministic
  (`--seed`), and every run verifies against the CPU reference.

---

## Facts to keep in mind

- **Free-tier limits:** T4 GPU, ~12.7 GB RAM, ~2 vCPUs, 12 h max session,
  idle disconnect ~90 min (keep the tab open / interact). Colab may also
  reclaim runtimes when busy — the cell is cheap to re-run (~2 min), and
  `/content` is wiped when the runtime goes.
- **Quick tunnels:** random hostname per run, no account, no SLA — if a URL
  404s or hangs, check the cell output, re-run it, use the new URL. QUIC/UDP
  can be blocked on some networks; the bootstrap forces `--protocol http2`
  so the tunnel only needs outbound TCP 443.
- **apt/pip** run as root on Colab and are fine; mpich gives you the same
  `mpicxx`/`mpirun` interface as on Kaggle.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `no GPU visible` in bootstrap | Runtime → Change runtime type → GPU, re-run |
| `make gpu build FAILED` | open code-server terminal, `cd /content/pcap && make gpu` and read the error (or use the web UI Build panel) |
| Frontend URL shows 502/nothing | Tunnel was just created — the cell already waits for it; if it persists, re-run the cell (new URL) |
| `cloudflared` exits / URL never appears | transient edge problem: re-run the cell; check output logs in `/tmp/pcap/cloudflared-*.log` in the code-server terminal |
| Everything died after ~90 min | idle recycle — re-run the cell |
| `Expected 200 from /api/health` | frontend crashed: `/tmp/pcap/frontend.log` (usually flask missing → re-run cell, it pip-installs) |
| Tunnel works but slow first load | Cloudflare edge warm-up + no CDN caching on quick tunnels; it settles |

## Alternatives (when you outgrow quick tunnels)

| Want | Do |
|---|---|
| Stable URL (no changing hostnames) | Cloudflare **named** tunnel + your own domain (free plan; `cloudflared tunnel create/route dns`), same bootstrap otherwise |
| Raw ssh from the laptop | Kaggle + bore: `remote/TUNNEL_SETUP.md` (kept working) |
| Just a terminal | `MODE="sshx"` only, or `MODE="web"` + `MODE="sshx"` |

## Files in `remote/` (this route)

| File | Role |
|---|---|
| `colab_bootstrap.py` | the cell you paste into Colab (source of truth) |
| `COLAB_SETUP.md` | this guide |
| `TUNNEL_SETUP.md`, `kaggle_bootstrap.*`, `kremote.sh`, `kaggle.env.example` | the alternative Kaggle + ssh-over-bore route |

## Files in `frontend/`

| File | Role |
|---|---|
| `server.py` | Flask API: `/api/search`, `/api/benchmark`, `/api/build` — runs the compiled binaries and parses their output |
| `index.html` | the single-page UI (no build step, Chart.js from CDN) |
