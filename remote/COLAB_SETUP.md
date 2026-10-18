# Run the project on a free Colab GPU (ssh tunnel + web frontend)

Colab gives you a free GPU box (usually a T4). `remote/colab_bootstrap.py`
turns it into a machine you can actually use:

- a **real ssh tunnel** — `ssh -p <port> root@bore.pub` from your laptop,
  with rsync and VS Code Remote-SSH (bare `cloudflared` quick tunnels are
  HTTP-only, so sshd is exposed through the public **bore** TCP relay instead), and
- **HTTP services through Cloudflare quick tunnels** (no account, no domain):
  the **web frontend** (`frontend/`) that runs searches/sweeps on the GPU and
  shows fused-vs-two-pass timings, **VS Code in the browser** (code-server),
  and/or an **sshx** shared terminal.

---

## TL;DR

```
1. Colab:  Runtime -> Change runtime type -> GPU (T4)
2. paste ONE cell and run it (first run ~2-3 min):

     import urllib.request as u; exec(u.urlopen(
       'https://raw.githubusercontent.com/JaiiMukundh/fused-knn-search/main/remote/colab_bootstrap.py').read())

   (equivalent: paste the whole of remote/colab_bootstrap.py into the cell)
3. within the first minute it prints the connection details, e.g.:
       ssh            : ssh -p 54321 root@bore.pub -i ~/.ssh/colab_pcap
       web frontend   : https://quiet-river-1234.trycloudflare.com
       shared terminal: https://sshx.io/#abc123
   ...then keeps installing/building until "COLAB GPU BOX READY".
4. from your laptop:
       ssh -p 54321 root@bore.pub -i ~/.ssh/colab_pcap
       cd /content/pcap && make check-gpu
```

The cell installs what it needs (mpich, openssh-server if missing, FAISS for the
ANN comparison), clones `github.com/JaiiMukundh/fused-knn-search` into
`/content/pcap`, compiles the CUDA/OpenMP/MPI binaries (`make gpu` + `make cpu`),
runs one brute-force-vs-ANN comparison, starts the tunnels and services, and
smoke-tests the toolchain. **The ssh tunnel comes up first** so you can connect
while the rest finishes.

---

## How it works

```
 your laptop                                      Colab GPU runtime
 +----------------+                              +-------------------------------+
 | ssh / rsync    |    bore.pub:<random port>    |  sshd :2222  (key-only login) |
 | VS Code (SSH)  |<-------- TCP tunnel ---------|      ^                        |
 |                |                              |      |  bore client           |
 | web browser:   |                              |      v                        |
 |   web frontend |    https://xxx.trycloudflare |  cloudflared -> :<free port>  |
 |   sshx         |<------- HTTPS tunnel --------|    frontend/server.py         |
 +----------------+                              |  /content/pcap   (git clone)  |
                                                 |  nvcc, mpicxx, mpirun         |
                                                 +-------------------------------+
```

- **ssh:** the bootstrap starts `sshd` and runs `bore` (`bore local 2222
  --to bore.pub`), which asks the public relay for a port and forwards it to
  the VM. Login is key-only with the public key baked into the bootstrap
  (`AUTHORIZED_KEYS`); this laptop already has the matching private key at
  `~/.ssh/colab_pcap`.
- **HTTP:** `cloudflared tunnel --url http://127.0.0.1:PORT` dials out to
  Cloudflare's edge and returns a random `*.trycloudflare.com` hostname that
  proxies back in. Nothing listens on a public port; every connection arrives
  through the tunnel processes, and all of it dies with the Colab runtime.
  The frontend port is chosen dynamically (Colab itself occupies 8080).

| Service | What you do with it | Where |
|---|---|---|
| **ssh** | terminal, rsync, VS Code Remote-SSH, `mpirun` — the full box | `bore.pub:<random>` |
| **web frontend** | run an exact kNN search, fused vs two-pass timings, size sweeps, build binaries | dynamic local port via cloudflared |
| **code-server** | VS Code in the browser (password), `MODE="codeserver"` | dynamic local port via cloudflared |
| **sshx** | shareable browser terminal — send the link to teammates | outbound |

---

## 1. Colab: start a session

1. <https://colab.research.google.com> → **New notebook**.
2. **Runtime → Change runtime type → T4 GPU** (free tier).
3. Put the bootstrap in one cell and run it (▶) — either paste the whole of
   `remote/colab_bootstrap.py`, or paste this loader (always fetches the
   latest pushed version):

   ```python
   import urllib.request as u; exec(u.urlopen(
     'https://raw.githubusercontent.com/JaiiMukundh/fused-knn-search/main/remote/colab_bootstrap.py').read())
   ```

Config knobs at the top of the file:

```python
MODE = "all"          # "all" = ssh + web + sshx; or "ssh", "ssh,web", "codeserver", ...
AUTHORIZED_KEYS = [   # already contains your laptop key (~/.ssh/colab_pcap)
    "ssh-ed25519 AAAA... colab-pcap",
]
SSH_TUNNEL = "auto"   # bore (default) or ngrok if you set NGROK_AUTHTOKEN
```

**Keep the tab open while you work.** Idle Colab runtimes get recycled (GPU
interactive sessions disconnect after ~90 min idle; 12 h max regardless), and
everything dies with the runtime. The ssh port and all URLs are **new every
run** — re-running the cell is how you get back in.

## 2. SSH from your laptop (recommended)

The cell prints an `SSH TUNNEL READY` block the moment the tunnel is up:

```bash
ssh -p 54321 root@bore.pub -i ~/.ssh/colab_pcap
```

That is the whole box: terminal, `mpirun`, rsync, scp. Recipes:

```bash
# build + verify CUDA/MPI on the GPU
ssh -p 54321 root@bore.pub -i ~/.ssh/colab_pcap \
    'cd /content/pcap && make check-gpu'

# run a real workload, then fetch the logs
ssh -p 54321 root@bore.pub -i ~/.ssh/colab_pcap \
    'cd /content/pcap && ./bin/knn_cuda --n 1000000 --d 128 --q 512 --k 10'
rsync -az -e 'ssh -p 54321 -i ~/.ssh/colab_pcap' \
    root@bore.pub:/content/pcap/out/ ./out/
```

Or add the `Host gpu-pcap` block from the cell output to `~/.ssh/config` and
use plain `ssh gpu-pcap` / VS Code Remote-SSH.

Notes

- `ssh` expects the private key `~/.ssh/colab_pcap` (already generated on this
  laptop and whitelisted in the bootstrap). If you run the cell with an empty
  `AUTHORIZED_KEYS`, it prints a one-time root password instead.
- The relay port is random per run; if ssh refuses after a runtime restart,
  re-run the cell and use the new port it prints.
- `bore.pub` is a public relay: the port is obscure but reachable by anyone —
  login still requires your key. Treat the tunnel like a draft demo.

## 3. The web frontend

Open the frontend URL and you get one page with five panels:

- **Search** — pick n / d / q / k / seed, engine (GPU `knn --device cuda` or the
  CPU `--device cpu` fallback), run: shows the fused kernel time, db throughput,
  GFLOP/s, the two-pass time + speedup factor, a PASS/FAIL exactness check
  against the double-precision CPU reference, and the actual top-k table
  (indices + distances) for the first few queries.
- **Sweep** — comma-separated database sizes; runs each with fused and
  two-pass and renders the comparison chart (log-scale ms) + table.
- **Compare** — Objective 3: brute force (our exact kernel) vs ANN (FAISS).
  It exports the shared dataset with `bin/gendata`, then sweeps IVF-PQ
  `nprobe` and HNSW `efSearch`, showing recall@k, ms/query, speedup vs exact
  and the recall-vs-latency curve.
- **Build** — `make gpu` / `make cpu` on the box, output shown live-ish.
- **How it works** — fused vs two-pass explanation for anyone looking over
  your shoulder.

The page *drives the real binaries* — every number comes from an actual run
with verification. Runs are capped (n·d ≤ 2e9 ≈ 8 GB host RAM; ≤ 6 sweep
points) and one job runs at a time.

Engine availability is probed, not assumed: the badge turns red with a reason
(`cuda unavailable: <driver error>`) unless the CUDA path has actually executed
a tiny search on this box. If a run fails, the panel shows the error and the
command instead of an empty table.

FAISS itself is installed by the bootstrap cell (`faiss-gpu-cu12`, falling back
to `faiss-gpu` then `faiss-cpu` — the badge says which). If only the CPU wheel
installs, recall@k is unchanged and the ANN latencies become CPU numbers; the
harness prints which device it used.

The bootstrap also runs one comparison automatically and prints the table, and
from a shell you can do the same:

```bash
cd /content/pcap && make compare        # writes data/faiss.json
```

## 4. Browser VS Code (code-server)

Run the cell with `MODE="codeserver"` (or `"ssh,codeserver"`): open the
printed URL, log in with the password — editor + terminal on the GPU box.
Typical loop: edit in the browser → `make gpu` in its terminal → look at
numbers in the web frontend → commit/push from your laptop later.

## 5. Editing from the laptop + GitHub flow

The repo on Colab is a plain `git clone` of
`github.com/JaiiMukundh/fused-knn-search`:

```bash
# laptop: edit + push as usual
git add -A && git commit -m "..." && git push

# Colab: get the changes — re-run the bootstrap cell (it pulls), or over ssh:
ssh -p 54321 root@bore.pub -i ~/.ssh/colab_pcap \
    'cd /content/pcap && git pull --ff-only && make gpu'
```

Pushing *from* Colab needs credentials; the simple flow is: laptop pushes,
Colab pulls.

## 6. Team sharing

- **Watch mode:** send the web frontend URL — anyone can run searches/sweeps.
- **Real terminal:** each teammate can ssh with their own key (add it to
  `AUTHORIZED_KEYS`) or use the sshx link (shared, end-to-end encrypted).
- Everyone sees the *same* numbers because datasets are deterministic
  (`--seed`) and every run verifies against the CPU reference.

---

## Facts to keep in mind

- **Free-tier limits:** T4 GPU, ~12.7 GB RAM, ~2 vCPUs, 12 h max session,
  idle disconnect ~90 min. Colab may reclaim busy runtimes — re-running the
  cell costs ~2-3 min and `/content` is wiped.
- **ssh side:** bore.pub gives a random port per run; the bootstrap forces
  no password auth once keys are present. If sshd or bore fails to start,
  the cell says so with a log tail.
- **HTTP side:** quick tunnels are HTTP-only (hence bore for ssh), random
  hostname per run, no SLA. QUIC/UDP is blocked on some networks; the
  bootstrap forces `--protocol http2` so cloudflared only needs outbound
  TCP 443.
- **Ports:** the frontend and code-server pick a free local port at startup
  (Colab always occupies 8080, so hardcoding a port breaks).
- **Local compile check (no GPU):** with any CUDA toolchain on the laptop
  (e.g. conda/micromamba `cuda-nvcc` + `cuda-cudart-dev`), you can compile
  both CUDA binaries without a GPU:
  `make gpu NVCC=<env>/bin/nvcc CUDA_HOME=<env> CUDA_ARCH_FLAGS='-arch=sm_75'`
  — catches API breakage before you spend GPU quota (this is how the CUDA-13
  `memoryClockRate` removal was found).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `no GPU visible` in bootstrap | Runtime → Change runtime type → GPU, re-run |
| `make gpu build FAILED` | ssh in (or code-server terminal), `cd /content/pcap && make gpu`, read the error |
| ssh: `Connection refused` | wrong/stale port — re-run the cell and use the new port |
| ssh: `Permission denied` | key mismatch: use `-i ~/.ssh/colab_pcap`, or re-run with your public key in `AUTHORIZED_KEYS` |
| bore port never appears | transient relay problem — re-run the cell; check `/tmp/pcap/bore.log` |
| Frontend URL shows 502/nothing | edge warm-up or tunnel restart — wait a minute, else re-run the cell (new URL) |
| Everything died after ~90 min | idle recycle — re-run the cell |
| Tunnel up but first page load slow | Cloudflare edge warm-up on quick tunnels; it settles |

## Alternatives

| Want | Do |
|---|---|
| Stable URL / own domain | Cloudflare **named** tunnel + domain (free plan), same bootstrap otherwise |
| Just a terminal | `MODE="ssh"` or `MODE="sshx"` |
| ngrok instead of bore | set `NGROK_AUTHTOKEN`, `SSH_TUNNEL="ngrok"` (free ngrok now wants a payment method for TCP) |

## Files in `remote/` (this route)

| File | Role |
|---|---|
| `colab_bootstrap.py` | the cell you paste into Colab (source of truth) |
| `COLAB_SETUP.md` | this guide |

## Files in `frontend/`

| File | Role |
|---|---|
| `server.py` | Flask API: `/api/search`, `/api/benchmark`, `/api/build` — runs the compiled binaries and parses their output |
| `index.html` | the single-page UI (no build step, Chart.js from CDN) |
