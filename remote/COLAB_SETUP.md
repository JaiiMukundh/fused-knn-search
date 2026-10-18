# Run the project on a free Colab GPU (ssh tunnel + web frontend)

Colab gives you a free GPU box (usually a T4). `remote/colab_bootstrap.py`
turns it into a machine you can actually use:

- a **real ssh tunnel** — `ssh -p <port> root@bore.pub` from your laptop,
  with rsync and VS Code Remote-SSH, same workflow as `kremote.sh` has on
  Kaggle (bare `cloudflared` quick tunnels are HTTP-only, so sshd is exposed
  through the public **bore** TCP relay instead), and
- **HTTP services through Cloudflare quick tunnels** (no account, no domain):
  the **web frontend** (`frontend/`) that runs searches/sweeps on the GPU and
  shows fused-vs-two-pass timings, **VS Code in the browser** (code-server),
  and/or an **sshx** shared terminal.

> Prefer Kaggle? `remote/TUNNEL_SETUP.md` is the Kaggle route with the same
> ssh-over-bore workflow.

---

## TL;DR

```
1. Colab:  Runtime -> Change runtime type -> GPU (T4)
2. paste remote/colab_bootstrap.py into one cell, run it  (~2-3 min first time)
3. it prints, e.g.:
       ssh            : ssh -p 54321 root@bore.pub -i ~/.ssh/kaggle_pcap
       web frontend   : https://quiet-river-1234.trycloudflare.com
       shared terminal: https://sshx.io/#abc123
4. jump in:
       ssh -p 54321 root@bore.pub -i ~/.ssh/kaggle_pcap
       cd /content/pcap && make check-gpu
```

The cell clones the GitHub repo, installs mpich + openssh-server, compiles
the CUDA/OpenMP/MPI binaries (`make gpu` + `make cpu`), starts the tunnels
and services, and smoke-tests the toolchain.

---

## How it works

```
 your laptop                                      Colab GPU runtime
 +----------------+                              +-------------------------------+
 | ssh / rsync    |    bore.pub:<random port>    |  sshd :2222  (key-only login) |
 | VS Code (SSH)  |<-------- TCP tunnel ---------|      ^                        |
 |                |                              |      |  bore client           |
 | web browser:   |                              |      v                        |
 |   web frontend |    https://xxx.trycloudflare |  cloudflared -> :8080         |
 |   sshx         |<------- HTTPS tunnel --------|    frontend/server.py         |
 +----------------+                              |  /content/pcap   (git clone)  |
                                                 |  nvcc, mpicxx, mpirun         |
                                                 +-------------------------------+
```

- **ssh:** the bootstrap starts `sshd` and runs `bore` (`bore local 2222
  --to bore.pub`), which asks the public relay for a port and forwards it to
  the VM. Login is key-only with the public key baked into the bootstrap
  (`AUTHORIZED_KEYS`); your laptop already has the matching private key at
  `~/.ssh/kaggle_pcap`.
- **HTTP:** `cloudflared tunnel --url http://127.0.0.1:PORT` dials out to
  Cloudflare's edge and returns a random `*.trycloudflare.com` hostname that
  proxies back in. Nothing listens on a public port; every connection arrives
  through the tunnel processes, and all of it dies with the Colab runtime.

| Service | What you do with it | Where |
|---|---|---|
| **ssh** | terminal, rsync, VS Code Remote-SSH, `mpirun` — the full box | `bore.pub:<random>` |
| **web frontend** | run an exact kNN search, fused vs two-pass timings, size sweeps, build binaries | `:8080` via cloudflared |
| **code-server** | VS Code in the browser (password), `MODE="codeserver"` | `:8443` via cloudflared |
| **sshx** | shareable browser terminal — send the link to teammates | outbound |

---

## 1. Colab: start a session

1. <https://colab.research.google.com> → **New notebook**.
2. **Runtime → Change runtime type → T4 GPU** (free tier).
3. Paste all of `remote/colab_bootstrap.py` into one cell, run it (▶).

Config knobs at the top of the file:

```python
MODE = "all"          # "all" = ssh + web + sshx; or "ssh", "ssh,web", "codeserver", ...
AUTHORIZED_KEYS = [   # already contains your laptop key (~/.ssh/kaggle_pcap)
    "ssh-ed25519 AAAA... kaggle-pcap",
]
SSH_TUNNEL = "auto"   # bore (default) or ngrok if you set NGROK_AUTHTOKEN
```

**Keep the tab open while you work.** Idle Colab runtimes get recycled (GPU
interactive sessions disconnect after ~90 min idle; 12 h max regardless), and
everything dies with the runtime. The ssh port and all URLs are **new every
run** — re-running the cell is how you get back in.

## 2. SSH from your laptop (recommended)

Copy the block the cell printed into `remote/kaggle.env`:

```
KAGGLE_HOST=bore.pub
KAGGLE_PORT=54321
KAGGLE_USER=root
KAGGLE_KEY=~/.ssh/kaggle_pcap
KAGGLE_DIR=/content/pcap
```

Then drive the box with `remote/kremote.sh`:

```bash
./remote/kremote.sh ssh                  # interactive shell
./remote/kremote.sh run make check-gpu   # build + verify CUDA/MPI on the GPU
./remote/kremote.sh run ./bin/knn_cuda --n 1000000 --d 128 --q 512 --k 10
./remote/kremote.sh push                 # sync local edits (excl. bin/ out/ data/)
./remote/kremote.sh pull                 # fetch out/ back
```

Or add the `Host gpu-pcap` block from the cell output to `~/.ssh/config` and
use plain `ssh gpu-pcap` / VS Code Remote-SSH.

Notes

- `ssh` expects the private key `~/.ssh/kaggle_pcap` (already generated on
  this laptop and whitelisted in the bootstrap). If you run the cell with an
  empty `AUTHORIZED_KEYS`, it prints a one-time root password instead.
- The relay port is random per run; if ssh refuses after a runtime restart,
  re-run the cell and paste the new port into `remote/kaggle.env`.
- `bore.pub` is a public relay: the port is obscure but reachable by anyone —
  login still requires your key. Treat the tunnel like a draft demo.

## 3. The web frontend

Open the frontend URL and you get one page with four panels:

- **Search** — pick n / d / q / k / seed, engine (`knn_cuda` or the CPU
  `knn_omp` fallback), run: shows the fused kernel time, db throughput,
  GFLOP/s, the two-pass time + speedup factor, a PASS/FAIL exactness check
  against the double-precision CPU reference, and the actual top-k table
  (indices + distances) for the first few queries.
- **Sweep** — comma-separated database sizes; runs each with fused and
  two-pass and renders the comparison chart (log-scale ms) + table.
- **Build** — `make gpu` / `make cpu` on the box, output shown live-ish.
- **How it works** — fused vs two-pass explanation for anyone looking over
  your shoulder.

The page *drives the real binaries* — every number comes from an actual run
with verification. Runs are capped (n·d ≤ 2e9 ≈ 8 GB host RAM; ≤ 6 sweep
points) and one job runs at a time.

## 4. Browser VS Code (code-server)

Run the cell with `MODE="codeserver"` (or `"ssh,codeserver"`): open the
printing URL, log in with the password — editor + terminal on the GPU box.
Typical loop: edit in the browser → `make gpu` in its terminal → look at
numbers in the web frontend → commit/push from your laptop later.

## 5. Editing from the laptop + GitHub flow

The repo on Colab is a plain `git clone` of
`github.com/JaiiMukundh/fused-knn-search`:

```bash
# laptop: edit + push as usual
git add -A && git commit -m "..." && git push

# Colab: get the changes — re-run the bootstrap cell (it pulls), or over ssh:
ssh -p 54321 root@bore.pub -i ~/.ssh/kaggle_pcap \
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
| ssh: `Connection refused` | wrong/stale port — re-run the cell, update `remote/kaggle.env` |
| ssh: `Permission denied` | key mismatch: use `-i ~/.ssh/kaggle_pcap`, or re-run with your public key in `AUTHORIZED_KEYS` |
| bore port never appears | transient relay problem — re-run the cell; check `/tmp/pcap/bore.log` |
| Frontend URL shows 502/nothing | edge warm-up or tunnel restart — re-run the cell (new URL) |
| Everything died after ~90 min | idle recycle — re-run the cell |
| Tunnel up but first page load slow | Cloudflare edge warm-up on quick tunnels; it settles |

## Alternatives

| Want | Do |
|---|---|
| Stable URL / own domain | Cloudflare **named** tunnel + domain (free plan), same bootstrap otherwise |
| Kaggle instead of Colab | `remote/TUNNEL_SETUP.md` (same ssh-over-bore workflow) |
| Just a terminal | `MODE="ssh"` or `MODE="sshx"` |
| ngrok instead of bore | set `NGROK_AUTHTOKEN`, `SSH_TUNNEL="ngrok"` (free ngrok now wants a payment method for TCP) |

## Files in `remote/` (this route)

| File | Role |
|---|---|
| `colab_bootstrap.py` | the cell you paste into Colab (source of truth) |
| `COLAB_SETUP.md` | this guide |
| `kremote.sh`, `kaggle.env` | laptop driver + connection settings (work with the Colab tunnel; the env var/file names are historical) |
| `TUNNEL_SETUP.md`, `kaggle_bootstrap.*` | the Kaggle route |

## Files in `frontend/`

| File | Role |
|---|---|
| `server.py` | Flask API: `/api/search`, `/api/benchmark`, `/api/build` — runs the compiled binaries and parses their output |
| `index.html` | the single-page UI (no build step, Chart.js from CDN) |
