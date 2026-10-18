# Run the project on a Kaggle GPU from your laptop

> **Looking for the simpler route?** `remote/COLAB_SETUP.md` sets the project
> up on a free Colab GPU with a web frontend and browser VS Code, no ssh
> needed (Cloudflare quick tunnels). This guide is the raw-ssh alternative —
> it works on Kaggle and gives you a real terminal, rsync and VS Code
> Remote-SSH from the laptop.

Our laptops have no NVIDIA GPU, so all CUDA compilation and execution happens
on a **Kaggle notebook** (free GPUs: 2× T4). This guide opens an SSH tunnel
from your laptop into that notebook session and then drives it like any
remote box: edit locally, `push`, `make check-gpu`, `pull` results.

Everything here is interactive development only — see
[Rules of the road](#rules-of-the-road) before hammering it.

---

## TL;DR

```bash
# --- one time ---
ssh-keygen -t ed25519 -f ~/.ssh/kaggle_pcap          # laptop key
cat ~/.ssh/kaggle_pcap.pub                            # paste into AUTHORIZED_KEYS below

# --- every session ---
# 1. Kaggle notebook: Accelerator = GPU T4 x2, Internet = on
# 2. paste remote/kaggle_bootstrap.py into one cell, run it
# 3. copy the KAGGLE_* block it prints into remote/kaggle.env
cp remote/kaggle.env.example remote/kaggle.env       # first time only

remote/kremote.sh ssh                                 # or status / run / push / pull
remote/kremote.sh run make check-gpu
```

The tunnel address is **new every session** (and after every re-run of the
cell). Re-copy `remote/kaggle.env` each time.

---

## How it works

```
   your laptop                                Kaggle notebook (GPU T4 x2)
 +------------------+                     +--------------------------------+
 | make / knn_cuda  |   ssh -p <port>     |  sshd :2222                    |
 | VS Code / rsync  |<--------------------|    ^                           |
 | remote/kremote.sh|   bore.pub relay    |    |  bore local 2222           |
 +------------------+                     |  your repo at REMOTE_DIR       |
                                          |  nvcc, mpicxx, mpirun          |
                                          +--------------------------------+
```

The Kaggle cell does four things:

1. reports the GPU / nvcc / cores,
2. `apt-get install`s `openssh-server`, `mpich` (gives `mpicxx`, `mpirun`),
   `rsync`, `tmux` (one-time, ~1 min),
3. starts `sshd` inside the container on port 2222 and exposes it on a public
   TCP relay (`bore.pub` by default, `ngrok` optional),
4. runs tiny CUDA / OpenMP / MPI smoke tests so you know the toolchain works.

Then your laptop connects `ssh -p <public port> root@bore.pub`. You get a
root shell on a machine with a GPU, `nvcc`, and MPI — i.e. everything this
repo needs.

---

## 0. One-time laptop setup

Generate the key you'll log in with (keep the private half on the laptop):

```bash
ssh-keygen -t ed25519 -f ~/.ssh/kaggle_pcap -C kaggle-pcap
```

Open `remote/kaggle_bootstrap.py` and paste the **public** half into
`AUTHORIZED_KEYS`:

```python
AUTHORIZED_KEYS = [
    "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAA... you@laptop",   # ~/.ssh/kaggle_pcap.pub
]
```

With a key present, `sshd` is started with pubkey-only login
(`PermitRootLogin prohibit-password`, `PasswordAuthentication no`). If
`AUTHORIZED_KEYS` is left empty the bootstrap instead sets a random root
password and prints it — usable, but prefer the key.

Also create your local config (once):

```bash
cp remote/kaggle.env.example remote/kaggle.env
```

There is also a generated notebook, `remote/kaggle_bootstrap.ipynb`, if you
prefer uploading a file over copy/paste. It is produced from the `.py` by
`python3 remote/make_ipynb.py` — edit the `.py`, regenerate the `.ipynb`.

---

## 1. Kaggle: start a session

1. Go to <https://www.kaggle.com/code> → **New Notebook**.
2. Right sidebar → **Settings**:
   - **Accelerator**: `GPU T4 x2` (two GPUs → tests the MPI multi-GPU path;
     a single P100 works too, `np=1`);
   - **Internet**: **on** (the cell downloads bore, installs apt packages).
3. Paste the whole of `remote/kaggle_bootstrap.py` into one cell and run it
   (▶). First run takes ~2 minutes; afterwards ~20 s.

The cell ends with a block like:

```
=========================================================
 SSH TUNNEL READY (via bore.pub)
=========================================================
 ssh -p 41673 root@bore.pub -i ~/.ssh/kaggle_pcap
 ...
   KAGGLE_HOST=bore.pub
   KAGGLE_PORT=41673
   KAGGLE_USER=root
   KAGGLE_KEY=~/.ssh/kaggle_pcap
   KAGGLE_DIR=/kaggle/working/pcap
```

Copy those five lines into `remote/kaggle.env` (overwrite the example
values). Smoke-test lines appear above it, e.g.
`device: Tesla T4 (sm_75) | cuda smoke: PASS` and `mpi smoke: hello from
rank 0/1`.

**Keep the notebook tab open while you work.** The session (and tunnel) dies
if the session times out; re-running the cell brings it back (new port!).

---

## 2. Laptop: connect

Three equivalent doors — pick per mood:

**a) Project helper (recommended):** `remote/kremote.sh` reads
`remote/kaggle.env` and wraps the ssh flags including the keep-alive:

```bash
remote/kremote.sh info     # show resolved host/port/key/dir
remote/kremote.sh status   # nvidia-smi -L, nvcc, cores, mpirun on the box
remote/kremote.sh ssh      # interactive shell (running in KAGGLE_DIR)
```

**b) Plain ssh + `~/.ssh/config`:** add the block the bootstrap printed:

```
Host kaggle-pcap
    HostName bore.pub
    Port 41673
    User root
    IdentityFile ~/.ssh/kaggle_pcap
    StrictHostKeyChecking accept-new
    ServerAliveInterval 30
    ServerAliveCountMax 6
```

then `ssh kaggle-pcap`. (The helper achieves the same by passing flags; it
doesn't need this block.)

**c) VS Code Remote-SSH** against `kaggle-pcap` (or run `MODE="vscode"` in
the cell instead — see [Tunnel modes](#tunnel-modes)): full editor + terminal
and IntelliSense for the CUDA sources, backed by the Kaggle GPU.

`ServerAliveInterval 30 / CountMax 6` means the connection tolerates ~3
minutes of network silence before dropping. Note `ssh host cmd` (what
`kremote.sh run` uses) doesn't read `~/.bashrc` — fine here, because the
bootstrap also exposes `nvcc` via `/usr/local/bin/nvcc` and exports CUDA
paths via `/etc/profile.d`.

---

## 3. First build on the GPU box

```bash
remote/kremote.sh push           # rsync this repo -> /kaggle/working/pcap
remote/kremote.sh run make check-gpu
```

`push` excludes `bin/  build/  data/  out/  .git/  .venv/` — sources only;
everything is rebuilt on the box in ~30 s. `check-gpu` compiles
`knn_cuda` + `knn_mpi_cuda` (Auto-detects the GPU's compute capability:
T4 → sm_75, P100 → sm_60) and runs, with verification:

```
./bin/knn_cuda     --n 1000000 --d 128 --q 16 --k 10 --verify 2
mpirun -np 2 ./bin/knn_mpi_cuda --n 1000000 --d 128 --q 16 --k 10 --verify 2
```

Each binary prints `[run]`, `[time]`, `[verify] … PASS/FAIL` lines. A FAIL
means the GPU result disagrees with the double-precision CPU reference —
investigate before trusting any timing. (Locally, the CPU side is already
validated: `make check-cpu` builds and verifies `knn_omp` and
`knn_mpi_omp`.)

For your own numbers, e.g. the fused-vs-two-pass comparison:

```bash
remote/kremote.sh run ./bin/knn_cuda --n 1000000 --d 128 --q 512 --k 10 --twopass 1
```

---

## 4. Daily workflow

```
edit files locally        ->  remote/kremote.sh push
build / run / iterate     ->  remote/kremote.sh run 'make gpu && ./bin/knn_cuda --n 20000000 --d 768 --q 64 --k 10'
interactive debugging     ->  remote/kremote.sh ssh         (tmux is installed)
fetch numbers             ->  remote/kremote.sh run 'mkdir -p out && ./bin/knn_cuda ... | tee out/cuda_n1M.log'
                              remote/kremote.sh pull out   -> ./out/ locally
```

Long runs: wrap them in `tmux` on the Kaggle side so a wifi hiccup doesn't
kill the job:

```bash
remote/kremote.sh run "tmux new-session -d -s run './bin/knn_mpi_cuda --n 20000000 --d 768 --q 64 > out/big.log 2>&1'"
remote/kremote.sh run "tmux ls; tail -5 out/big.log"
```

Remember the Kaggle-side ceiling: a session lives at most **12 h**, and the
weekly GPU quota is **~30 h**, so keep the big sweeps lean and log them to
`out/`. Also note `make` there should use `-j$(nproc)` sparingly — the box
has plenty of cores, the quota counts GPU *time*, not CPU.

---

## Tunnel modes

`MODE` in `remote/kaggle_bootstrap.py` selects how you get in. `ssh` is the
default because it composes with everything (rsync, VS Code, plain terminal).

| `MODE`      | What it gives you                                    | Good for                                   | Notes |
|-------------|------------------------------------------------------|--------------------------------------------|-------|
| `ssh`       | real sshd + a public TCP tunnel to it (bore.pub, or ngrok if token set) | default; rsync, `kremote.sh`, VS Code Remote-SSH | port changes every run |
| `vscode`    | Microsoft VS Code tunnel (`code tunnel`)              | browser/desktop VS Code without extra creds | one-time GitHub device-code login per session |
| `codeserver`| `code-server` (VS Code in browser) via bore tunnel    | nothing to install locally, just a URL      | random password printed; HTTP through bore |
| `sshx`      | shareable end-to-end-encrypted browser terminal       | team session, zero config                   | everyone with the link shares one shell — great for pair-debugging or a 3-person session |

**Sharing with the team:** `sshx` is the fastest: paste its link in the team
chat, everyone gets the same terminal. For individual logins hand over
`kaggle.env` + the key (or each member adds their own public key to
`AUTHORIZED_KEYS` — multiple keys are fine). For the `vscode` mode, the
same GitHub account (or an account you grant) can connect.

### bore vs ngrok

`TUNNEL="auto"` uses ngrok **only if** an authtoken is available (constant
`NGROK_AUTHTOKEN`, env var, or an attached Kaggle Secret with that name);
otherwise — and by default — `bore.pub`. As of 2024, ngrok's free plan
requires a payment method on file for TCP endpoints, so treat ngrok as
optional. If bore fails, the helper automatically tries the other one.

If *both* are unreachable from your network, fallbacks that need no account:

| Alternative | How | Caveat |
|---|---|---|
| `ssh -R` relay | `ssh -R 2222:localhost:2222 serveo.net` (or `localhost.run`) inside the box | free relays are flaky/rate-limited; some only carry HTTP |
| Cloudflare Tunnel | `cloudflared tunnel --url ...` | quick tunnels are HTTP(S)-only — fine for the `codeserver` mode, not for raw ssh |
| Tailscale / ZeroTier | join laptop + Kaggle box to your own VPN, then ssh to the tailnet IP | needs login/account on both ends; only works while both online |
| `MODE="vscode"` / `MODE="sshx"` | see table above | no raw ssh (no rsync/Remote-SSH) |

---

## Kaggle facts to keep in mind

- **Session length:** max 12 h (9 h on TPU accelerators); GPU notebooks get
  **~30 GPU-hours/week** across all your runs. Every tunnel session eats
  quota while it's running.
- **Idle shutdown:** a GPU session that sits idle for ~20 minutes (CPU:
  ~40) gets stopped. Keep interacting, or at least keep the tab active.
  Closing the notebook tab can also end the session early.
- **The filesystem is ephemeral.** Treat `/kaggle/working` as scratch: it
  can vanish when the session ends. Your laptop is the source of truth
  (`kremote.sh push` restores the repo in seconds); push results back with
  `pull`, and commit per-run numbers to git if they matter.
- **The tunnel is per-session.** New session (or re-run of the cell) ⇒ new
  port; `remote/kaggle.env` must be updated, `kremote.sh status` is the
  quick check.
- **Internet toggle** must be on, or apt and the tunnel downloads fail with
  `Temporary failure resolving ...`.

## Rules of the road

This setup is for **interactive development** — you sitting at the keyboard
compiling and running the project.

- Don't use it to dodge Kaggle's quota limits, run batch workloads, or keep
  a session alive without working in it. Kaggle has been known to act on
  accounts that abuse long-lived sessions/tunnels, and shared/community
  GPU time is a gift we'd like to keep.
- Keep runs bounded (what you'd type yourself), log to `out/`, and shut the
  session down when done.
- The sshd only accepts keys (when configured) and dies with the session;
  nothing is left listening after it stops. Still, don't paste secrets into
  the notebook; use Kaggle Secrets if you must.
- Present teammates on your own account: sharing a notebook *link* shares the
  code, not your session — they need their own Kaggle account + key to follow
  this guide, or use the `sshx` link for a shared view.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Connection refused` / timeout to `bore.pub:PORT` | Tunnel died or port changed (session restart, cell re-run, idle shutdown). Re-run the bootstrap cell and update `remote/kaggle.env`. |
| `Permission denied (publickey)` | `AUTHORIZED_KEYS` doesn't contain your `~/.ssh/kaggle_pcap.pub`, or `KAGGLE_KEY` points at the wrong file. Verify with `ssh -v`. |
| `Host key verification failed` | Old host key for `bore.pub`. The helper keeps its own `~/.ssh/known_hosts_kaggle` and uses `accept-new`; for plain ssh, `ssh-keygen -R [bore.pub]:PORT`. |
| `apt has no network` (in bootstrap output) | Internet toggle off in notebook settings. |
| `no GPU visible` in bootstrap | Accelerator not enabled in settings; re-run the cell after enabling. |
| `nvcc not found` in a remote shell | Bootstrap couldn't find `/usr/local/cuda*/bin/nvcc`. Check the `[cuda]` line in its output; `ssh` a fresh shell (the symlink + profile live from the bootstrap run). |
| `mpirun` refuses ("running as root") | You're on OpenMPI, not the mpich we install. Use `MPIRUN_FLAGS=--allow-run-as-root` with `make`. |
| VS Code tunnel login expired | Codes last ~15 min; re-run the cell and re-enter the fresh code. |
| Terminal freezes / ssh drops | Wifi outage > keep-alive window (~3 min). Reconnect; use `tmux` on the box for long jobs. |
| Built for sm_60 but runs on T4 (or vice versa) | `make clean && make gpu` — the Makefile detects the compute capability from `nvidia-smi` at build time and embeds matching SASS plus PTX (PTX lets a *newer* GPU JIT a binary built in an older session; the reverse needs a rebuild); stale objects are the usual culprit. |

Logs on the Kaggle side live under `/tmp/pcap/` (`bore.log`, `sshd.log`,
`ngrok.log`, `vscode_tunnel.log`, `code-server.log`, `sshx.log`,
`apt.log`), so:

```bash
remote/kremote.sh run 'tail -20 /tmp/pcap/bore.log'
```

---

## Files in `remote/`

| File | Role |
|---|---|
| `kaggle_bootstrap.py` | the cell you paste into Kaggle (source of truth) |
| `kaggle_bootstrap.ipynb` | generated: same cell, ready to upload (`make_ipynb.py`) |
| `make_ipynb.py` | regenerates the notebook from the `.py` |
| `kaggle.env.example` | template for the local connection settings |
| `kaggle.env` | your real settings (gitignored) |
| `kremote.sh` | local driver: `ssh / run / push / pull / status / info` |
| `TUNNEL_SETUP.md` | this guide |
