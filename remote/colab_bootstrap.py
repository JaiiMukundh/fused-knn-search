# ============================================================================
#  PCAP mini-project -- Google Colab bootstrap (ssh tunnel + web frontend)
#
#  Paste this whole file into one cell of a Colab notebook (Runtime ->
#  Change runtime type -> GPU) and run it. Full guide: remote/COLAB_SETUP.md
#
#  Two kinds of access, both credential-free:
#
#    ssh over a public TCP relay (bore.pub by default, ngrok optional):
#    MODE="ssh"         sshd + tunnel -> real terminal, rsync and VS Code
#                       Remote-SSH from your laptop:
#                           ssh -p <port> root@bore.pub -i ~/.ssh/kaggle_pcap
#                       (cloudflared quick tunnels are HTTP-only and cannot
#                       carry raw ssh, which is why sshd goes through bore)
#
#    HTTP services via Cloudflare quick tunnels:
#    MODE="web"         a web frontend for the project (search + benchmarks)
#                       served by frontend/server.py, tunnelled to a public
#                       https://<random>.trycloudflare.com URL
#    MODE="codeserver"  VS Code in the browser (code-server + password),
#                       tunnelled the same way -> edit + build in the browser
#    MODE="sshx"        shareable browser terminal (sshx.io), zero config
#
#    MODE="all"         ssh + web + sshx (default); or combine, e.g.
#                       MODE="ssh,web" / MODE="ssh,codeserver"
#
#  The cell also clones/pulls the GitHub repo, builds the binaries (make gpu)
#  and runs tiny CUDA/OpenMP/MPI smoke tests so you know the toolchain works.
#
#  Quick tunnels get a new random URL every run; everything here dies with
#  the Colab runtime. Keep the tab open while you work (idle runtimes are
#  recycled), and treat this as interactive development only.
# ============================================================================

import glob
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import tarfile
import time
import urllib.request

# ------------------------------- CONFIG ------------------------------------
# see the header: "all" = ssh + web + sshx, or list modes explicitly, e.g.
# MODE = "ssh,web"  |  "ssh"  |  "codeserver"  |  "sshx"
MODE = "all"
REPO_URL = "https://github.com/JaiiMukundh/fused-knn-search"
WORK_DIR = "/content/pcap"        # where the repo lives
FRONTEND_PORT = 8080              # frontend/server.py listens here
CODESERVER_PORT = 8443            # code-server listens here
CODESERVER_PASSWORD = ""          # empty -> random password is generated
SSH_PORT = 2222                   # sshd listens on this port inside the VM
SSH_TUNNEL = "auto"               # "auto" | "bore" | "ngrok"
AUTHORIZED_KEYS = [               # your LAPTOP public key(s) -> passwordless ssh
    "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIH1/9gyPtYo3E8LFNXxGCNcsgiVmxKR1la2DR5yeycMG kaggle-pcap",
]
NGROK_AUTHTOKEN = ""              # optional: only needed for SSH_TUNNEL="ngrok"
BUILD_ON_START = True             # make gpu after clone/pull (needs nvcc)
INSTALL_MPI = True                # apt-get install mpich (mpicxx, mpirun)
EXTRA_PACKAGES = ["tmux", "rsync", "htop"]
RUN_SMOKE_TESTS = True
# ----------------------------------------------------------------------------

BIN = "/usr/local/bin"
LOG_DIR = "/tmp/pcap"
PKG_LOG = f"{LOG_DIR}/apt.log"

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def line(txt=""):
    print(txt, flush=True)


def fail(msg):
    raise RuntimeError(msg)


def run(cmd, check=True, timeout=900, quiet=False, env=None):
    printable = cmd if isinstance(cmd, str) else " ".join(cmd)
    if not quiet:
        line(f"$ {printable}")
    p = subprocess.run(cmd, shell=isinstance(cmd, str), text=True, capture_output=True,
                       timeout=timeout, env=env)
    out = (p.stdout or "") + (p.stderr or "")
    if check and p.returncode != 0:
        tail = "\n".join(out.strip().splitlines()[-15:])
        fail(f"command failed (rc={p.returncode}): {printable}\n{tail}")
    return p


def spawn(args, logpath, cwd="/", env=None):
    """Start a process detached from the kernel (survives cell re-runs)."""
    os.makedirs(os.path.dirname(logpath), exist_ok=True)
    logf = open(logpath, "ab")
    p = subprocess.Popen(args, stdout=logf, stderr=subprocess.STDOUT,
                         start_new_session=True, cwd=cwd, env=env)
    logf.close()
    return p


def download_file(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "pcap-colab-bootstrap"})
    with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    return dest


def extract(tgz, dest):
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(tgz) as t:
        try:
            t.extractall(dest, filter="data")          # py >= 3.11.4
        except TypeError:                              # pragma: no cover
            t.extractall(dest)


def wait_port(host, port, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.3)
    fail(f"port {host}:{port} did not open within {timeout}s")


def wait_for_regex(logpath, pattern, timeout, what):
    rx = re.compile(pattern)
    text = ""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with open(logpath, "r", errors="replace") as f:
                text = f.read()
        except OSError:
            text = ""
        m = rx.search(ANSI_RE.sub("", text))
        if m:
            return m
        time.sleep(0.5)
    tail = "\n".join(text.strip().splitlines()[-12:])
    fail(f"timed out waiting for {what} (log: {logpath})\n--- log tail ---\n{tail}")


def log_tail(logpath, n=6):
    try:
        with open(logpath, "r", errors="replace") as f:
            lines = f.read().strip().splitlines()
        return "\n".join(lines[-n:])
    except OSError:
        return "(no log)"


def http_ok(url, timeout=10):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "pcap-colab-bootstrap"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def kill_previous():
    """Stop tunnels/services from an earlier run of this cell."""
    for pat in ("cloudflared tunnel", "frontend/server.py", "code-server -",
                "/usr/local/bin/sshx", "/usr/sbin/sshd", "bore local"):
        try:
            subprocess.run(["pkill", "-f", pat], capture_output=True)
        except OSError:
            pass


def _modes():
    raw = {m.strip() for m in MODE.replace(",", " ").split() if m.strip()}
    if "all" in raw:
        raw = {"ssh", "web", "sshx"}
    unknown = raw - {"ssh", "web", "codeserver", "sshx"}
    if unknown:
        fail(f'unknown MODE entries: {sorted(unknown)} (use ssh|web|codeserver|sshx or "all")')
    return raw


# ----------------------------- environment ---------------------------------
def env_summary():
    line("=" * 72)
    line(" PCAP Colab bootstrap (Cloudflare quick tunnels)")
    line("=" * 72)
    if os.geteuid() != 0:
        line("! not running as root -- Colab runtimes are root; some steps may fail")
    if shutil.which("nvidia-smi"):
        gpus = run(["nvidia-smi", "-L"], check=False, quiet=True)
    else:
        gpus = subprocess.CompletedProcess([], 1, "", "")
    if gpus.returncode == 0 and gpus.stdout.strip():
        for gl in gpus.stdout.strip().splitlines():
            line(f"[gpu ] {gl}")
    else:
        line("[gpu ] no GPU visible (Runtime -> Change runtime type -> GPU, then re-run)")
    nvcc = shutil.which("nvcc")
    if nvcc:
        v = run([nvcc, "--version"], check=False, quiet=True).stdout
        m = re.search(r"release [\d.]+", v)
        line(f"[cuda] nvcc: {nvcc} ({m.group(0) if m else 'version unknown'})")
    else:
        line("[cuda] nvcc not on PATH yet (will look in /usr/local/cuda*/bin)")
    mem = run("grep MemTotal /proc/meminfo", check=False, quiet=True).stdout.strip()
    free = shutil.disk_usage("/content" if os.path.isdir("/content") else "/").free / 1e9
    line(f"[cpu ] cores={os.cpu_count()}  {mem}  disk free: {free:.0f} GB")
    line(f"[work] {WORK_DIR}")
    line()


def missing_pkgs(pkgs):
    out = []
    for p in pkgs:
        try:
            r = subprocess.run(["dpkg-query", "-W", "-f=${Status}", p],
                               capture_output=True, text=True)
            if "install ok installed" not in r.stdout:
                out.append(p)
        except OSError:
            out.append(p)
    return out


def apt_install(pkgs):
    if not pkgs:
        return
    line(f"[apt ] installing: {', '.join(pkgs)}  (one-time, ~1 min)")
    env = dict(os.environ, DEBIAN_FRONTEND="noninteractive")
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(PKG_LOG, "ab") as logf:
        for cmd in (["apt-get", "-o", "Acquire::Retries=3", "update", "-qq"],
                    ["apt-get", "-o", "Acquire::Retries=3", "install", "-y", "-qq"] + pkgs):
            p = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
            if p.returncode != 0:
                tail = log_tail(PKG_LOG, 15)
                fail(f"apt failed: {' '.join(cmd)}\n{tail}")


def ensure_tools():
    want = list(EXTRA_PACKAGES)
    if INSTALL_MPI:
        want.append("mpich")
    if "ssh" in _modes():
        want.append("openssh-server")
    apt_install(missing_pkgs(want))


def ensure_nvcc():
    """Make nvcc (and CUDA libs) available for every future shell."""
    nvcc = shutil.which("nvcc")
    if not nvcc:
        cands = ["/usr/local/cuda/bin/nvcc"] + sorted(glob.glob("/usr/local/cuda-*/bin/nvcc"))
        nvcc = next((c for c in cands if os.path.exists(c)), None)
    if not nvcc:
        line("[cuda] !! nvcc not found on this box. GPU builds will not work.")
        line("       Re-run with a GPU runtime; on Colab it lives in /usr/local/cuda/bin/.")
        return None
    cuda_home = os.path.dirname(os.path.dirname(nvcc))
    profile = "/etc/profile.d/10-pcap-cuda.sh"
    with open(profile, "w") as f:
        f.write(f'export PATH="{cuda_home}/bin:$PATH"\n')
        f.write(f'export LD_LIBRARY_PATH="{cuda_home}/lib64:$LD_LIBRARY_PATH"\n')
    try:
        if not os.path.exists(f"{BIN}/nvcc"):
            os.symlink(nvcc, f"{BIN}/nvcc")  # nvcc resolves its own install dir
    except OSError:
        pass
    return nvcc


# --------------------------------- sshd ------------------------------------
def ensure_sshd():
    os.makedirs("/run/sshd", exist_ok=True)
    os.makedirs("/root/.ssh", mode=0o700, exist_ok=True)
    run(["ssh-keygen", "-A"], quiet=True)

    keys = [k.strip() for k in AUTHORIZED_KEYS if k.strip()]
    password = None
    if keys:
        with open("/root/.ssh/authorized_keys", "w") as f:
            f.write("\n".join(keys) + "\n")
        os.chmod("/root/.ssh/authorized_keys", 0o600)
    else:
        password = secrets.token_urlsafe(12)
        run(["bash", "-c", f"echo 'root:{password}' | chpasswd"], quiet=True)
        line("! AUTHORIZED_KEYS is empty -> falling back to password login")
        line("! (add your laptop's public key to remote/colab_bootstrap.py)")

    logpath = f"{LOG_DIR}/sshd.log"
    args = ["/usr/sbin/sshd", "-D", "-e",
            "-p", str(SSH_PORT),
            "-o", f"PidFile={LOG_DIR}/sshd.pid",
            "-o", "PermitRootLogin=" + ("prohibit-password" if keys else "yes"),
            "-o", "PubkeyAuthentication=yes",
            "-o", "PasswordAuthentication=" + ("no" if keys else "yes"),
            "-o", "UsePAM=no", "-o", "StrictModes=no",
            "-o", "AllowTcpForwarding=yes", "-o", "PrintMotd=no",
            "-o", "ClientAliveInterval=60", "-o", "ClientAliveCountMax=30"]
    spawn(args, logpath)
    try:
        wait_port("127.0.0.1", SSH_PORT, timeout=10)
    except RuntimeError as e:
        fail(f"sshd did not start\n{e}\n--- sshd log ---\n{log_tail(logpath, 20)}")
    line(f"[ssh ] sshd listening on port {SSH_PORT}")
    return password


# ------------------------------- ssh tunnel --------------------------------
def selected_tunnel():
    if SSH_TUNNEL not in ("auto", "bore", "ngrok"):
        fail(f'unknown SSH_TUNNEL="{SSH_TUNNEL}" (use auto|bore|ngrok)')
    if SSH_TUNNEL == "auto":
        return "ngrok" if ngrok_token() else "bore"
    return SSH_TUNNEL


def ngrok_token():
    if NGROK_AUTHTOKEN:
        return NGROK_AUTHTOKEN
    return os.environ.get("NGROK_AUTHTOKEN")


def ensure_bore():
    binp = f"{BIN}/bore"
    if os.path.exists(binp):
        return binp
    line("[tun ] downloading bore (github.com/ekzhang/bore)")
    rel = json.load(urllib.request.urlopen(
        "https://api.github.com/repos/ekzhang/bore/releases/latest", timeout=60))
    asset = next(a["browser_download_url"] for a in rel["assets"]
                 if a["name"].endswith("x86_64-unknown-linux-musl.tar.gz"))
    tgz = download_file(asset, f"{LOG_DIR}/bore.tgz")
    extract(tgz, f"{LOG_DIR}/bore_extract")
    src = glob.glob(f"{LOG_DIR}/bore_extract/*/bore") + glob.glob(f"{LOG_DIR}/bore_extract/bore")
    os.replace(src[0], binp)
    os.chmod(binp, 0o755)
    return binp


def start_bore(local_port):
    binp = ensure_bore()
    logpath = f"{LOG_DIR}/bore.log"
    spawn([binp, "local", str(local_port), "--to", "bore.pub"], logpath)
    m = wait_for_regex(logpath, r"listening at [^:\s]+:(\d+)", 40, "bore public port")
    return "bore.pub", int(m.group(1))


def ensure_ngrok():
    binp = f"{BIN}/ngrok"
    if os.path.exists(binp):
        return binp
    line("[tun ] downloading ngrok agent")
    tgz = download_file("https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz",
                        f"{LOG_DIR}/ngrok.tgz")
    extract(tgz, f"{LOG_DIR}/ngrok_extract")
    os.replace(f"{LOG_DIR}/ngrok_extract/ngrok", binp)
    os.chmod(binp, 0o755)
    return binp


def start_ngrok(local_port):
    token = ngrok_token()
    if not token:
        return None
    binp = ensure_ngrok()
    run([binp, "config", "add-authtoken", token], quiet=True)
    logpath = f"{LOG_DIR}/ngrok.log"
    spawn([binp, "tcp", str(local_port), "--log", "stdout", "--log-format", "logfmt"], logpath)
    deadline = time.time() + 45
    while time.time() < deadline:  # prefer the local REST API
        try:
            with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=2) as r:
                data = json.load(r)
            for t in data.get("tunnels", []):
                u = t.get("public_url", "")
                if u.startswith("tcp://"):
                    host, port = u[6:].rsplit(":", 1)
                    return host, int(port)
        except Exception:
            pass
        time.sleep(1)
    m = wait_for_regex(logpath, r"url=tcp://([^:\s]+):(\d+)", 10, "ngrok tcp url")
    return m.group(1), int(m.group(2))


def start_ssh_tunnel():
    order = [selected_tunnel()]
    if order[0] == "ngrok":
        order.append("bore")
    elif ngrok_token():
        order.append("ngrok")
    last_err = None
    for name in order:
        try:
            if name == "ngrok":
                got = start_ngrok(SSH_PORT)
                if got:
                    return got[0], got[1], "ngrok"
                line("[tun ] no ngrok authtoken available; falling back")
            else:
                host, port = start_bore(SSH_PORT)
                return host, port, "bore.pub"
        except Exception as e:  # try the next tunnel
            last_err = e
            line(f"[tun ] {name} failed: {e}")
    fail(f"no ssh tunnel could be established\n{last_err}")


# -------------------------------- repo -------------------------------------
def clone_repo():
    if not REPO_URL:
        os.makedirs(WORK_DIR, exist_ok=True)
        line(f"[git ] REPO_URL empty -- push sources yourself into {WORK_DIR}")
        return
    if os.path.isdir(f"{WORK_DIR}/.git"):
        run(["git", "-C", WORK_DIR, "pull", "--ff-only"], quiet=True)
        line(f"[git ] pulled latest into {WORK_DIR}")
    else:
        run(["git", "clone", REPO_URL, WORK_DIR], quiet=True)
        line(f"[git ] cloned {REPO_URL} -> {WORK_DIR}")


def build_repo(nvcc):
    if not BUILD_ON_START:
        return
    if not os.path.isdir(WORK_DIR):
        return
    line("[make] building (make gpu; first build ~1-2 min)")
    p = run(["make", "gpu"], check=False, quiet=True, timeout=1800, env=None)
    ok = p.returncode == 0
    tail = "\n".join(((p.stdout or "") + (p.stderr or "")).strip().splitlines()[-6:])
    if ok:
        line("[make] gpu build OK (bin/knn_cuda, bin/knn_mpi_cuda)")
    else:
        line("[make] gpu build FAILED -- fix via the web UI's Build button or:")
        line(f"       cd {WORK_DIR} && make gpu")
        line(tail)
    p = run(["make", "cpu"], check=False, quiet=True, timeout=900)
    if p.returncode == 0:
        line("[make] cpu build OK (bin/knn_omp, bin/knn_mpi_omp)")


# ------------------------------ cloudflared --------------------------------
def ensure_cloudflared():
    binp = f"{BIN}/cloudflared"
    if os.path.exists(binp):
        return binp
    line("[tun ] downloading cloudflared (Cloudflare Tunnel agent)")
    download_file("https://github.com/cloudflare/cloudflared/releases/latest/download/"
                  "cloudflared-linux-amd64", binp)
    os.chmod(binp, 0o755)
    return binp


def start_cloudflared(local_port, name, metrics_port=20241):
    """Expose http://127.0.0.1:<local_port> on a random trycloudflare.com URL."""
    binp = ensure_cloudflared()
    logpath = f"{LOG_DIR}/cloudflared-{name}.log"
    stdout_path = f"{LOG_DIR}/cloudflared-{name}.stdout.log"
    spawn([binp, "tunnel", "--url", f"http://127.0.0.1:{local_port}",
           "--no-autoupdate", "--protocol", "http2",
           "--metrics", f"127.0.0.1:{metrics_port}",
           "--logfile", logpath], stdout_path)
    m = wait_for_regex(logpath, r"https://[a-z0-9-]+\.trycloudflare\.com", 60,
                       f"quick tunnel URL ({name})")
    url = m.group(0)
    for _ in range(40):  # the edge takes a few seconds to start answering
        if http_ok(url + "/", timeout=8):
            break
        time.sleep(1)
    return url


# ------------------------------ code-server --------------------------------
def start_codeserver():
    binp = f"{BIN}/code-server"
    if not os.path.exists(binp):
        line("[tun ] downloading code-server (coder/code-server)")
        rel = json.load(urllib.request.urlopen(
            "https://api.github.com/repos/coder/code-server/releases/latest", timeout=60))
        asset = next(a["browser_download_url"] for a in rel["assets"]
                     if a["name"].endswith("linux-amd64.tar.gz"))
        download_file(asset, f"{LOG_DIR}/code-server.tar.gz")
        extract(f"{LOG_DIR}/code-server.tar.gz", f"{LOG_DIR}/code-server")
        src = glob.glob(f"{LOG_DIR}/code-server/*/bin/code-server") or \
            glob.glob(f"{LOG_DIR}/code-server/**/code-server", recursive=True)
        os.replace(src[0], binp)
        os.chmod(binp, 0o755)
    pw = CODESERVER_PASSWORD or secrets.token_urlsafe(10)
    env = dict(os.environ, PASSWORD=pw)
    logpath = f"{LOG_DIR}/code-server.log"
    spawn([binp, "--bind-addr", f"127.0.0.1:{CODESERVER_PORT}", "--disable-telemetry",
           "--disable-update-check", WORK_DIR], logpath, env=env)
    wait_port("127.0.0.1", CODESERVER_PORT, timeout=60)
    url = start_cloudflared(CODESERVER_PORT, "codeserver", metrics_port=20242)
    return {"url": url, "password": pw}


# --------------------------------- sshx ------------------------------------
def start_sshx():
    binp = f"{BIN}/sshx"
    if not os.path.exists(binp):
        line("[tun ] downloading sshx (github.com/ekzhang/sshx)")
        rel = json.load(urllib.request.urlopen(
            "https://api.github.com/repos/ekzhang/sshx/releases/latest", timeout=60))
        asset = next(a["browser_download_url"] for a in rel["assets"]
                     if a["name"].endswith("x86_64-unknown-linux-musl.tar.gz"))
        download_file(asset, f"{LOG_DIR}/sshx.tgz")
        extract(f"{LOG_DIR}/sshx.tgz", f"{LOG_DIR}/sshx")
        src = glob.glob(f"{LOG_DIR}/sshx/**/sshx", recursive=True)
        os.replace(src[0], binp)
        os.chmod(binp, 0o755)
    logpath = f"{LOG_DIR}/sshx.log"
    spawn([binp], logpath)
    m = wait_for_regex(logpath, r"https://sshx\.io/\S+", 40, "sshx share link")
    return m.group(0)


# ------------------------------- frontend ----------------------------------
def start_frontend():
    if not os.path.isdir(WORK_DIR):
        fail(f"{WORK_DIR} missing (repo not cloned?) -- nothing to serve")
    if subprocess.run(["python3", "-c", "import flask"], capture_output=True).returncode != 0:
        line("[web ] installing flask (one-time)")
        run(["python3", "-m", "pip", "install", "-q", "flask"], quiet=True, timeout=300)
    logpath = f"{LOG_DIR}/frontend.log"
    spawn(["python3", f"{WORK_DIR}/frontend/server.py", "--port", str(FRONTEND_PORT)],
          logpath, cwd=WORK_DIR)
    try:
        wait_port("127.0.0.1", FRONTEND_PORT, timeout=40)
    except RuntimeError as e:
        fail(f"frontend server did not start\n{e}\n--- log ---\n{log_tail(logpath, 20)}")
    if not http_ok(f"http://127.0.0.1:{FRONTEND_PORT}/api/health", timeout=5):
        fail(f"frontend health check failed\n--- log ---\n{log_tail(logpath, 20)}")
    url = start_cloudflared(FRONTEND_PORT, "web", metrics_port=20241)
    return url


# ------------------------------ smoke tests --------------------------------
CUDA_SMOKE = r"""
#include <cuda_runtime.h>
#include <cstdio>
__global__ void add(float* x){ x[threadIdx.x] += 1.0f; }
int main() {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess || n < 1) { printf("no cuda device: %s\n", cudaGetErrorString(e)); return 1; }
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    float h[8]; for (int i = 0; i < 8; ++i) h[i] = (float)i;
    float* d = nullptr; cudaMalloc(&d, sizeof(h));
    cudaMemcpy(d, h, sizeof(h), cudaMemcpyHostToDevice);
    add<<<1, 8>>>(d);
    cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost);
    int ok = (cudaGetLastError() == cudaSuccess);
    for (int i = 0; i < 8; ++i) ok = ok && (h[i] == (float)i + 1.0f);
    printf("device: %s (sm_%d%d) | cuda smoke: %s\n", p.name, p.major, p.minor, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
"""

OMP_SMOKE = r"""
#include <omp.h>
#include <stdio.h>
int main() {
    long sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < 1000000; ++i) sum += i;
    printf("openmp smoke: %s (threads=%d, sum=%ld)\n",
           sum == 499999500000L ? "PASS" : "FAIL", omp_get_max_threads(), sum);
    return sum == 499999500000L ? 0 : 1;
}
"""

MPI_SMOKE = r"""
#include <mpi.h>
#include <stdio.h>
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = -1, size = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("mpi smoke: hello from rank %d/%d\n", rank, size);
    MPI_Finalize();
    return 0;
}
"""


def smoke_tests(nvcc):
    line()
    line("[test] smoke tests")
    os.makedirs(f"{LOG_DIR}/smoke", exist_ok=True)

    if nvcc:
        with open(f"{LOG_DIR}/smoke/cuda.cu", "w") as f:
            f.write(CUDA_SMOKE)
        p = run([nvcc, "-o", f"{LOG_DIR}/smoke/cuda_smoke", f"{LOG_DIR}/smoke/cuda.cu"],
                check=False, quiet=True, timeout=600)
        if p.returncode == 0:
            p = run([f"{LOG_DIR}/smoke/cuda_smoke"], check=False, quiet=True, timeout=120)
            line("[test] " + (p.stdout.strip() or p.stderr.strip() or "cuda smoke failed"))
        else:
            line("[test] CUDA smoke compile FAILED:")
            line((p.stderr or p.stdout).strip()[-800:])

    with open(f"{LOG_DIR}/smoke/omp.c", "w") as f:
        f.write(OMP_SMOKE)
    p = run(["gcc", "-O2", "-fopenmp", "-o", f"{LOG_DIR}/smoke/omp_smoke",
             f"{LOG_DIR}/smoke/omp.c"], check=False, quiet=True, timeout=120)
    if p.returncode == 0:
        p = run([f"{LOG_DIR}/smoke/omp_smoke"], check=False, quiet=True, timeout=120)
        line("[test] " + (p.stdout.strip() or "openmp smoke failed"))
    else:
        line("[test] OpenMP smoke compile failed: " + (p.stderr or "").strip()[-300:])

    mpicxx, mpirun = shutil.which("mpicxx"), shutil.which("mpirun")
    if INSTALL_MPI and mpicxx and mpirun:
        with open(f"{LOG_DIR}/smoke/mpi.c", "w") as f:
            f.write(MPI_SMOKE)
        p = run([mpicxx, "-o", f"{LOG_DIR}/smoke/mpi_smoke", f"{LOG_DIR}/smoke/mpi.c"],
                check=False, quiet=True, timeout=180)
        if p.returncode == 0:
            p = run([mpirun, "-np", "2", f"{LOG_DIR}/smoke/mpi_smoke"], check=False,
                    quiet=True, timeout=120)
            out = (p.stdout or "") + (p.stderr or "")
            if p.returncode != 0 and "root" in out.lower():
                p = run([mpirun, "--allow-run-as-root", "-np", "2",
                         f"{LOG_DIR}/smoke/mpi_smoke"], check=False, quiet=True, timeout=120)
                out = (p.stdout or "") + (p.stderr or "")
            line("[test] " + (out.strip() or "mpi smoke failed"))
        else:
            line("[test] MPI smoke compile failed: " + (p.stderr or "").strip()[-300:])


# --------------------------------- output ----------------------------------
def print_ssh_info(host, port, via, password):
    key_hint = "~/.ssh/kaggle_pcap"
    line()
    line("=" * 72)
    line(f" SSH TUNNEL READY (via {via})")
    line("=" * 72)
    line(f" ssh -p {port} root@{host} -i {key_hint}")
    if password:
        line(f" (password login fallback: ssh -p {port} root@{host}   password: {password})")
    line()
    line(" optional ~/.ssh/config entry (enables plain `ssh gpu-pcap`,")
    line(" rsync helpers and VS Code Remote-SSH):")
    line()
    line("   Host gpu-pcap")
    line(f"       HostName {host}")
    line(f"       Port {port}")
    line("       User root")
    line(f"       IdentityFile {key_hint}")
    line("       StrictHostKeyChecking accept-new")
    line("       ServerAliveInterval 30")
    line("       ServerAliveCountMax 6")
    line()
    line(" or write it into remote/kaggle.env and drive the box with")
    line(" remote/kremote.sh (the same tooling as the Kaggle route):")
    line()
    line(f"   KAGGLE_HOST={host}")
    line(f"   KAGGLE_PORT={port}")
    line("   KAGGLE_USER=root")
    line(f"   KAGGLE_KEY={key_hint}")
    line(f"   KAGGLE_DIR={WORK_DIR}")
    line()
    line(" first commands once you are in:")
    line(f"   cd {WORK_DIR} && make check-gpu   # build + verify CUDA/MPI on the GPU")
    line("=" * 72)


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    modes = _modes()
    env_summary()
    kill_previous()
    ensure_tools()
    nvcc = ensure_nvcc()
    clone_repo()
    build_repo(nvcc)

    ssh_info = None
    web_url = cs = sshx_url = None
    if "ssh" in modes:
        password = ensure_sshd()
        host, port, via = start_ssh_tunnel()
        ssh_info = (host, port, via, password)
    if "web" in modes:
        web_url = start_frontend()
    if "codeserver" in modes:
        cs = start_codeserver()
    if "sshx" in modes:
        sshx_url = start_sshx()

    if ssh_info:
        print_ssh_info(*ssh_info)

    if RUN_SMOKE_TESTS:
        smoke_tests(nvcc)

    line()
    line("=" * 72)
    line(f" COLAB GPU BOX READY  (modes: {', '.join(sorted(modes))})")
    line("=" * 72)
    if ssh_info:
        host, port, via, _ = ssh_info
        line(f" ssh            : ssh -p {port} root@{host} -i ~/.ssh/kaggle_pcap")
    if web_url:
        line(f" web frontend   : {web_url}")
    if cs:
        line(f" browser VS Code: {cs['url']}   (password: {cs['password']})")
    if sshx_url:
        line(f" shared terminal: {sshx_url}")
    line(f" work dir       : {WORK_DIR}")
    line()
    line(" ssh gives you the whole box (terminal, rsync, VS Code Remote-SSH,")
    line(" mpirun); the web frontend runs searches/sweeps and shows the")
    line(f" fused-vs-two-pass timings. first check:  cd {WORK_DIR} && make check-gpu")
    line()
    line(" every hostname/port above changes on each run (and when the runtime")
    line(" restarts, re-run this cell). Keep the tab open while you work --")
    line(" idle Colab runtimes are recycled and everything here dies with the")
    line(" runtime. guide + troubleshooting: remote/COLAB_SETUP.md")
    line("=" * 72)


try:
    main()
except Exception as e:
    line()
    line("BOOTSTRAP FAILED -- details above. Common causes:")
    line("  * the runtime has no GPU (Runtime -> Change runtime type -> GPU)")
    line("  * transient download/pip failure -- just run the cell again")
    line("  * Cloudflare edge slow to publish the URL -- re-run in a minute")
    line(f"error: {e!r}")
