# ============================================================================
#  PCAP mini-project -- Kaggle remote-run bootstrap
#
#  Paste this whole file into one cell of a Kaggle notebook and run it.
#  Full walkthrough / troubleshooting: remote/TUNNEL_SETUP.md
#
#  It will:
#    1. report the GPU(s), nvcc, CPU cores and RAM of the Kaggle box
#    2. install the tools the project needs (openssh-server, rsync, tmux,
#       mpich) -- MPICH gives you mpicxx / mpirun for the MPI part
#    3. open a tunnel back to your laptop so you can edit/compile/run here
#       from a real terminal, one of:
#         MODE="ssh"         sshd on this box + bore.pub (or ngrok) TCP tunnel
#                            -> ssh / rsync / VS Code Remote-SSH from local
#         MODE="vscode"      VS Code tunnel (log in once with GitHub)
#                            -> connect from local VS Code or a browser
#         MODE="codeserver"  code-server (VS Code in the browser) via bore
#         MODE="sshx"        zero-config browser terminal, shareable with the team
#    4. verify the toolchain with tiny CUDA + OpenMP + MPI smoke tests
#
#  The tunnel address changes every session (and after every re-run of this
#  cell). Everything here dies with the Kaggle session; re-run the cell when
#  you start a new session. Keep the notebook tab open while you work, and
#  do not use this to dodge Kaggle's quota/usage rules -- interactive
#  development only. See the "Kaggle rules of the road" section in
#  remote/TUNNEL_SETUP.md.
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
MODE = "ssh"            # "ssh" | "vscode" | "codeserver" | "sshx"
TUNNEL = "auto"         # MODE=="ssh" only: "auto" (use ngrok if a token is set, else bore) | "bore" | "ngrok"

SSH_PORT = 2222         # port sshd listens on inside the container
AUTHORIZED_KEYS = [      # paste your LAPTOP public key(s) here, e.g.
    "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIH1/9gyPtYo3E8LFNXxGCNcsgiVmxKR1la2DR5yeycMG kaggle-pcap",
]
NGROK_AUTHTOKEN = ""    # optional: needed only for TUNNEL="ngrok" (free ngrok now wants a
                        # payment method on file for TCP tunnels). Or add a Kaggle Secret
                        # named NGROK_AUTHTOKEN and attach it to the notebook.
CODESERVER_PASSWORD = ""  # MODE="codeserver": empty -> random password is generated
REPO_URL = ""           # optional: git URL of this project; if set, it is cloned/pulled
                        # into REMOTE_DIR so you can start right away
REMOTE_DIR = "/kaggle/working/pcap"   # where the project lives on the Kaggle box
INSTALL_MPI = True                        # apt-get install mpich (mpicxx, mpirun)
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


def spawn(args, logpath, env=None):
    """Start a process detached from the kernel (survives cell re-runs)."""
    os.makedirs(os.path.dirname(logpath), exist_ok=True)
    logf = open(logpath, "ab")
    p = subprocess.Popen(args, stdout=logf, stderr=subprocess.STDOUT,
                         start_new_session=True, cwd="/", env=env)
    logf.close()
    return p


def download_file(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "pcap-kaggle-bootstrap"})
    with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    return dest


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


def kill_previous():
    """Stop tunnels/sshd from an earlier run of this cell."""
    for pidfile in (f"{LOG_DIR}/sshd.pid",):
        try:
            with open(pidfile) as f:
                pid = int(f.read().strip())
            os.kill(pid, 15)
            time.sleep(0.3)
        except Exception:
            pass
    for pat in ("/usr/local/bin/bore", "ngrok tcp", "code tunnel", "code-server -",
                "/usr/sbin/sshd -D", "/usr/local/bin/sshx"):
        try:
            subprocess.run(["pkill", "-f", pat], capture_output=True)
        except OSError:
            pass


# ----------------------------- environment ---------------------------------
def env_summary():
    line("=" * 72)
    line(" PCAP Kaggle bootstrap")
    line("=" * 72)
    if os.geteuid() != 0:
        line("! not running as root -- Kaggle notebooks are root; some steps may fail")
    if shutil.which("nvidia-smi"):
        gpus = run(["nvidia-smi", "-L"], check=False, quiet=True)
    else:
        gpus = subprocess.CompletedProcess([], 1, "", "")
    if gpus.returncode == 0 and gpus.stdout.strip():
        for gl in gpus.stdout.strip().splitlines():
            line(f"[gpu ] {gl}")
    else:
        line("[gpu ] no GPU visible (is the accelerator enabled in the right sidebar?)")
    nvcc = shutil.which("nvcc")
    if nvcc:
        v = run([nvcc, "--version"], check=False, quiet=True).stdout
        m = re.search(r"release [\d.]+", v)
        line(f"[cuda] nvcc: {nvcc} ({m.group(0) if m else 'version unknown'})")
    else:
        line("[cuda] nvcc not on PATH yet (will look in /usr/local/cuda*/bin)")
    mem = run("grep MemTotal /proc/meminfo", check=False, quiet=True).stdout.strip()
    line(f"[cpu ] cores={os.cpu_count()}  {mem}")
    line(f"[work] {REMOTE_DIR}")
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
                if "Temporary failure resolving" in tail or "Could not resolve" in tail:
                    fail("apt has no network. Turn ON 'Internet' in the notebook's "
                         "settings (right sidebar) and re-run this cell.\n" + tail)
                fail(f"apt failed: {' '.join(cmd)}\n{tail}")


def ensure_tools():
    want = []
    if MODE == "ssh":
        want.append("openssh-server")
    if INSTALL_MPI:
        want.append("mpich")
    want += EXTRA_PACKAGES
    apt_install(missing_pkgs(want))


def ensure_nvcc():
    """Make nvcc (and CUDA libs) available for every future shell."""
    nvcc = shutil.which("nvcc")
    if not nvcc:
        cands = ["/usr/local/cuda/bin/nvcc"] + sorted(glob.glob("/usr/local/cuda-*/bin/nvcc"))
        nvcc = next((c for c in cands if os.path.exists(c)), None)
    if not nvcc:
        line("[cuda] !! nvcc not found on this box. GPU builds will not work.")
        line("       On a GPU notebook it should live in /usr/local/cuda/bin/nvcc;")
        line("       if this is a CPU notebook, that is expected -- re-run with a GPU.")
        return None
    cuda_home = os.path.dirname(os.path.dirname(nvcc))
    profile = "/etc/profile.d/10-pcap-cuda.sh"
    with open(profile, "w") as f:
        f.write(f'export PATH="{cuda_home}/bin:$PATH"\n')
        f.write(f'export LD_LIBRARY_PATH="{cuda_home}/lib64:$LD_LIBRARY_PATH"\n')
    # non-login shells (plain `ssh host cmd`) do not read profile.d or .bashrc,
    # so also expose nvcc via a symlink in /usr/local/bin (nvcc resolves its
    # own install dir, so the symlink is safe)
    try:
        if not os.path.exists(f"{BIN}/nvcc"):
            os.symlink(nvcc, f"{BIN}/nvcc")
    except OSError:
        pass
    bashrc = "/root/.bashrc"
    mark = "# pcap bootstrap"
    if os.path.exists(bashrc):
        with open(bashrc) as f:
            content = f.read()
        if mark not in content:
            with open(bashrc, "a") as f:
                f.write(f"\n{mark}\n"
                        f'export PATH="{cuda_home}/bin:$PATH"\n'
                        f'export LD_LIBRARY_PATH="{cuda_home}/lib64:$LD_LIBRARY_PATH"\n')
    return nvcc


# -------------------------------- sshd -------------------------------------
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
        line("! (add your laptop's public key next time; see remote/TUNNEL_SETUP.md)")

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


# ------------------------------- tunnels -----------------------------------
def selected_tunnel():
    if TUNNEL not in ("auto", "bore", "ngrok"):
        fail(f'unknown TUNNEL="{TUNNEL}" (use auto|bore|ngrok)')
    if TUNNEL == "auto":
        return "ngrok" if ngrok_token() else "bore"
    return TUNNEL


def ngrok_token():
    if NGROK_AUTHTOKEN:
        return NGROK_AUTHTOKEN
    if os.environ.get("NGROK_AUTHTOKEN"):
        return os.environ["NGROK_AUTHTOKEN"]
    try:  # Kaggle notebook secret, if the user attached one
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("NGROK_AUTHTOKEN")
    except Exception:
        return None


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
    with tarfile.open(tgz) as t:
        t.extractall(f"{LOG_DIR}/bore_extract")
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
    with tarfile.open(tgz) as t:
        t.extractall(f"{LOG_DIR}/ngrok_extract")
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
    while time.time() < deadline:  # prefer the REST API
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
    else:
        if ngrok_token():
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
    fail(f"no tunnel could be established\n{last_err}")


def start_vscode_tunnel():
    cli = f"{BIN}/code"
    if not os.path.exists(cli):
        line("[tun ] downloading VS Code CLI (stable, alpine-x64 build)")
        tgz = download_file("https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64",
                            f"{LOG_DIR}/vscode_cli.tar.gz")
        with tarfile.open(tgz) as t:
            t.extractall(f"{LOG_DIR}/vscode_cli")
        src = glob.glob(f"{LOG_DIR}/vscode_cli/*/code") + glob.glob(f"{LOG_DIR}/vscode_cli/code")
        os.replace(src[0], cli)
        os.chmod(cli, 0o755)
    name = "pcap-kaggle"
    logpath = f"{LOG_DIR}/vscode_tunnel.log"
    spawn([cli, "tunnel", "--accept-server-license-terms", "--name", name, "--log", "trace"],
          logpath)
    wait_for_regex(logpath, r"github\.com/login/device", 60, "device-code login prompt")
    with open(logpath, errors="replace") as f:
        logged = ANSI_RE.sub("", f.read())
    mc = re.search(r"use code\s+([A-Z0-9]{4}-[A-Z0-9]{4})", logged)
    line()
    line("=" * 72)
    line(" VS CODE TUNNEL -- one-time login")
    line("=" * 72)
    line(" 1. open  https://github.com/login/device  in your browser")
    line(f" 2. enter code:  {mc.group(1) if mc else 'see the log below'}")
    line(" 3. authorise 'Visual Studio Code - Tunnel' with your GitHub account")
    line()
    line(" then connect from your laptop:")
    line("   * VS Code -> Command Palette -> 'Remote-Tunnels: Connect to Tunnel'")
    line(f"     -> pick '{name}'   (or any account that got access)")
    line(f"   * or browser:  https://vscode.dev/tunnel/{name}")
    line("=" * 72)
    line(f" (waiting for login; live log: {logpath})")
    try:
        wait_for_regex(logpath, r"Connected to tunnel|tunnel connected", 180,
                       "tunnel connection")
        line("[tun ] login complete, tunnel is up")
    except RuntimeError:
        line("[tun ] still waiting -- complete the code entry, or check the log")


def start_codeserver():
    binp = f"{BIN}/code-server"
    if not os.path.exists(binp):
        line("[tun ] downloading code-server (coder/code-server)")
        rel = json.load(urllib.request.urlopen(
            "https://api.github.com/repos/coder/code-server/releases/latest", timeout=60))
        asset = next(a["browser_download_url"] for a in rel["assets"]
                     if a["name"].endswith("linux-amd64.tar.gz"))
        tgz = download_file(asset, f"{LOG_DIR}/code-server.tar.gz")
        with tarfile.open(tgz) as t:
            t.extractall(f"{LOG_DIR}/code-server")
        src = glob.glob(f"{LOG_DIR}/code-server/*/bin/code-server") or \
            glob.glob(f"{LOG_DIR}/code-server/**/code-server", recursive=True)
        os.replace(src[0], binp)
        os.chmod(binp, 0o755)
    pw = CODESERVER_PASSWORD or secrets.token_urlsafe(10)
    env = dict(os.environ, PASSWORD=pw)
    logpath = f"{LOG_DIR}/code-server.log"
    spawn([binp, "--bind-addr", "127.0.0.1:8443", "--disable-telemetry",
           "--disable-update-check"], logpath, env=env)
    wait_port("127.0.0.1", 8443, timeout=60)
    host, port = start_bore(8443)
    line()
    line("=" * 72)
    line(" CODE-SERVER READY")
    line("=" * 72)
    line(f" open:      http://{host}:{port}")
    line(f" password:  {pw}")
    line(" you get a full VS Code in the browser (editor + terminal) on the")
    line(" GPU box. Anything you build here is on the Kaggle machine.")
    line("=" * 72)


def start_sshx():
    binp = f"{BIN}/sshx"
    if not os.path.exists(binp):
        line("[tun ] downloading sshx (github.com/ekzhang/sshx)")
        rel = json.load(urllib.request.urlopen(
            "https://api.github.com/repos/ekzhang/sshx/releases/latest", timeout=60))
        asset = next(a["browser_download_url"] for a in rel["assets"]
                     if a["name"].endswith("x86_64-unknown-linux-musl.tar.gz"))
        tgz = download_file(asset, f"{LOG_DIR}/sshx.tgz")
        with tarfile.open(tgz) as t:
            t.extractall(f"{LOG_DIR}/sshx")
        src = glob.glob(f"{LOG_DIR}/sshx/**/sshx", recursive=True)
        os.replace(src[0], binp)
        os.chmod(binp, 0o755)
    logpath = f"{LOG_DIR}/sshx.log"
    spawn([binp], logpath)
    m = wait_for_regex(logpath, r"https://sshx\.io/\S+", 40, "sshx share link")
    line()
    line("=" * 72)
    line(" SSHX TERMINAL READY (browser terminal, end-to-end encrypted)")
    line("=" * 72)
    line(f" open: {m.group(0)}")
    line(" share that link with your teammates; everyone gets the same shell.")
    line("=" * 72)


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
    line(" optional ~/.ssh/config entry (enables plain `ssh kaggle-pcap`,")
    line(" rsync helpers and VS Code Remote-SSH):")
    line()
    line("   Host kaggle-pcap")
    line(f"       HostName {host}")
    line(f"       Port {port}")
    line("       User root")
    line(f"       IdentityFile {key_hint}")
    line("       StrictHostKeyChecking accept-new")
    line("       ServerAliveInterval 30")
    line("       ServerAliveCountMax 6")
    line()
    line(" or paste into remote/kaggle.env in the project and use remote/kremote.sh:")
    line()
    line(f"   KAGGLE_HOST={host}")
    line(f"   KAGGLE_PORT={port}")
    line("   KAGGLE_USER=root")
    line(f"   KAGGLE_KEY={key_hint}")
    line(f"   KAGGLE_DIR={REMOTE_DIR}")
    line()
    line(" first commands once you are in:")
    line("   cd " + REMOTE_DIR + " && make check-gpu   # build + verify CUDA/MPI")
    line("=" * 72)


def clone_repo():
    if not REPO_URL:
        os.makedirs(REMOTE_DIR, exist_ok=True)
        return
    if os.path.isdir(f"{REMOTE_DIR}/.git"):
        run(["git", "-C", REMOTE_DIR, "pull", "--ff-only"], quiet=True)
        line(f"[git ] pulled latest into {REMOTE_DIR}")
    else:
        run(["git", "clone", REPO_URL, REMOTE_DIR], quiet=True)
        line(f"[git ] cloned {REPO_URL} -> {REMOTE_DIR}")


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    env_summary()
    kill_previous()
    ensure_tools()
    nvcc = ensure_nvcc()
    clone_repo()

    if MODE == "ssh":
        password = ensure_sshd()
        host, port, via = start_ssh_tunnel()
        print_ssh_info(host, port, via, password)
    elif MODE == "vscode":
        start_vscode_tunnel()
    elif MODE == "codeserver":
        start_codeserver()
    elif MODE == "sshx":
        start_sshx()
    else:
        fail(f'unknown MODE="{MODE}" (use ssh|vscode|codeserver|sshx)')

    if RUN_SMOKE_TESTS:
        smoke_tests(nvcc)

    line()
    line("[done] remember: this tunnel lives and dies with this notebook session.")
    line("       keep the tab open while you work; re-run this cell after a restart.")
    line("       guide + troubleshooting: remote/TUNNEL_SETUP.md")


try:
    main()
except Exception as e:
    line()
    line("BOOTSTRAP FAILED -- details above. Common causes:")
    line("  * 'Internet' toggle is off in the notebook settings")
    line("  * the accelerator is not enabled (no GPU / no nvcc)")
    line("  * a transient download failure -- just run the cell again")
    line(f"error: {e!r}")
