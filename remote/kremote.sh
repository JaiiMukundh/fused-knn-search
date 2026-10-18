#!/usr/bin/env bash
# kremote.sh -- drive the Kaggle GPU box from this repo.
#
# Config comes from remote/kaggle.env (copy remote/kaggle.env.example and paste
# the KAGGLE_* block printed by remote/kaggle_bootstrap.py).
#
#   remote/kremote.sh ssh            interactive shell on the Kaggle box
#   remote/kremote.sh run make check-gpu
#   remote/kremote.sh push           rsync repo -> KAGGLE_DIR
#   remote/kremote.sh pull out       rsync KAGGLE_DIR/out -> ./out
#   remote/kremote.sh status         GPU / nvcc / mpirun health check
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
ENV_FILE="${KREMOTE_ENV:-$HERE/kaggle.env}"

usage() {
  cat <<'TXT'
usage: kremote.sh <command> [args...]

  ssh              open an interactive shell on the Kaggle box
  run <cmd...>     run a command there, inside KAGGLE_DIR
                   e.g.  remote/kremote.sh run make check-gpu
  push             rsync this repo -> KAGGLE_DIR (bin/, data/, out/, .git excluded)
  pull [subdir]    copy KAGGLE_DIR/<subdir> (default: out) -> ./out
  status           nvidia-smi -L, nvcc, cores, mpirun on the box
  info             print resolved connection settings
TXT
}

if [[ ! -f "$ENV_FILE" ]]; then
  echo "missing $ENV_FILE" >&2
  echo "copy remote/kaggle.env.example -> remote/kaggle.env and fill in the" >&2
  echo "KAGGLE_* block printed by the Kaggle bootstrap cell." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

: "${KAGGLE_HOST:?set KAGGLE_HOST in kaggle.env}"
: "${KAGGLE_PORT:?set KAGGLE_PORT in kaggle.env}"
KAGGLE_USER="${KAGGLE_USER:-root}"
KAGGLE_DIR="${KAGGLE_DIR:-/kaggle/working/pcap}"
KAGGLE_KEY="${KAGGLE_KEY:-$HOME/.ssh/kaggle_pcap}"
KAGGLE_KEY="${KAGGLE_KEY/#\~/$HOME}"          # expand a leading ~
KNOWN="$HOME/.ssh/known_hosts_kaggle"

SSH_OPTS=(-p "$KAGGLE_PORT"
          -o "UserKnownHostsFile=$KNOWN"
          -o StrictHostKeyChecking=accept-new
          -o ServerAliveInterval=30
          -o ServerAliveCountMax=6
          -o LogLevel=ERROR)
if [[ -f "$KAGGLE_KEY" ]]; then
  SSH_OPTS+=(-i "$KAGGLE_KEY" -o IdentitiesOnly=yes)
fi

TARGET="$KAGGLE_USER@$KAGGLE_HOST"
RSYNC_SSH="ssh ${SSH_OPTS[*]}"

CMD="${1:-help}"
shift || true

case "$CMD" in
  ssh)
    exec ssh "${SSH_OPTS[@]}" "$TARGET"
    ;;
  run)
    if [[ $# -eq 0 ]]; then echo "usage: kremote.sh run <cmd...>" >&2; exit 2; fi
    # shellcheck disable=SC2029
    ssh "${SSH_OPTS[@]}" "$TARGET" "cd '$KAGGLE_DIR' && $*"
    ;;
  push)
    rsync -az --delete \
      --exclude '.git' --exclude 'bin' --exclude 'build' \
      --exclude 'data' --exclude 'out' --exclude '.venv' \
      -e "$RSYNC_SSH" "$ROOT/" "$TARGET:$KAGGLE_DIR/"
    echo "pushed $ROOT -> $TARGET:$KAGGLE_DIR"
    ;;
  pull)
    SUB="${1:-out}"
    mkdir -p "$ROOT/out"
    rsync -az -e "$RSYNC_SSH" "$TARGET:$KAGGLE_DIR/$SUB/" "$ROOT/out/"
    echo "pulled $KAGGLE_DIR/$SUB -> $ROOT/out"
    ;;
  status)
    ssh "${SSH_OPTS[@]}" "$TARGET" \
      'nvidia-smi -L 2>/dev/null || echo "no GPU"; \
       nvcc --version 2>/dev/null | tail -1 || echo "nvcc: not found"; \
       echo "cores: $(nproc)"; \
       (mpirun --version 2>/dev/null | head -1) || echo "mpirun: not installed"'
    ;;
  info)
    printf 'host=%s port=%s user=%s dir=%s key=%s\n' \
      "$KAGGLE_HOST" "$KAGGLE_PORT" "$KAGGLE_USER" "$KAGGLE_DIR" "$KAGGLE_KEY"
    ;;
  help|-h|--help|"")
    usage
    ;;
  *)
    echo "unknown command: $CMD" >&2
    usage
    exit 2
    ;;
esac
