#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/sync_interface_formatting_study_to_gpu.sh <user@host> [remote_dir] [ssh_key] [ssh_port]

Copies the minimal INTERFACE_FORMATTING_STUDY repo payload to a GPU host. It excludes local caches,
large generated outputs, and model caches, but includes the source dataset and
small reusable behavioral/conflict artifacts when present.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 1 ]]; then
  usage
  exit 0
fi

REMOTE="${1:?remote user@host is required}"
REMOTE_DIR="${2:-/home/ubuntu/interface_formatting_study}"
SSH_KEY="${3:-}"
SSH_PORT="${4:-}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SSH_ARGS=(-o StrictHostKeyChecking=accept-new)
if [[ -n "$SSH_KEY" ]]; then
  SSH_ARGS+=(-i "$SSH_KEY")
fi
if [[ -n "$SSH_PORT" ]]; then
  SSH_ARGS+=(-p "$SSH_PORT")
fi

RSYNC_RSH=(ssh "${SSH_ARGS[@]}")

echo "Creating ${REMOTE_DIR} on ${REMOTE}"
ssh "${SSH_ARGS[@]}" "$REMOTE" "mkdir -p '$REMOTE_DIR'"

echo "Syncing INTERFACE_FORMATTING_STUDY payload from ${ROOT_DIR}"
rsync -az --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.mypy_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '.cache/' \
  --exclude 'artifacts/' \
  --exclude 'gpu_artifacts/' \
  --exclude 'run_logs/' \
  --exclude 'results/raw/behavioral_shards/' \
  --exclude 'results/processed/patching_results.parquet' \
  --exclude 'results/processed/interface_formatting_study_vector.pt' \
  --exclude 'results/processed/shuffled_pair_vector.pt' \
  --exclude 'results/processed/alpha_tuning_details.parquet' \
  --exclude 'results/processed/access_vector_results.parquet' \
  --exclude 'results/processed/control_results.parquet' \
  --exclude 'results/processed/removal_results.parquet' \
  --exclude 'results/processed/content_free_control.parquet' \
  --exclude 'results/figures/' \
  --exclude 'results/tables/' \
  --exclude 'results/smoke/' \
  -e "${RSYNC_RSH[*]}" \
  "${ROOT_DIR}/" "${REMOTE}:${REMOTE_DIR}/"

echo "Remote payload ready at ${REMOTE}:${REMOTE_DIR}"
