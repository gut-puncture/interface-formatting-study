#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/sync_interface_formatting_study_to_gpu.sh <user@host> [remote_dir] [ssh_key] [ssh_port]

Copies only source, configuration, packaging metadata, and the active dataset
to a GPU host. Existing results, paper assets, analysis, tests, logs, and caches
are never transferred.
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
PAYLOAD_BYTES="$(du -sk \
  "${ROOT_DIR}/src" \
  "${ROOT_DIR}/configs" \
  "${ROOT_DIR}/pyproject.toml" \
  "${ROOT_DIR}/README.md" \
  "${ROOT_DIR}/mmlu_20_wrapper_robustness_60000.jsonl" | awk '{total += $1} END {print total * 1024}')"
echo "Thin payload bytes: ${PAYLOAD_BYTES}"
rsync -az --delete \
  --include '/src/' \
  --include '/src/***' \
  --include '/configs/' \
  --include '/configs/***' \
  --include '/pyproject.toml' \
  --include '/README.md' \
  --include '/mmlu_20_wrapper_robustness_60000.jsonl' \
  --exclude '*' \
  -e "${RSYNC_RSH[*]}" \
  "${ROOT_DIR}/" "${REMOTE}:${REMOTE_DIR}/"

echo "Remote payload ready at ${REMOTE}:${REMOTE_DIR}"
