#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/sync_interface_formatting_study_to_gpu.sh <user@host> [remote_dir] [ssh_key] [ssh_port] [active_dataset]

Copies only source, configuration, packaging metadata, and the active dataset
to a GPU host. Existing results, paper assets, analysis, tests, logs, and caches
are never transferred. active_dataset is a project-relative path and defaults
to mmlu_20_wrapper_robustness_60000.jsonl.
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
ACTIVE_DATASET="${5:-mmlu_20_wrapper_robustness_60000.jsonl}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTIVE_DATASET_PATH="${ROOT_DIR}/${ACTIVE_DATASET}"
if [[ ! -f "$ACTIVE_DATASET_PATH" ]]; then
  echo "Active dataset does not exist: ${ACTIVE_DATASET_PATH}" >&2
  exit 2
fi
case "$(cd "$(dirname "$ACTIVE_DATASET_PATH")" && pwd)/$(basename "$ACTIVE_DATASET_PATH")" in
  "${ROOT_DIR}"/*) ;;
  *) echo "active_dataset must be inside ${ROOT_DIR}" >&2; exit 2 ;;
esac

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
  "${ROOT_DIR}/requirements-gpu.lock" \
  "${ROOT_DIR}/README.md" \
  "${ACTIVE_DATASET_PATH}" | awk '{total += $1} END {print total * 1024}')"
echo "Thin payload bytes: ${PAYLOAD_BYTES}"
rsync -az --delete \
  --exclude '/src/*.egg-info/' \
  --exclude '/src/**/__pycache__/' \
  --include '/src/' \
  --include '/src/***' \
  --include '/configs/' \
  --include '/configs/***' \
  --include '/pyproject.toml' \
  --include '/requirements-gpu.lock' \
  --include '/README.md' \
  --include '/scripts/' \
  --include '/scripts/bootstrap_causal_followup_gpu.sh' \
  --include '/scripts/run_causal_followup_gpu.sh' \
  --include '/scripts/control_causal_followup_gpu.sh' \
  --include '/scripts/cache_causal_models.py' \
  --exclude '*' \
  -e "${RSYNC_RSH[*]}" \
  "${ROOT_DIR}/" "${REMOTE}:${REMOTE_DIR}/"

rsync -az --relative \
  -e "${RSYNC_RSH[*]}" \
  "${ROOT_DIR}/./${ACTIVE_DATASET}" \
  "${REMOTE}:${REMOTE_DIR}/"
if [[ -f "${ACTIVE_DATASET_PATH}.manifest.json" ]]; then
  rsync -az --relative \
    -e "${RSYNC_RSH[*]}" \
    "${ROOT_DIR}/./${ACTIVE_DATASET}.manifest.json" \
    "${REMOTE}:${REMOTE_DIR}/"
fi

echo "Remote payload ready at ${REMOTE}:${REMOTE_DIR}"
