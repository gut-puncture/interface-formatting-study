#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/fetch_interface_formatting_study_artifacts.sh <user@host> <model-slug> <semantic-run-id> [remote_dir] [local_dir] [ssh_key] [ssh_port] [complete|partial]

Fetches one model-specific run tree and validates semantic identity plus every
incremental shard checksum on the Mac.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 3 ]]; then
  usage
  exit 0
fi

REMOTE="${1:?remote user@host is required}"
MODEL_SLUG="${2:?model slug is required}"
SEMANTIC_RUN_ID="${3:?semantic run id is required}"
REMOTE_DIR="${4:-/home/ubuntu/interface_formatting_study}"
LOCAL_DIR="${5:-$(pwd)/gpu_artifacts}"
SSH_KEY="${6:-}"
SSH_PORT="${7:-}"
FETCH_MODE="${8:-complete}"
if [[ "$FETCH_MODE" != "complete" && "$FETCH_MODE" != "partial" ]]; then
  echo "fetch mode must be complete or partial" >&2
  exit 2
fi

SSH_ARGS=(-o StrictHostKeyChecking=accept-new)
if [[ -n "$SSH_KEY" ]]; then
  SSH_ARGS+=(-i "$SSH_KEY")
fi
if [[ -n "$SSH_PORT" ]]; then
  SSH_ARGS+=(-p "$SSH_PORT")
fi

RSYNC_RSH=(ssh "${SSH_ARGS[@]}")
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DEST="${LOCAL_DIR}/${MODEL_SLUG}/${SEMANTIC_RUN_ID}/${STAMP}"
mkdir -p "$DEST"

REMOTE_RUN="${REMOTE_DIR}/results/model_runs/${MODEL_SLUG}/${SEMANTIC_RUN_ID}"
echo "Fetching ${REMOTE_RUN} into ${DEST}"
rsync -az \
  -e "${RSYNC_RSH[*]}" \
  "${REMOTE}:${REMOTE_RUN}/" "${DEST}/"

python3 "$(dirname "$0")/verify_interface_formatting_study_artifacts.py" \
  "$DEST" "$SEMANTIC_RUN_ID" "$MODEL_SLUG" "$FETCH_MODE"

echo "Fetched and independently verified model run in ${DEST}"
