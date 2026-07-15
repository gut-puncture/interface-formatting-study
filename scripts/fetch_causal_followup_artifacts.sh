#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 3 ]]; then
  echo "Usage: $0 <user@host> <model-slug> <semantic-run-id> [remote_dir] [local_dir] [ssh_key] [ssh_port] [complete|partial] [canary-name]"
  exit 0
fi

REMOTE="$1"
MODEL_SLUG="$2"
SEMANTIC_RUN_ID="$3"
REMOTE_DIR="${4:-/home/ubuntu/interface_formatting_study}"
LOCAL_DIR="${5:-$(pwd)/gpu_artifacts/causal_followup}"
SSH_KEY="${6:-}"
SSH_PORT="${7:-}"
FETCH_MODE="${8:-complete}"
CANARY_NAME="${9:-}"
[[ "$FETCH_MODE" == "complete" || "$FETCH_MODE" == "partial" ]] || { echo "mode must be complete or partial" >&2; exit 2; }
[[ -z "$CANARY_NAME" || "$CANARY_NAME" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "invalid canary name" >&2; exit 2; }

SSH_ARGS=(-o StrictHostKeyChecking=accept-new)
[[ -z "$SSH_KEY" ]] || SSH_ARGS+=(-i "$SSH_KEY")
[[ -z "$SSH_PORT" ]] || SSH_ARGS+=(-p "$SSH_PORT")
RSYNC_RSH=(ssh "${SSH_ARGS[@]}")
RUN_SUFFIX=""
[[ -z "$CANARY_NAME" ]] || RUN_SUFFIX="/canaries/${CANARY_NAME}"
DEST="${LOCAL_DIR}/${MODEL_SLUG}/${SEMANTIC_RUN_ID}${RUN_SUFFIX}/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$DEST"

rsync -az -e "${RSYNC_RSH[*]}" \
  "${REMOTE}:${REMOTE_DIR}/results/causal_runs/${MODEL_SLUG}/${SEMANTIC_RUN_ID}${RUN_SUFFIX}/" \
  "${DEST}/"
PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)/src" python3 "$(dirname "$0")/verify_causal_followup_artifacts.py" \
  "$DEST" "$SEMANTIC_RUN_ID" "$MODEL_SLUG" "$FETCH_MODE"
echo "Fetched and verified causal run in ${DEST}"
