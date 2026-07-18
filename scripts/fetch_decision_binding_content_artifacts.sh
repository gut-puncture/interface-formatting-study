#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 3 ]]; then
  echo "Usage: $0 <user@host> <model-slug> <semantic-run-id> [remote-dir] [local-dir] [ssh-key] [ssh-port] [canary|complete]"
  exit 0
fi

REMOTE="$1"; MODEL_SLUG="$2"; RUN_ID="$3"
REMOTE_DIR="${4:-/home/ubuntu/interface_formatting_study}"
LOCAL_DIR="${5:-$(pwd)/gpu_artifacts/decision_binding_content}"
SSH_KEY="${6:-}"; SSH_PORT="${7:-}"; MODE="${8:-complete}"
[[ "$MODE" =~ ^(canary|complete)$ ]] || { echo "invalid fetch mode" >&2; exit 2; }
SSH_ARGS=(-o StrictHostKeyChecking=accept-new)
[[ -z "$SSH_KEY" ]] || SSH_ARGS+=(-i "$SSH_KEY")
[[ -z "$SSH_PORT" ]] || SSH_ARGS+=(-p "$SSH_PORT")
DEST="$LOCAL_DIR/$MODEL_SLUG/$RUN_ID/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$DEST"
rsync -az --exclude '/activation_shards/' --exclude '/shards/' \
  -e "ssh ${SSH_ARGS[*]}" \
  "$REMOTE:$REMOTE_DIR/results/decision_binding_content_runs/$MODEL_SLUG/$RUN_ID/" "$DEST/"
python3 -m interface_formatting_study.decision_binding_content_cli verify \
  --run-root "$DEST" --run-id "$RUN_ID" --mode "$MODE"
echo "Fetched and verified $DEST"
