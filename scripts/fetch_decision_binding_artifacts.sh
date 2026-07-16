#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 3 ]]; then
  echo "Usage: $0 <user@host> <model-slug> <semantic-run-id> [remote-dir] [local-dir] [ssh-key] [ssh-port] [complete|partial]"
  exit 0
fi

REMOTE="$1"; MODEL_SLUG="$2"; RUN_ID="$3"
REMOTE_DIR="${4:-/home/ubuntu/interface_formatting_study}"
LOCAL_DIR="${5:-$(pwd)/gpu_artifacts/decision_binding}"
SSH_KEY="${6:-}"; SSH_PORT="${7:-}"; MODE="${8:-complete}"
[[ "$MODE" =~ ^(complete|partial)$ ]] || { echo "invalid fetch mode" >&2; exit 2; }
SSH_ARGS=(-o StrictHostKeyChecking=accept-new)
[[ -z "$SSH_KEY" ]] || SSH_ARGS+=(-i "$SSH_KEY")
[[ -z "$SSH_PORT" ]] || SSH_ARGS+=(-p "$SSH_PORT")
DEST="$LOCAL_DIR/$MODEL_SLUG/$RUN_ID/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$DEST"
rsync -az -e "ssh ${SSH_ARGS[*]}" \
  "$REMOTE:$REMOTE_DIR/results/decision_binding_runs/$MODEL_SLUG/$RUN_ID/" "$DEST/"
python3 "$(dirname "$0")/verify_decision_binding_artifacts.py" "$DEST" "$RUN_ID" "$MODEL_SLUG" "$MODE"
echo "Fetched and verified $DEST"
