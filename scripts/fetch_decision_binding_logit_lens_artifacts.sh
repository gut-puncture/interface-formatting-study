#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 2 ]]; then
  echo "Usage: $0 <user@host> <semantic-run-id> [remote-dir] [local-dir] [ssh-key] [ssh-port] [partial|startup|complete]"
  exit 0
fi

REMOTE="$1"
RUN_ID="$2"
REMOTE_DIR="${3:-/home/ubuntu/interface_formatting_study}"
LOCAL_DIR="${4:-$(pwd)/gpu_artifacts/decision_binding_logit_lens}"
SSH_KEY="${5:-}"
SSH_PORT="${6:-}"
MODE="${7:-complete}"
[[ "$MODE" =~ ^(partial|startup|complete)$ ]] || { echo "invalid fetch mode" >&2; exit 2; }

SSH_ARGS=(-o StrictHostKeyChecking=accept-new)
[[ -z "$SSH_KEY" ]] || SSH_ARGS+=(-i "$SSH_KEY")
[[ -z "$SSH_PORT" ]] || SSH_ARGS+=(-p "$SSH_PORT")
MODEL_SLUG="mistral-7b-instruct-v0.3"
DEST="$LOCAL_DIR/$MODEL_SLUG/$RUN_ID/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$DEST"

rsync -az -e "ssh ${SSH_ARGS[*]}" \
  "$REMOTE:$REMOTE_DIR/results/decision_binding_logit_lens_runs/$MODEL_SLUG/$RUN_ID/" \
  "$DEST/"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "missing executable Python runtime: $PYTHON_BIN" >&2; exit 2; }
"$PYTHON_BIN" -m interface_formatting_study.decision_binding_logit_lens_cli verify \
  --run-root "$DEST" \
  --run-id "$RUN_ID" \
  --mode "$MODE"
echo "Fetched and verified Mistral logit-lens artifacts in $DEST"
