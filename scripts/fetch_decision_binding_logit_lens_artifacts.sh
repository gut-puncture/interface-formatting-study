#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 3 ]]; then
  echo "Usage: $0 <mistral|phi|qwen> <user@host> <semantic-run-id> [remote-dir] [local-dir] [ssh-key] [ssh-port] [partial|startup|complete]"
  exit 0
fi

PROFILE="$1"
REMOTE="$2"
RUN_ID="$3"
REMOTE_DIR="${4:-/home/ubuntu/interface_formatting_study}"
LOCAL_DIR="${5:-$(pwd)/gpu_artifacts/decision_binding_logit_lens}"
SSH_KEY="${6:-}"
SSH_PORT="${7:-}"
MODE="${8:-complete}"
[[ "$PROFILE" =~ ^(mistral|phi|qwen)$ ]] || { echo "invalid profile" >&2; exit 2; }
[[ "$MODE" =~ ^(partial|startup|complete)$ ]] || { echo "invalid fetch mode" >&2; exit 2; }

SSH_ARGS=(-o StrictHostKeyChecking=accept-new)
[[ -z "$SSH_KEY" ]] || SSH_ARGS+=(-i "$SSH_KEY")
[[ -z "$SSH_PORT" ]] || SSH_ARGS+=(-p "$SSH_PORT")
case "$PROFILE" in
  mistral) MODEL_SLUG="mistral-7b-instruct-v0.3" ;;
  phi) MODEL_SLUG="phi-3.5-mini-instruct" ;;
  qwen) MODEL_SLUG="qwen2.5-1.5b-instruct" ;;
esac
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
echo "Fetched and verified $PROFILE logit-lens artifacts in $DEST"
