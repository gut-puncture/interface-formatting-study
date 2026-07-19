#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/run_decision_binding_logit_lens_gpu.sh <startup|full> <prepared-bundle> <token-audit>

Runs the pinned Mistral logit-lens scorer. Runtime knobs are BATCH_SIZE (8),
MAX_BATCH_TOKENS (24000), CAPTURE_CHUNK_SIZE (startup 4; full 64),
STARTUP_ITEMS (8), and the optional MAX_CHUNKS_THIS_INVOCATION stop/resume seam.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 3 ]]; then
  usage
  exit 0
fi

MODE="$1"
BUNDLE="$2"
TOKEN_AUDIT="$3"
[[ "$MODE" =~ ^(startup|full)$ ]] || { echo "mode must be startup or full" >&2; exit 2; }
if [[ -n "${CAPTURE_CHUNK_SIZE:-}" ]]; then
  CHUNK_SIZE="$CAPTURE_CHUNK_SIZE"
elif [[ "$MODE" == "startup" ]]; then
  CHUNK_SIZE="${STARTUP_CAPTURE_CHUNK_SIZE:-4}"
else
  CHUNK_SIZE="${FULL_CAPTURE_CHUNK_SIZE:-64}"
fi

ARGS=(
  run-model
  --profile "mistral"
  --bundle "$BUNDLE"
  --token-audit "$TOKEN_AUDIT"
  --batch-size "${BATCH_SIZE:-8}"
  --max-batch-tokens "${MAX_BATCH_TOKENS:-24000}"
  --capture-chunk-size "$CHUNK_SIZE"
  --local-files-only
)
if [[ "$MODE" == "startup" ]]; then
  ARGS+=(--startup-items "${STARTUP_ITEMS:-8}")
fi
if [[ -n "${MAX_CHUNKS_THIS_INVOCATION:-}" ]]; then
  ARGS+=(--max-chunks-this-invocation "$MAX_CHUNKS_THIS_INVOCATION")
fi

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "missing executable Python runtime: $PYTHON_BIN" >&2; exit 2; }
exec "$PYTHON_BIN" -m interface_formatting_study.decision_binding_logit_lens_cli "${ARGS[@]}"
