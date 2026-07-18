#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 3 ]]; then
  echo "Usage: $0 <profile> <functional|full> <prepared-bundle>"
  exit 0
fi

PROFILE="$1"; MODE="$2"; BUNDLE="$3"
[[ "$PROFILE" =~ ^(qwen|phi|mistral)$ ]] || { echo "invalid profile" >&2; exit 2; }
[[ "$MODE" =~ ^(functional|full)$ ]] || { echo "invalid mode" >&2; exit 2; }

ARGS=(
  run-model
  --profile "$PROFILE"
  --bundle "$BUNDLE"
  --batch-size "${BATCH_SIZE:-16}"
  --max-batch-tokens "${MAX_BATCH_TOKENS:-24000}"
  --capture-chunk-size "${CAPTURE_CHUNK_SIZE:-64}"
  --local-files-only
)
if [[ "$MODE" == "functional" ]]; then
  ARGS+=(--canary-items "${CANARY_ITEMS:-8}")
fi

exec python -m interface_formatting_study.decision_binding_content_cli "${ARGS[@]}"
