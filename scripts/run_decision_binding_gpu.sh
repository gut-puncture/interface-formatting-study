#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/run_decision_binding_gpu.sh <profile> <functional|full> <bundle> [frozen-run]

Runs one pinned model through the resumable decision-binding experiment. A
confirmation bundle requires the discovery run path as the fourth argument.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 3 ]]; then
  usage
  exit 0
fi

PROFILE="$1"
MODE="$2"
BUNDLE="$3"
FROZEN_RUN="${4:-}"
[[ "$PROFILE" =~ ^(qwen|phi|mistral)$ ]] || { echo "invalid profile" >&2; exit 2; }
[[ "$MODE" =~ ^(functional|full)$ ]] || { echo "invalid mode" >&2; exit 2; }

COMMON=(
  run-model
  --profile "$PROFILE"
  --bundle "$BUNDLE"
  --batch-size "${BATCH_SIZE:-16}"
  --max-batch-tokens "${MAX_BATCH_TOKENS:-24000}"
  --readout-chunk-size "${READOUT_CHUNK_SIZE:-128}"
  --patch-shard-size "${PATCH_SHARD_SIZE:-16}"
  --local-files-only
)
if [[ -n "$FROZEN_RUN" ]]; then
  COMMON+=(--frozen-run "$FROZEN_RUN")
fi
if [[ "$MODE" == "functional" ]]; then
  COMMON+=(--max-pairs "${CANARY_PAIRS:-2}")
fi

exec python -m interface_formatting_study.decision_binding_cli "${COMMON[@]}"
