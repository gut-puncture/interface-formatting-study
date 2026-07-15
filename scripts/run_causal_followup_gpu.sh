#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/run_causal_followup_gpu.sh <profile> <functional|profiling|full> [design]

Runs one pinned model through the resumable controlled causal follow-up. Start it
through control_causal_followup_gpu.sh on paid hardware. Environment knobs:
BATCH_SIZE (32), MAX_BATCH_TOKENS (40000), CHECKPOINT_SIZE (256), CANARY_ITEMS (8 or 32).
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 2 ]]; then
  usage
  exit 0
fi

PROFILE="$1"
MODE="$2"
DESIGN="${3:-artifacts/causal_followup/v2_source_preserving/design_train_validation.parquet}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS:-40000}"
CHECKPOINT_SIZE="${CHECKPOINT_SIZE:-256}"

COMMON=(
  run
  --profile "$PROFILE"
  --design "$DESIGN"
  --batch-size "$BATCH_SIZE"
  --max-batch-tokens "$MAX_BATCH_TOKENS"
  --checkpoint-size "$CHECKPOINT_SIZE"
  --local-files-only
)

case "$MODE" in
  functional)
    COMMON+=(--canary --canary-name functional --canary-items "${CANARY_ITEMS:-8}" --profile-timings)
    ;;
  profiling)
    COMMON+=(--canary --canary-name profiling --canary-items "${CANARY_ITEMS:-32}" --profile-timings)
    ;;
  full)
    ;;
  *)
    echo "mode must be functional, profiling, or full" >&2
    exit 2
    ;;
esac

exec python -m interface_formatting_study.causal_cli "${COMMON[@]}"
