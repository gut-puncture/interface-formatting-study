#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/sync_interface_formatting_study_to_gpu.sh <user@host> [remote_dir] [ssh_key] [ssh_port] [active_dataset]

Copies only source, configuration, packaging metadata, and the active dataset
to a GPU host. Existing results, paper assets, analysis, tests, logs, and caches
are never transferred. active_dataset is a project-relative path and defaults
to mmlu_20_wrapper_robustness_60000.jsonl.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 1 ]]; then
  usage
  exit 0
fi

REMOTE="${1:?remote user@host is required}"
REMOTE_DIR="${2:-/home/ubuntu/interface_formatting_study}"
SSH_KEY="${3:-}"
SSH_PORT="${4:-}"
ACTIVE_DATASET="${5:-mmlu_20_wrapper_robustness_60000.jsonl}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTIVE_DATASET_PATH="${ROOT_DIR}/${ACTIVE_DATASET}"
if [[ ! -f "$ACTIVE_DATASET_PATH" ]]; then
  echo "Active dataset does not exist: ${ACTIVE_DATASET_PATH}" >&2
  exit 2
fi
case "$(cd "$(dirname "$ACTIVE_DATASET_PATH")" && pwd)/$(basename "$ACTIVE_DATASET_PATH")" in
  "${ROOT_DIR}"/*) ;;
  *) echo "active_dataset must be inside ${ROOT_DIR}" >&2; exit 2 ;;
esac

SSH_ARGS=(-o StrictHostKeyChecking=accept-new)
if [[ -n "$SSH_KEY" ]]; then
  SSH_ARGS+=(-i "$SSH_KEY")
fi
if [[ -n "$SSH_PORT" ]]; then
  SSH_ARGS+=(-p "$SSH_PORT")
fi

RSYNC_RSH=(ssh "${SSH_ARGS[@]}")

sync_relative_file() {
  local relative_path="$1"
  (
    cd "$ROOT_DIR"
    rsync -az --relative \
      -e "${RSYNC_RSH[*]}" \
      "./$relative_path" \
      "${REMOTE}:${REMOTE_DIR}/"
  )
}

echo "Creating ${REMOTE_DIR} on ${REMOTE}"
ssh "${SSH_ARGS[@]}" "$REMOTE" "mkdir -p '$REMOTE_DIR'"

echo "Syncing INTERFACE_FORMATTING_STUDY payload from ${ROOT_DIR}"
PAYLOAD_BYTES="$(du -sk \
  "${ROOT_DIR}/src" \
  "${ROOT_DIR}/configs" \
  "${ROOT_DIR}/pyproject.toml" \
  "${ROOT_DIR}/requirements-gpu.lock" \
  "${ROOT_DIR}/README.md" \
  "${ACTIVE_DATASET_PATH}" | awk '{total += $1} END {print total * 1024}')"
echo "Thin payload bytes: ${PAYLOAD_BYTES}"
rsync -az --delete \
  --exclude '/src/*.egg-info/' \
  --exclude '/src/**/__pycache__/' \
  --include '/src/' \
  --include '/src/***' \
  --include '/configs/' \
  --include '/configs/***' \
  --include '/pyproject.toml' \
  --include '/requirements-gpu.lock' \
  --include '/README.md' \
  --include '/SCIENTIFIC_NORTH_STAR.md' \
  --include '/DECISION_BINDING_LOGIT_LENS_RUN_CARD.md' \
  --include '/analysis/' \
  --include '/analysis/analyze_decision_binding_logit_lens.py' \
  --include '/scripts/' \
  --include '/scripts/bootstrap_causal_followup_gpu.sh' \
  --include '/scripts/run_causal_followup_gpu.sh' \
  --include '/scripts/control_causal_followup_gpu.sh' \
  --include '/scripts/run_decision_binding_gpu.sh' \
  --include '/scripts/control_decision_binding_gpu.sh' \
  --include '/scripts/run_decision_binding_content_gpu.sh' \
  --include '/scripts/control_decision_binding_content_gpu.sh' \
  --include '/scripts/fetch_decision_binding_content_artifacts.sh' \
  --include '/scripts/run_decision_binding_logit_lens_gpu.sh' \
  --include '/scripts/control_decision_binding_logit_lens_gpu.sh' \
  --include '/scripts/fetch_decision_binding_logit_lens_artifacts.sh' \
  --include '/scripts/cache_causal_models.py' \
  --exclude '*' \
  -e "${RSYNC_RSH[*]}" \
  "${ROOT_DIR}/" "${REMOTE}:${REMOTE_DIR}/"

sync_relative_file "$ACTIVE_DATASET"
if [[ -f "${ACTIVE_DATASET_PATH}.manifest.json" ]]; then
  sync_relative_file "${ACTIVE_DATASET}.manifest.json"
fi
PREPARED_MANIFEST_PATH="$(dirname "$ACTIVE_DATASET_PATH")/prepared_manifest.json"
if [[ -f "$PREPARED_MANIFEST_PATH" ]]; then
  PREPARED_MANIFEST_RELATIVE="${PREPARED_MANIFEST_PATH#"${ROOT_DIR}/"}"
  sync_relative_file "$PREPARED_MANIFEST_RELATIVE"
fi
APPLICABILITY_PATH="${ACTIVE_DATASET_PATH%.*}.applicability.parquet"
if [[ -f "$APPLICABILITY_PATH" ]]; then
  APPLICABILITY_RELATIVE="${ACTIVE_DATASET%.*}.applicability.parquet"
  sync_relative_file "$APPLICABILITY_RELATIVE"
fi
DECISION_PAIR_PATH="$(dirname "$ACTIVE_DATASET_PATH")/patch_pair_ledger.parquet"
DECISION_MANIFEST_PATH="$(dirname "$ACTIVE_DATASET_PATH")/bundle_manifest.json"
for SIDECAR_PATH in "$DECISION_PAIR_PATH" "$DECISION_MANIFEST_PATH"; do
  if [[ -f "$SIDECAR_PATH" ]]; then
    SIDECAR_RELATIVE="${SIDECAR_PATH#"${ROOT_DIR}/"}"
    sync_relative_file "$SIDECAR_RELATIVE"
  fi
done

echo "Remote payload ready at ${REMOTE}:${REMOTE_DIR}"
