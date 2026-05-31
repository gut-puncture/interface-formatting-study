#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/fetch_interface_formatting_study_artifacts.sh <user@host> [remote_dir] [local_dir] [ssh_key] [ssh_port]

Fetches only compact INTERFACE_FORMATTING_STUDY analysis bundles from the GPU host.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 1 ]]; then
  usage
  exit 0
fi

REMOTE="${1:?remote user@host is required}"
REMOTE_DIR="${2:-/home/ubuntu/interface_formatting_study}"
LOCAL_DIR="${3:-$(pwd)/gpu_artifacts}"
SSH_KEY="${4:-}"
SSH_PORT="${5:-}"

SSH_ARGS=(-o StrictHostKeyChecking=accept-new)
if [[ -n "$SSH_KEY" ]]; then
  SSH_ARGS+=(-i "$SSH_KEY")
fi
if [[ -n "$SSH_PORT" ]]; then
  SSH_ARGS+=(-p "$SSH_PORT")
fi

RSYNC_RSH=(ssh "${SSH_ARGS[@]}")
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DEST="${LOCAL_DIR}/${STAMP}"
mkdir -p "$DEST"

echo "Fetching compact INTERFACE_FORMATTING_STUDY artifacts into ${DEST}"
rsync -az \
  -e "${RSYNC_RSH[*]}" \
  --include='artifacts/' \
  --include='artifacts/interface_formatting_study_analysis_*.tar.gz' \
  --include='artifacts/interface_formatting_study_focused_mechanistic_*.tar.gz' \
  --include='artifacts/SHA256SUMS_*.txt' \
  --include='artifacts/SHA256SUMS_focused_mechanistic_*.txt' \
  --include='run_logs/' \
  --include='run_logs/interface_formatting_study_gpu_*.log' \
  --exclude='*' \
  "${REMOTE}:${REMOTE_DIR}/" "${DEST}/"

shopt -s nullglob
checksum_files=("${DEST}"/artifacts/SHA256SUMS_*.txt)
if (( ${#checksum_files[@]} == 0 )); then
  echo "No SHA256SUMS files were fetched" >&2
  exit 1
fi

for checksum in "${checksum_files[@]}"; do
  echo "Verifying $(basename "$checksum")"
  while read -r expected path; do
    [[ -z "${expected:-}" || -z "${path:-}" ]] && continue
    local_path="${DEST}/${path}"
    if [[ "$path" == artifacts/* ]]; then
      local_path="${DEST}/${path}"
    fi
    if [[ ! -f "$local_path" ]]; then
      if [[ "$(basename "$path")" == interface_formatting_study_*.tar.gz ]]; then
        local_path="${DEST}/artifacts/$(basename "$path")"
      else
        echo "Skipping remote-only checksum entry not fetched separately: ${path}"
        continue
      fi
    fi
    actual="$(shasum -a 256 "$local_path" | awk '{print $1}')"
    if [[ "$actual" != "$expected" ]]; then
      echo "Checksum mismatch for ${local_path}" >&2
      echo "expected ${expected}" >&2
      echo "actual   ${actual}" >&2
      exit 1
    fi
  done < "$checksum"
done

echo "Fetched and verified compact artifacts in ${DEST}"
