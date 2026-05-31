#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_interface_formatting_study_gpu.sh

Run this on the Ubuntu GPU host from the INTERFACE_FORMATTING_STUDY repo root.

Optional environment:
  INTERFACE_FORMATTING_STUDY_ROOT=/root/interface_formatting_study
  INTERFACE_FORMATTING_STUDY_RUN_ID=20260529T000000Z
  INTERFACE_FORMATTING_STUDY_CONFIG=configs/default.yaml
  INTERFACE_FORMATTING_STUDY_SMOKE_LIMIT=5
  INTERFACE_FORMATTING_STUDY_SMOKE_MODEL=Qwen/Qwen2.5-0.5B-Instruct
  INTERFACE_FORMATTING_STUDY_REQUIRE_CUDA=1
  INTERFACE_FORMATTING_STUDY_SKIP_TESTS=0
  INTERFACE_FORMATTING_STUDY_SKIP_FULL=0
  INTERFACE_FORMATTING_STUDY_CLEAN_RESULTS=0
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ROOT_DIR="${INTERFACE_FORMATTING_STUDY_ROOT:-$(pwd)}"
CONFIG="${INTERFACE_FORMATTING_STUDY_CONFIG:-configs/default.yaml}"
RUN_ID="${INTERFACE_FORMATTING_STUDY_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
SMOKE_LIMIT="${INTERFACE_FORMATTING_STUDY_SMOKE_LIMIT:-5}"
SMOKE_MODEL="${INTERFACE_FORMATTING_STUDY_SMOKE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
REQUIRE_CUDA="${INTERFACE_FORMATTING_STUDY_REQUIRE_CUDA:-1}"
SKIP_TESTS="${INTERFACE_FORMATTING_STUDY_SKIP_TESTS:-0}"
SKIP_FULL="${INTERFACE_FORMATTING_STUDY_SKIP_FULL:-0}"
CLEAN_RESULTS="${INTERFACE_FORMATTING_STUDY_CLEAN_RESULTS:-0}"

cd "$ROOT_DIR"
mkdir -p run_logs artifacts .cache/huggingface .cache/torch .cache/matplotlib
LOG_FILE="${ROOT_DIR}/run_logs/interface_formatting_study_gpu_${RUN_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

export HF_HOME="${HF_HOME:-${ROOT_DIR}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${ROOT_DIR}/.cache/torch}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT_DIR}/.cache/matplotlib}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1

run_step() {
  local name="$1"
  shift
  echo
  echo "== ${name} =="
  "$@"
}

apt_install() {
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "apt-get is unavailable; cannot install missing system packages" >&2
    return 1
  fi
  if [[ "$(id -u)" == "0" ]]; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y "$@"
  else
    sudo apt-get update
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "$@"
  fi
}

echo "INTERFACE_FORMATTING_STUDY GPU run ${RUN_ID}"
echo "Repo: ${ROOT_DIR}"
echo "Config: ${CONFIG}"
echo "Smoke limit: ${SMOKE_LIMIT}"
echo "Smoke model: ${SMOKE_MODEL}"
echo "HF cache: ${HF_HOME}"
date -u +"UTC start: %Y-%m-%dT%H:%M:%SZ"

if [[ "$CLEAN_RESULTS" == "1" ]]; then
  run_step "Clean previous generated outputs" rm -rf results/raw results/processed results/figures results/tables results/smoke
fi

run_step "System snapshot" bash -c '
  uname -a
  python3 --version || true
  df -h .
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
'

if ! python3 -m venv .venv >/tmp/interface_formatting_study_venv_check.log 2>&1; then
  echo "python3 -m venv failed; installing python3-venv and retrying"
  cat /tmp/interface_formatting_study_venv_check.log || true
  apt_install python3-venv
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

run_step "Install Python dependencies" bash -c '
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install -e ".[dev]"
'

run_step "CUDA preflight" python - <<'PY'
import os
import torch

print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device count", torch.cuda.device_count())
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        print(f"cuda:{idx}", props.name, f"{props.total_memory / (1024 ** 3):.1f} GiB")
elif os.environ.get("INTERFACE_FORMATTING_STUDY_REQUIRE_CUDA", "1") == "1":
    raise SystemExit("CUDA is required for this run, but torch.cuda.is_available() is false")
PY

if [[ "$SKIP_TESTS" != "1" ]]; then
  run_step "Remote unit tests" pytest -q -p no:cacheprovider
fi

run_step "Dataset audit" interface_formatting_study --config "$CONFIG" audit-dataset
run_step "Wrapper audit" interface_formatting_study --config "$CONFIG" wrapper-audit

run_step "Audit guardrails" python - <<'PY'
import json
from pathlib import Path

from interface_formatting_study.calibration import content_free_contains_answer_text, make_content_free_prompt_with_metadata
from interface_formatting_study.experiment import prepare_dataset
from interface_formatting_study.utils import read_yaml

expected_wrappers = {
    "csv_inline",
    "graphql_query",
    "html_form",
    "ini_file",
    "key_equals",
    "protobuf_msg",
    "shell_heredoc",
    "toml_config",
}

audit = json.loads(Path("results/metadata/dataset_audit.json").read_text())
wrappers = set(audit["wrapper_counts"])
if wrappers != expected_wrappers:
    raise SystemExit(f"unexpected active wrappers: {sorted(wrappers)}")
if audit["num_rows"] != 24000 or audit["num_items"] != 3000 or audit["num_wrappers"] != 8:
    raise SystemExit(f"unexpected active dataset dimensions: {audit}")
if audit.get("wrapper_policy") != "verified_pure_interface_only":
    raise SystemExit(f"unexpected wrapper policy: {audit.get('wrapper_policy')}")

df, _ = prepare_dataset(read_yaml("configs/default.yaml"))
leaks = []
for row in df.to_dict("records"):
    meta = make_content_free_prompt_with_metadata(row)
    if meta["content_free_calibration_kind"] == "same_wrapper_redaction":
        prompt = str(meta["content_free_prompt"])
        if row.get("question") and str(row["question"]).strip().lower() in prompt.lower():
            leaks.append((row["item_id"], row["wrapper_name"], "question"))
        if content_free_contains_answer_text(prompt, row.get("choices")):
            leaks.append((row["item_id"], row["wrapper_name"], "choice"))
if leaks:
    raise SystemExit(f"content-free redaction leaks detected: {leaks[:5]}")
print("verified active wrappers, dataset dimensions, and content-free redaction guardrails")
PY

run_step "Behavioral smoke" interface_formatting_study --config "$CONFIG" behavioral-smoke --limit "$SMOKE_LIMIT" --model "$SMOKE_MODEL"
run_step "Smoke conflicts" interface_formatting_study --config "$CONFIG" conflicts --behavioral results/smoke/raw/behavioral_smoke.parquet
run_step "Smoke figures" interface_formatting_study --config "$CONFIG" figures --behavioral results/smoke/raw/behavioral_smoke.parquet
run_step "Smoke tables" interface_formatting_study --config "$CONFIG" tables --behavioral results/smoke/raw/behavioral_smoke.parquet

run_step "Smoke artifact checks" python - <<PY
from pathlib import Path
import pandas as pd

limit = int("${SMOKE_LIMIT}")
smoke = Path("results/smoke/raw/behavioral_smoke.parquet")
if not smoke.exists():
    raise SystemExit("missing smoke behavioral parquet")
frame = pd.read_parquet(smoke)
if len(frame) != limit:
    raise SystemExit(f"smoke row count {len(frame)} != expected {limit}")
required = {"wrapper_category", "content_free_calibration_kind", "cal_margin", "cal_correct"}
missing = required - set(frame.columns)
if missing:
    raise SystemExit(f"smoke output missing columns: {sorted(missing)}")
if Path("results/raw/behavioral_scores.parquet").exists() and "${SKIP_FULL}" == "1":
    raise SystemExit("final behavioral file already exists during smoke-only mode")
print(f"smoke passed with {len(frame)} rows")
PY

if [[ "$SKIP_FULL" == "1" ]]; then
  echo "INTERFACE_FORMATTING_STUDY_SKIP_FULL=1 set; stopping after smoke gate."
  exit 0
fi

run_step "Full behavioral scoring" interface_formatting_study --config "$CONFIG" behavioral --resume
run_step "Full conflict construction" interface_formatting_study --config "$CONFIG" conflicts --behavioral results/raw/behavioral_scores.parquet
run_step "Semantic-anchor patching sweep" interface_formatting_study --config "$CONFIG" patching-sweep --conflicts results/processed/conflict_pairs.parquet
run_step "Final figures" interface_formatting_study --config "$CONFIG" figures --behavioral results/raw/behavioral_scores.parquet
run_step "Final tables" interface_formatting_study --config "$CONFIG" tables --behavioral results/raw/behavioral_scores.parquet
run_step "Experiment manifest" interface_formatting_study --config "$CONFIG" write-manifest
run_step "Final deliverable validation" interface_formatting_study --config "$CONFIG" deliverables --no-compile

LOG_SNAPSHOT="run_logs/interface_formatting_study_gpu_${RUN_ID}_snapshot.log"
cp "$LOG_FILE" "$LOG_SNAPSHOT"
run_step "Package compact analysis bundle" bash -c "
  bundle='artifacts/interface_formatting_study_analysis_${RUN_ID}.tar.gz'
  tar -czf \"\$bundle\" \
    results/metadata/dataset_audit.json \
    results/metadata/wrapper_audit.csv \
    results/metadata/split_manifest.json \
    results/metadata/selected_location.json \
    results/metadata/experiment_manifest.json \
    results/raw/behavioral_scores.parquet \
    results/processed/conflict_pairs.parquet \
    results/processed/item_outcomes.csv \
    results/processed/patching_results.parquet \
    results/figures \
    results/tables \
    paper \
    \"${LOG_SNAPSHOT}\"
  sha256sum \"\$bundle\" > artifacts/SHA256SUMS_${RUN_ID}.txt
  for path in \
    results/metadata/dataset_audit.json \
    results/metadata/experiment_manifest.json \
    results/raw/behavioral_scores.parquet \
    results/processed/conflict_pairs.parquet \
    results/processed/patching_results.parquet \
    results/tables/table3_semantic_patching.csv; do
      sha256sum \"\$path\" >> artifacts/SHA256SUMS_${RUN_ID}.txt
  done
  ls -lh \"\$bundle\" artifacts/SHA256SUMS_${RUN_ID}.txt
"

date -u +"UTC finish: %Y-%m-%dT%H:%M:%SZ"
echo "INTERFACE_FORMATTING_STUDY GPU run complete. Fetch artifacts/interface_formatting_study_analysis_${RUN_ID}.tar.gz and artifacts/SHA256SUMS_${RUN_ID}.txt"
