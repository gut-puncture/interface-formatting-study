#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_interface_formatting_study_focused_mechanistic_gpu.sh

Run this on the Ubuntu GPU host from the INTERFACE_FORMATTING_STUDY repo root.

Optional environment:
  INTERFACE_FORMATTING_STUDY_ROOT=/home/ubuntu/interface_formatting_study
  INTERFACE_FORMATTING_STUDY_RUN_ID=20260531T000000Z
  INTERFACE_FORMATTING_STUDY_CONFIG=configs/default.yaml
  INTERFACE_FORMATTING_STUDY_SMOKE_LIMIT=5
  INTERFACE_FORMATTING_STUDY_SMOKE_MODEL=Qwen/Qwen2.5-0.5B-Instruct
  INTERFACE_FORMATTING_STUDY_REQUIRE_CUDA=1
  INTERFACE_FORMATTING_STUDY_SKIP_TESTS=0
  INTERFACE_FORMATTING_STUDY_SKIP_FULL_BEHAVIORAL=0
  INTERFACE_FORMATTING_STUDY_CLEAN_FOCUSED=0
  INTERFACE_FORMATTING_STUDY_FOCUSED_SPLIT=validation
  INTERFACE_FORMATTING_STUDY_FOCUSED_CAP=300
  INTERFACE_FORMATTING_STUDY_FOCUSED_LAYERS=2,4,6,8,10,12,14,16
  INTERFACE_FORMATTING_STUDY_FOCUSED_ANCHORS=options_end,all_option_ends
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ROOT_DIR="${INTERFACE_FORMATTING_STUDY_ROOT:-$(pwd)}"
CONFIG="${INTERFACE_FORMATTING_STUDY_CONFIG:-configs/default.yaml}"
RUN_ID="${INTERFACE_FORMATTING_STUDY_RUN_ID:-focused_mechanistic_$(date -u +%Y%m%dT%H%M%SZ)}"
SMOKE_LIMIT="${INTERFACE_FORMATTING_STUDY_SMOKE_LIMIT:-5}"
SMOKE_MODEL="${INTERFACE_FORMATTING_STUDY_SMOKE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
REQUIRE_CUDA="${INTERFACE_FORMATTING_STUDY_REQUIRE_CUDA:-1}"
SKIP_TESTS="${INTERFACE_FORMATTING_STUDY_SKIP_TESTS:-0}"
SKIP_FULL_BEHAVIORAL="${INTERFACE_FORMATTING_STUDY_SKIP_FULL_BEHAVIORAL:-0}"
CLEAN_FOCUSED="${INTERFACE_FORMATTING_STUDY_CLEAN_FOCUSED:-0}"
FOCUSED_SPLIT="${INTERFACE_FORMATTING_STUDY_FOCUSED_SPLIT:-validation}"
FOCUSED_CAP="${INTERFACE_FORMATTING_STUDY_FOCUSED_CAP:-300}"
FOCUSED_LAYERS="${INTERFACE_FORMATTING_STUDY_FOCUSED_LAYERS:-2,4,6,8,10,12,14,16}"
FOCUSED_ANCHORS="${INTERFACE_FORMATTING_STUDY_FOCUSED_ANCHORS:-options_end,all_option_ends}"

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

echo "INTERFACE_FORMATTING_STUDY focused mechanistic GPU run ${RUN_ID}"
echo "Repo: ${ROOT_DIR}"
echo "Config: ${CONFIG}"
echo "Focused split/cap: ${FOCUSED_SPLIT}/${FOCUSED_CAP}"
echo "Focused layers: ${FOCUSED_LAYERS}"
echo "Focused anchors: ${FOCUSED_ANCHORS}"
date -u +"UTC start: %Y-%m-%dT%H:%M:%SZ"

if [[ "$CLEAN_FOCUSED" == "1" ]]; then
  run_step "Clean previous focused generated outputs" rm -f \
    results/processed/attention_diagnostics.parquet \
    results/processed/vanilla_convergence.parquet \
    results/processed/focused_patching_controls.parquet \
    results/tables/table_attention_diagnostics.csv \
    results/tables/table_vanilla_convergence.csv \
    results/tables/table_focused_controls.csv \
    results/tables/table3_semantic_patching.csv \
    results/tables/table3_semantic_patching.tex \
    paper/tables/table3_semantic_patching.csv \
    paper/tables/table3_semantic_patching.tex \
    results/metadata/experiment_manifest.json
fi

run_step "System snapshot" bash -c '
  uname -a
  python3 --version || true
  df -h .
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
'

if ! python3 -m venv .venv >/tmp/interface_formatting_study_focused_venv_check.log 2>&1; then
  echo "python3 -m venv failed; installing python3-venv and retrying"
  cat /tmp/interface_formatting_study_focused_venv_check.log || true
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
    raise SystemExit("CUDA is required for this focused run, but torch.cuda.is_available() is false")
PY

if [[ "$SKIP_TESTS" != "1" ]]; then
  run_step "Remote unit tests" pytest -q -p no:cacheprovider
  run_step "Compile source and tests" python3 -m compileall src tests
fi

run_step "Dataset audit" interface_formatting_study --config "$CONFIG" audit-dataset
run_step "Wrapper audit" interface_formatting_study --config "$CONFIG" wrapper-audit

run_step "Focused guardrails" python - <<'PY'
import json
from pathlib import Path

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
print("verified focused wrapper policy and active dataset dimensions")
PY

run_step "Behavioral smoke" interface_formatting_study --config "$CONFIG" behavioral-smoke --limit "$SMOKE_LIMIT" --model "$SMOKE_MODEL"
run_step "Smoke conflicts" interface_formatting_study --config "$CONFIG" conflicts --behavioral results/smoke/raw/behavioral_smoke.parquet
run_step "Smoke figures" interface_formatting_study --config "$CONFIG" figures --behavioral results/smoke/raw/behavioral_smoke.parquet
run_step "Smoke tables" interface_formatting_study --config "$CONFIG" tables --behavioral results/smoke/raw/behavioral_smoke.parquet

if [[ -s results/raw/behavioral_scores.parquet ]]; then
  run_step "Reuse existing behavioral scores" python - <<'PY'
from pathlib import Path
import json
import pandas as pd

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
behavioral = Path("results/raw/behavioral_scores.parquet")
frame = pd.read_parquet(behavioral)
if len(frame) != int(audit["num_rows"]):
    raise SystemExit(f"existing behavioral table has {len(frame)} rows; expected {audit['num_rows']}")
wrappers = set(frame["wrapper_name"].astype(str))
if wrappers != expected_wrappers:
    raise SystemExit(f"existing behavioral table has unexpected wrappers: {sorted(wrappers)}")
if set(frame["wrapper_category"].astype(str)) != {"pure_interface"}:
    raise SystemExit("existing behavioral table is not verified pure-interface only")
if "same_wrapper_redaction" not in set(frame["content_free_calibration_kind"].astype(str)):
    raise SystemExit("existing behavioral table lacks same-wrapper redaction calibration rows")
print(f"reusing {behavioral} with {len(frame)} rows")
PY
elif [[ "$SKIP_FULL_BEHAVIORAL" == "1" ]]; then
  echo "INTERFACE_FORMATTING_STUDY_SKIP_FULL_BEHAVIORAL=1 set, but results/raw/behavioral_scores.parquet is missing" >&2
  exit 1
else
  run_step "Full behavioral scoring" interface_formatting_study --config "$CONFIG" behavioral --resume
fi

if [[ ! -s results/processed/conflict_pairs.parquet ]]; then
  run_step "Full conflict construction" interface_formatting_study --config "$CONFIG" conflicts --behavioral results/raw/behavioral_scores.parquet
else
  if ! run_step "Validate existing conflict pairs" python - <<'PY'
import pandas as pd

pairs = pd.read_parquet("results/processed/conflict_pairs.parquet")
if pairs.empty:
    raise SystemExit("existing conflict_pairs.parquet is empty")
required = {
    "wrapper_category_subset",
    "clean_content_free_calibration_kind",
    "corrupt_content_free_calibration_kind",
    "question",
    "choices",
    "split",
}
missing = required - set(pairs.columns)
if missing:
    raise SystemExit(f"existing conflict_pairs.parquet missing columns: {sorted(missing)}")
if set(pairs["wrapper_category_subset"].astype(str)) != {"pure_interface"}:
    raise SystemExit("existing conflict pairs are not pure-interface only")
for column in ("clean_content_free_calibration_kind", "corrupt_content_free_calibration_kind"):
    if set(pairs[column].astype(str)) != {"same_wrapper_redaction"}:
        raise SystemExit(f"existing conflict pairs have invalid {column}")
if "validation" not in set(pairs["split"].astype(str)):
    raise SystemExit("existing conflict pairs have no validation split")
print(f"reusing conflict_pairs.parquet with {len(pairs)} rows")
PY
  then
    run_step "Regenerate conflict pairs after failed validation" interface_formatting_study --config "$CONFIG" conflicts --behavioral results/raw/behavioral_scores.parquet
  fi
fi

run_step "Final behavioral figures" interface_formatting_study --config "$CONFIG" figures --behavioral results/raw/behavioral_scores.parquet
run_step "Remove stale legacy paper-path tables" rm -f \
  results/tables/table3_semantic_patching.csv \
  results/tables/table3_semantic_patching.tex \
  paper/tables/table3_semantic_patching.csv \
  paper/tables/table3_semantic_patching.tex
run_step "Final behavioral tables" interface_formatting_study --config "$CONFIG" tables --behavioral results/raw/behavioral_scores.parquet
run_step "Attention diagnostics" interface_formatting_study --config "$CONFIG" attention-diagnostics \
  --conflicts results/processed/conflict_pairs.parquet \
  --behavioral results/raw/behavioral_scores.parquet \
  --split "$FOCUSED_SPLIT" \
  --cap "$FOCUSED_CAP" \
  --anchors "$FOCUSED_ANCHORS"
run_step "Vanilla convergence diagnostics" interface_formatting_study --config "$CONFIG" vanilla-convergence \
  --conflicts results/processed/conflict_pairs.parquet \
  --split "$FOCUSED_SPLIT" \
  --cap "$FOCUSED_CAP" \
  --layers "$FOCUSED_LAYERS" \
  --anchors "$FOCUSED_ANCHORS"
run_step "Focused causal controls" interface_formatting_study --config "$CONFIG" focused-patching-controls \
  --conflicts results/processed/conflict_pairs.parquet \
  --split "$FOCUSED_SPLIT" \
  --cap "$FOCUSED_CAP" \
  --layers "$FOCUSED_LAYERS" \
  --anchors "$FOCUSED_ANCHORS"
run_step "Experiment manifest" interface_formatting_study --config "$CONFIG" write-manifest
run_step "Final deliverable validation" interface_formatting_study --config "$CONFIG" deliverables --no-compile

LOG_SNAPSHOT="run_logs/interface_formatting_study_gpu_${RUN_ID}_snapshot.log"
cp "$LOG_FILE" "$LOG_SNAPSHOT"
run_step "Package compact focused bundle" bash -c "
  bundle='artifacts/interface_formatting_study_focused_mechanistic_${RUN_ID}.tar.gz'
  tar -czf \"\$bundle\" \
    results/metadata/dataset_audit.json \
    results/metadata/wrapper_audit.csv \
    results/metadata/split_manifest.json \
    results/metadata/experiment_manifest.json \
    results/raw/behavioral_scores.parquet \
    results/processed/conflict_pairs.parquet \
    results/processed/attention_diagnostics.parquet \
    results/processed/vanilla_convergence.parquet \
    results/processed/focused_patching_controls.parquet \
    results/figures/wrapper_accuracy_heatmap.png \
    results/figures/format_conflict_distribution.png \
    results/tables/table1_dataset_wrapper_audit.csv \
    results/tables/table2_behavioral_conflicts.csv \
    results/tables/table_attention_diagnostics.csv \
    results/tables/table_vanilla_convergence.csv \
    results/tables/table_focused_controls.csv \
    paper/main.tex \
    paper/references.bib \
    paper/figures \
    paper/tables \
    \"${LOG_SNAPSHOT}\"
  sha256sum \"\$bundle\" > artifacts/SHA256SUMS_focused_mechanistic_${RUN_ID}.txt
  for path in \
    results/metadata/dataset_audit.json \
    results/metadata/experiment_manifest.json \
    results/raw/behavioral_scores.parquet \
    results/processed/conflict_pairs.parquet \
    results/processed/attention_diagnostics.parquet \
    results/processed/vanilla_convergence.parquet \
    results/processed/focused_patching_controls.parquet \
    results/tables/table_focused_controls.csv; do
      sha256sum \"\$path\" >> artifacts/SHA256SUMS_focused_mechanistic_${RUN_ID}.txt
  done
  ls -lh \"\$bundle\" artifacts/SHA256SUMS_focused_mechanistic_${RUN_ID}.txt
"

date -u +"UTC finish: %Y-%m-%dT%H:%M:%SZ"
echo "INTERFACE_FORMATTING_STUDY focused mechanistic GPU run complete. Fetch artifacts/interface_formatting_study_focused_mechanistic_${RUN_ID}.tar.gz and artifacts/SHA256SUMS_focused_mechanistic_${RUN_ID}.txt"
