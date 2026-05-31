# Reproducibility Bundle

This anonymized bundle contains the code, configuration, active dataset file, compact result artifacts, figures, tables, and metadata needed to reproduce the paper's main results.

## Contents

- `src/`, `tests/`, `analysis/`, `configs/`, `scripts/`: experiment code and checks.
- `mmlu_20_wrapper_robustness_60000.jsonl`: source wrapper dataset used by the active experiment.
- `results/`: compact result tables, parquet outputs, metadata manifests, and generated figures from the final focused run.
- `paper/`: LaTeX source, tables, and figures used to build the submitted PDF.

The bundle excludes model weights, virtual environments, raw GPU caches, and raw historical logs.

## Reproduce Main Tables And Figures

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
python3 analysis/make_paper_assets.py
cd paper
tectonic -X compile main.tex --outdir build
```

The final focused mechanistic run used `Qwen/Qwen2.5-1.5B-Instruct`, a validation cap of 300 conflict pairs, and one NVIDIA RTX A6000 48GB-class GPU.
