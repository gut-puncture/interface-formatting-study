# Reproducibility

This repository contains the code, active dataset file, compact final result
tables, figures, and paper source for the interface-formatting experiment.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Recreate Paper Tables And Figures From Included Results

```bash
python3 analysis/make_paper_assets.py
cd paper
tectonic -X compile main.tex --outdir build
```

## Re-run The Focused Experiment

The main run used `Qwen/Qwen2.5-1.5B-Instruct` with the configuration in
`configs/default.yaml`. On a CUDA GPU host:

```bash
scripts/run_interface_formatting_study_focused_mechanistic_gpu.sh
```

The script writes behavioral scores, conflict pairs, attention diagnostics,
focused controls, tables, figures, and a manifest under `results/`.

## Included And Excluded Artifacts

Included:

- `mmlu_20_wrapper_robustness_60000.jsonl`: active wrapper dataset.
- `results/`: compact final outputs used by the paper.
- `paper/`: LaTeX source, tables, and figures.
- `src/`, `tests/`, `configs/`, `scripts/`, `analysis/`: reproducibility code.

Excluded:

- model weights and local caches,
- virtual environments,
- raw GPU logs,
- raw non-default run archives.
