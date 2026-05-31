# Interface Formatting Study

This repository tests whether interface formatting alone can change how a language model answers the same multiple-choice question.

The final experiment uses exactly eight verified pure-interface wrappers: `csv_inline`, `graphql_query`, `html_form`, `ini_file`, `key_equals`, `protobuf_msg`, `shell_heredoc`, and `toml_config`. Answer-label likelihood is the primary metric, content-free calibration is provenance-tracked, splits are by MMLU item, and mechanistic claims are restricted to focused semantic-anchor diagnostics and controls.

The current paper pipeline is intentionally narrow: behavioral wrapper conflicts, attention diagnostics, vanilla-prompt convergence, and focused causal controls at `options_end` and `all_option_ends`.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick Checks

```bash
pytest
interface-formatting-study --config configs/default.yaml audit-dataset
interface-formatting-study --config configs/default.yaml wrapper-audit
```

From an uninstalled checkout, prefix commands with `PYTHONPATH=src python3 -m interface_formatting_study.cli`.

## Smoke Scoring

This runs calibrated answer-label scoring on a tiny subset. Use the fallback model if memory is tight.

```bash
interface-formatting-study --config configs/default.yaml behavioral-smoke --limit 20 --model Qwen/Qwen2.5-0.5B-Instruct
```

## Full Pipeline

On a GPU host, run the focused paper pipeline:

```bash
scripts/run_interface_formatting_study_focused_mechanistic_gpu.sh
```

The script runs audits, smoke scoring, full or reused behavioral scoring, conflict construction, focused diagnostics, focused controls, tables, manifest validation, and a compact artifact bundle. If `results/raw/behavioral_scores.parquet` and `results/processed/conflict_pairs.parquet` already exist and match the active dataset audit, the script reuses them instead of rerunning full behavioral scoring.

The equivalent manual command sequence is:

```bash
interface-formatting-study --config configs/default.yaml audit-dataset
interface-formatting-study --config configs/default.yaml wrapper-audit
interface-formatting-study --config configs/default.yaml behavioral
interface-formatting-study --config configs/default.yaml conflicts --behavioral results/raw/behavioral_scores.parquet
interface-formatting-study --config configs/default.yaml figures --behavioral results/raw/behavioral_scores.parquet
interface-formatting-study --config configs/default.yaml tables --behavioral results/raw/behavioral_scores.parquet
interface-formatting-study --config configs/default.yaml attention-diagnostics --conflicts results/processed/conflict_pairs.parquet --behavioral results/raw/behavioral_scores.parquet
interface-formatting-study --config configs/default.yaml vanilla-convergence --conflicts results/processed/conflict_pairs.parquet
interface-formatting-study --config configs/default.yaml focused-patching-controls --conflicts results/processed/conflict_pairs.parquet
interface-formatting-study --config configs/default.yaml write-manifest
interface-formatting-study --config configs/default.yaml deliverables
```

The internal test partition must not be used for layer, anchor, alpha, or metric selection. The source MMLU records in this dataset come from the MMLU source `test` split; this repository creates item-disjoint internal train/validation/test partitions within those source-test items.
Smoke runs write figures and tables under `results/smoke/` and are rejected by the final paper-packaging command.
The source manifest contains 20 wrappers, but the active experiment filters to the eight verified pure-interface wrappers before scoring. Rows whose content-free prompt falls back to the generic scaffold are excluded from primary conflict construction.

## Scientific Guardrails

- Do not parse generated prose for the main metric.
- Do not infer the correct answer from prompt text.
- Do not bootstrap wrapper rows independently; cluster by `item_id`.
- Do not call same-item patching proof of sufficiency or necessity.
- Do not use late `answer_anchor` patching as primary mechanistic evidence.
- Do not report `content_end` as primary mechanistic evidence.
- Treat attention as diagnostic evidence, not causal evidence.
- Do not call semantic-anchor recovery a complete router circuit.
