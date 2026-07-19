# Interface Formatting Study

This repository tests whether interface formatting alone can change how a language model answers the same multiple-choice question.

The experiment uses eight intended pure-interface wrappers: `csv_inline`, `graphql_query`, `html_form`, `ini_file`, `key_equals`, `protobuf_msg`, `shell_heredoc`, and `toml_config`. Answer-label likelihood is the primary metric, content-free calibration is provenance-tracked, splits are by MMLU item, and mechanistic claims are restricted to focused semantic-anchor diagnostics and controls.

The current paper pipeline is intentionally narrow: behavioral wrapper conflicts, attention diagnostics, vanilla-prompt convergence, and focused causal controls at `options_end` and `all_option_ends`.

## Completed Three-Model Result

The paper now covers 72,000 wrapped prompts: 24,000 each for the original Qwen
run and the completed Phi and Mistral replications. Calibrated accuracy is
48.4% for Qwen, 56.8% for Phi, and 50.3% for Mistral. The same item changes
between correct and wrong across wrappers in 62.0%, 55.1%, and 66.2% of cases,
respectively. The replication diagnostics preserve the conservative conclusion:
successful and failed wrappers have similar content attention, and the causal
controls do not isolate a same-item wrapper-independent activation vector.
An outcome-blind audit found 111 non-preserving or uncertain source prompts;
removing them leaves conflict rates at 61.5%, 54.8%, and 66.1%, respectively.

The manuscript source is `paper/main.tex`; the new-model comparison tables and
PDF/PNG figures are under `results/cross_model/`.

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

### Isolated Phi and Mistral runs

The focused `run-model` entrypoint runs the same paper protocol for either of
the two pinned additions while keeping their outputs and resumes isolated:

```bash
interface-formatting-study --config configs/default.yaml run-model --profile mistral
interface-formatting-study --config configs/default.yaml run-model --profile phi
```

The profiles pin `microsoft/Phi-3.5-mini-instruct` and
`mistralai/Mistral-7B-Instruct-v0.3` to exact model/tokenizer revisions. Each
uses Transformers' built-in architecture implementation so the declared SDPA
backend is honored rather than an older repository-side model class. Each
run writes to
`results/model_runs/<model-slug>/<semantic-run-id>/`; the semantic ID binds the
model and tokenizer revisions, active dataset, scientific configuration, and
output-affecting source. A mismatched model, dataset, configuration, or source
cannot resume those shards.

Functional and profiling canaries must use different names so their evidence
cannot satisfy one another:

```bash
interface-formatting-study --config configs/default.yaml run-model \
  --profile mistral --canary --canary-name functional-mistral-64 --canary-items 64
interface-formatting-study --config configs/default.yaml run-model \
  --profile mistral --canary --canary-name profile-mistral-64 --canary-items 64 --profile-timings
```

SIGINT or SIGTERM stops after the current shard boundary. Repeating the same
command resumes only matching work. A full run is complete only when its
manifest independently accounts for all 24,000 behavioral work keys plus the
selected vanilla, control, and attention work keys.

### Causal follow-up (prepared, not yet run)

The controlled follow-up preserves the exact stored wrapper prompts plus a
matched plain MCQ. It separately rotates physical answer position and displayed
answer letter, and separately requests generated exact answer text. All 2,401
train and validation source items are retained. Of 21,609 item-format blocks,
21,548 support both interventions; the remaining 61 genuinely have no
independent answer labels, so they retain their baseline and answer-text rows
without inventing a position or label manipulation. The 599 internal-test items
remain held out.

```bash
interface-formatting-causal prepare \
  --option-audit artifacts/causal_option_audit/20260716-v3-final \
  --choice-audit artifacts/causal_choice_audit/20260716-v1
scripts/run_causal_followup_gpu.sh mistral functional
```

The checked design has 172,506 rows per model. Runs are model-, environment-,
and design-identity bound, atomically sharded, signal-safe, resumable, and locally
checksum-verified before teardown. The paid-run gates and commands are in
`CAUSAL_FOLLOWUP_RUN_CARD.md`.

### Decision-binding mechanism (three-model GPU proof complete)

The next experiment asks where the model represents the winning answer content
and where it binds that content to the displayed A-D label. It trains separate
content, physical-position, and displayed-label linear readers on all seven
stored training variants, plus the original baseline-only reader for comparison.
It selects layers on 300 validation items and accepts the readers only on a
separate 300-item validation gate. Test data cannot select layers or tune the
mechanism.

Authenticated schema-3 discovery bundles for all three models are under
`artifacts/decision_binding/v2/discovery/`. Each contains 50,335 exact scored
prompt rows and preserves all 1,801 training and 600 validation items. A run
refuses altered causal scores, mixed model identities, legacy bundles without
scored-source attestation, and a limited canary presented as a complete
discovery run. Readout-only discovery is complete for Mistral, Phi, and Qwen.
Every run reached 295/295 deterministic work units, was fetched through the
strict verifier, and was independently checksum-verified locally. Held-out
macro accuracy for content / position / displayed label was
72.2% / 80.6% / 97.6% for Mistral, 50.1% / 75.7% / 98.6% for Phi, and
50.1% / 70.5% / 95.9% for Qwen. Position and label passed their predeclared
selectivity gates in all three models; content failed its selectivity gate in
all three. Each manifest therefore correctly sets `patch_eligible=false`, and
no causal patching was run. All 599 test items remain untouched.

```bash
scripts/control_decision_binding_gpu.sh start mistral readout \
  artifacts/decision_binding/v2/discovery/mistral-7b-instruct-v0.3
scripts/control_decision_binding_gpu.sh status mistral
scripts/control_decision_binding_gpu.sh stop mistral
```

The exact prepare, discovery, confirmation, resume, fetch, verification, and
teardown contract is in `DECISION_BINDING_RUN_CARD.md`. Confirmation is prepared
only after discovery is complete and frozen, and then uses all 599 held-out test
items without changing the frozen layer/mechanism specification.

### Candidate-local content reader (implemented, GPU run pending)

The bounded follow-up captures each answer option's final audited
content-bearing token at every layer and fits a shared linear ranker to decode
the model's chosen answer content. It trains only on 1,801 discovery-training
items, selects layer and regularization on 300 validation items, and applies the
frozen choice to a separate 300-item gate. A faithful fresh BF16 forward may
change an almost-tied winner; those rows are retained with explicit stored and
fresh scores and an ineligibility reason, never silently dropped or relabeled.
The same persisted eligibility population drives readers, controls, metrics,
strata, parity, and resume.

Runtime batching remains configurable and identity-bound. The identity also
binds GPU model and compute capability, Torch/CUDA, BF16, SDPA, and library
versions, preventing A100 and H100 numerical shards from being mixed. The first
paid stop/go model is Mistral; Phi and Qwen remain locked unless verified
Mistral Tier 2 passes. The 599-question confirmation set and confirmation code
remain locked until that decision.

```bash
scripts/control_decision_binding_content_gpu.sh start mistral functional \
  artifacts/decision_binding/v3_content/discovery/mistral-7b-instruct-v0.3
scripts/control_decision_binding_content_gpu.sh status mistral
scripts/control_decision_binding_content_gpu.sh stop mistral
```

The binding scientific contract is in
`DECISION_BINDING_CONTENT_READER_RUN_CARD.md`; exact launch, stop/go, fetch,
cost, and teardown steps are in
`DECISION_BINDING_CONTENT_EXECUTION_CHECKLIST.md`.

### Mistral two-contract logit lens

The bounded experiment projects every Mistral transformer-block output
through the model's existing final normalization and output head. It compares
plain/wrapped answer-content and A/B/C/D trajectories under matched letter-only
and exact-answer-text instructions. Complete multi-token candidates are scored
as exact teacher-forced continuation paths; this is descriptive continuation
compatibility, not literal access to thought or causal proof.

Only the existing 1,801 train and 600 validation items are in scope. The final
599 remain inaccessible. The frozen scientific, resume, parity, Hopper startup,
cost, fetch, and teardown contracts are in
`DECISION_BINDING_LOGIT_LENS_RUN_CARD.md` and
`DECISION_BINDING_LOGIT_LENS_EXECUTION_CHECKLIST.md`.

### Legacy Qwen run

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
The source manifest contains 20 wrappers, but the active experiment filters to eight wrappers intended as pure-interface formats before scoring. The row audit and defect-exclusion sensitivity are reported alongside the full-population result.

## Scientific Guardrails

- Do not parse generated prose for the main metric.
- Do not infer the correct answer from prompt text.
- Do not bootstrap wrapper rows independently; cluster by `item_id`.
- Do not call same-item patching proof of sufficiency or necessity.
- Do not use late `answer_anchor` patching as primary mechanistic evidence.
- Do not report `content_end` as primary mechanistic evidence.
- Treat attention as diagnostic evidence, not causal evidence.
- Do not call semantic-anchor recovery a complete router circuit.
