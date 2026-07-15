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

The original main run used `Qwen/Qwen2.5-1.5B-Instruct` with the configuration
in `configs/default.yaml`. On a CUDA GPU host, its legacy entrypoint is:

```bash
scripts/run_interface_formatting_study_focused_mechanistic_gpu.sh
```

The script writes behavioral scores, conflict pairs, attention diagnostics,
focused controls, tables, figures, and a manifest under `results/`.

The isolated two-model reproduction uses the focused runner:

```bash
interface-formatting-study --config configs/default.yaml run-model --profile mistral
interface-formatting-study --config configs/default.yaml run-model --profile phi
```

Pinned profiles:

- `microsoft/Phi-3.5-mini-instruct` at model/tokenizer revision
  `2fe192450127e6a83f7441aef6e3ca586c338b77`;
- `mistralai/Mistral-7B-Instruct-v0.3` at model/tokenizer revision
  `c170c708c41dac9275d15a8fff4eca08d52bab71`.

Both expect 32 transformer blocks and use mechanistic layers
`[2, 5, 7, 9, 11, 14, 16, 18]`. Behavioral, vanilla-convergence, and causal
control phases use BF16 plus SDPA. Attention diagnostics reload the selected
model once with eager attention. Both profiles use Transformers' built-in
architecture classes; this avoids the pinned Phi repository's older remote
class, which does not advertise SDPA support.

Outputs are written under
`results/model_runs/<model-slug>/<semantic-run-id>/`. The semantic identity and
every shard bind the selected model/tokenizer revision, dataset, scientific
configuration, and output-affecting source. Shards are atomic; matching work
resumes, identical duplicates merge once, and identity or content conflicts
fail closed.

Use distinct canary namespaces for functional and profiling evidence:

```bash
interface-formatting-study --config configs/default.yaml run-model \
  --profile phi --canary --canary-name functional-phi-64 --canary-items 64
interface-formatting-study --config configs/default.yaml run-model \
  --profile phi --canary --canary-name profile-phi-64 --canary-items 64 --profile-timings
```

After both full model runs have been independently fetched and verified, build
the cross-model tables, PDF/PNG figures, and paper assets on the Mac:

```bash
python3 analysis/compare_model_runs.py \
  --model-run Mistral=/path/to/mistral/run \
  --model-run Phi=/path/to/phi/run \
  --output-dir results/cross_model \
  --paper-dir paper
```

The completed full-run identities are:

- Mistral semantic run `e468f7b1bb2a121e6904`: 24,000 behavioral work keys
  and 300 work keys in each mechanistic phase, independently verified across
  151 shards;
- Phi semantic run `a0f18fe1c8c223d8aec8`: 24,000 behavioral work keys and
  264 work keys in each mechanistic phase, independently verified across 145
  shards.

Both identities bind dataset SHA-256
`57c676b3ead7627b5d720c0aacdba1284925cc84fee83bff063d724c87ce085d`.
The exact fetched artifacts and their `LOCAL_SHA256SUMS.txt` ledgers are retained
under `gpu_artifacts/<model-slug>/<semantic-run-id>/<fetch-timestamp>/` in the
completed workspace and in the final compact archive.

The paid campaign used one full-run A6000 pod plus three bounded failed launch
attempts. Known campaign IDs posted $2.18; conservative launch-price-by-elapsed
accounting was approximately $2.44, which is the recorded campaign cost. The
full-run pod was terminated immediately after the second verified fetch, the
final provider audit showed zero live pods, the pre-campaign SSH key was restored
as primary, and the temporary cloud and local SSH keys were deleted. Full
attempt, timing, throughput, canary, deviation, and teardown evidence is retained
in `docs/TWO_MODEL_EXPERIMENT_RUN_CARD.md` inside the final archive.

The operator contract, spend gates, and exact teardown proof are in
`docs/TWO_MODEL_EXPERIMENT_RUN_CARD.md`.

## Prepared causal follow-up

The next experiment uses a deterministic 124,704-row design over 1,732 safely
transformable train and validation items. For each of eight exact stored
wrapper prompts and one matched plain MCQ, it has one controlled baseline,
three position-only rotations, three letter-only rotations, and one generated
answer-text prompt.
Raw and content-free-calibrated letter outcomes are saved; deterministic exact
generation is the primary text outcome, with candidate likelihoods secondary.
The internal test partition is not included.

```bash
interface-formatting-causal prepare \
  --output artifacts/causal_followup/v2_source_preserving/design_train_validation.parquet
scripts/run_causal_followup_gpu.sh mistral functional
```

The design manifest, numerical-environment-bound semantic identity, atomic
shards, verified partial/complete fetch, analysis command, canary forecast,
budget gates, and teardown procedure are in `CAUSAL_FOLLOWUP_RUN_CARD.md`.
The baseline rows are byte-identical to the completed original-prompt runs.
Items that cannot be transformed without guessing are excluded as a whole and
recorded in the checksum-bound exclusion ledger.

The outcome-blind audit artifacts are retained under
`artifacts/wrapper_audit/20260715-v1/`. Its exact-coverage first pass and
separate model-assisted adjudication labels 107 old prompts as content-changing and four as
ambiguous. `analysis/compare_model_runs.py --wrapper-audit <final parquet>`
writes the corresponding sensitivity table.

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
