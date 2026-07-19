# Decision-Binding Candidate-Local Content Reader Execution Checklist

This is the binding task ledger for the candidate-local content-reader experiment. It does not authorize paid execution. Fill every receipt from live state; do not infer completion from this document.

## Scope And Release Lock

- [x] Branch is `codex/decision-binding-mechanism`; exact release commit: `29e4be646bb4fca215cf64c84c9127ca3e5c3507`.
- [x] `git status --short --branch` was clean before launch and after fetch verification; release commit matched `origin/codex/decision-binding-mechanism`.
- [x] No stored prompt, wrapper, audit, split, v2 artifact, or final-set artifact changed.
- [x] The implementation remains inside the existing content module/CLI/operator path; no model-specific batching branch or campaign framework was added.
- [x] Exactly two independent reviewers closed the frozen implementation diff: scientific/data semantics `/root/scientific_diff_review`; runtime/resume/operator `/root/runtime_diff_review`; frozen implementation commit `1e09966`, coordinated fix through `56b118e`.
- [x] One final full local suite passed against the reviewed stable diff: `python3 -m pytest -q`; 319 passed; 28.80 seconds on 2026-07-19.

## Frozen Scientific Decisions

- [x] The hard content target is usable only when the stored winner is unique, answer content is unambiguous, the fresh winner is unique, and stored/fresh winning content IDs agree.
- [x] Exclusive ineligibility reason precedence is frozen as: `stored_raw_tie`, `ambiguous_answer_content`, `fresh_raw_tie`, `stored_fresh_content_mismatch`, `eligible`.
- [x] Ineligible rows are retained and counted, never relabeled, and excluded only from hard-target fitting, selection, controls, metrics, confidence intervals, and gates.
- [x] No numerical epsilon is used for scientific target eligibility.
- [x] Batch/scalar parity limits are frozen at `2e-2` for activations, raw log probabilities, and selected-reader probabilities. A winner swap is ambiguous only when both top-two margins are no greater than twice the observed coordinate drift. The `1e-12` comparison allowance only represents the declared decimal boundary in floating point; it is not tuned from data.
- [x] L2 grid is `1e-4 1e-3 1e-2 1e-1`; seed is `0`; inference is BF16; fitting is float32; attention backend is SDPA.
- [x] Bootstrap samples are `5000`; within-item target permutations are `1000`.
- [x] The claim remains candidate-local linear decodability of chosen answer content, not causation, factual understanding, or a unique circuit.

## Discovery Source Receipts

- [x] Prepared Mistral bundle: `artifacts/decision_binding/v3_content/discovery/mistral-7b-instruct-v0.3/`.
- [x] Prepared manifest hash: `d3745397e01a92bc01fbc75fe827ebd0ea60cf8cca478e71c5cf87504e74118c`.
- [x] Candidate-sites hash: `c79d0ee9167c4675ac9bdf1e415118a04b84bbb7f2807ee119a696b56e7327f3`.
- [x] Rows/items match the frozen discovery contract: 50,335 rows; 1,801 training items; 300 layer-selection items; 300 reader-gate items.
- [x] Source eligibility receipt matches 49,841 eligible rows and 494 ineligible rows: 473 stored ties and 21 ambiguous-content rows.
- [x] Training baseline receipt matches 1,801 represented items, including 1,779 source-eligible, 19 stored-tie, and 3 ambiguous-content items before fresh-forward classification.
- [x] Phi and Qwen source bundles remained unopened for execution because Mistral did not pass the selection gate; no confirmation bundle was opened.
- [x] The untouched 599-question confirmation outcomes were not opened or inspected.

## Local Implementation And Proof

- [x] RED evidence is recorded for forward eligibility, activation-state resume, uniform masks, control selection, parity, hardware identity, optimizer exhaustion, activation corruption, operator telemetry, and frozen-artifact verification.
- [x] Focused GREEN command passed:

  ```bash
  python3 -m pytest -q \
    tests/test_decision_binding_content.py \
    tests/test_decision_binding_content_cli.py \
    tests/test_run_identity.py \
    tests/test_model_loader.py \
    tests/test_model_profiles.py \
    tests/test_cache_causal_models.py \
    tests/test_cli_contracts.py
  ```

- [x] Interrupted/resumed synthetic execution produced the same eligibility hash and accepted training keys as uninterrupted execution.
- [x] Artifact tamper tests reject stale identity, changed eligibility state, non-finite or misaligned activations, inconsistent repeated work keys, and altered frozen ranker hashes.
- [x] Review batch and focused re-review receipts are linked to commits `1e09966`, `94d77db`, and `56b118e`.
- [x] Final suite passed exactly once after review closure:

  ```bash
  python3 -m pytest -q
  ```

## Thin Deploy And Runtime Preflight

- [x] Thin source sync excluded `.git`, local caches, generated results, GPU artifacts, model cache, and private credentials; the prepared Mistral discovery bundle was transferred separately.
- [x] Release/dirty receipt: clean pushed commit `29e4be646bb4fca215cf64c84c9127ca3e5c3507`; remote working directory `/root/interface_formatting_study`.
- [x] Provider receipt: Prime Intellect/DataCrunch FI, non-spot H100 80 GB SXM5, Ubuntu 22 CUDA 12 image, no task-owned persistent disk, `$3.25/hour`; task-owned pod `da7e55340a064408adc632e19b50cde3`.
- [x] Paid launch had explicit approval in the owner task on 2026-07-19.
- [x] SSH proved `root`, adequate disk, NVIDIA H100 80GB HBM3, CUDA 12.6 runtime, and no conflicting Keyhole task process.
- [x] Repository sync and bootstrap completed under `/root/interface_formatting_study`.
- [x] Bootstrap/cache used the validated virtual environment:

  ```bash
  cd /root/interface_formatting_study
  scripts/bootstrap_causal_followup_gpu.sh
  .venv/bin/python scripts/cache_causal_models.py
  ```

- [x] Runtime receipt is identity-bound: NVIDIA H100 80GB HBM3, compute capability 9.0, BF16, SDPA, Torch 2.7.1+cu126, CUDA 12.6, Transformers 4.53.2, tokenizers 0.21.2, Python 3.11.15, and Mistral revision `c170c708c41dac9275d15a8fff4eca08d52bab71`.
- [x] Task-owned pod ID and hourly price are recorded above without credentials; SSH endpoint is intentionally omitted from this durable public ledger.

## One Changed-Surface Mistral Startup/Resume Check

- [x] Used one bounded production-entrypoint startup/stop/resume check; no repetitive canary chain.
- [x] Exported the startup configuration:

  ```bash
  export BATCH_SIZE=32
  export MAX_BATCH_TOKENS=40000
  export CAPTURE_CHUNK_SIZE=1
  export CANARY_ITEMS=8
  ```

- [x] Started with the prepared Mistral discovery bundle:

  ```bash
  scripts/control_decision_binding_content_gpu.sh start mistral functional \
    artifacts/decision_binding/v3_content/discovery/mistral-7b-instruct-v0.3
  ```

- [x] `status`/`tail` telemetry showed phase, completed/total units, elapsed time, throughput, and peak VRAM; the interrupted and terminal manifests were checked.
- [x] Stopped after layer-selection progress, confirmed a clean `SIGTERM` interrupted manifest, then resumed with semantic run ID `2523de246414badb3924`.
- [x] Resume reused completed layer-selection shards and advanced into `reader_gate` without duplicate or lost work keys.
- [x] H100 identity, eligibility receipts, batch/scalar parity, ranker receipts, and compact canary artifacts passed strict local verification.
- [x] Canary fetch: `gpu_artifacts/decision_binding_content/mistral-7b-instruct-v0.3/2523de246414badb3924/20260718T234552Z`.
- [x] Resumed attempt wall time 47.754 seconds; peak VRAM 14,996,925,440 bytes; observed utilization reached active-compute load. Canary gate booleans were non-scientific and ignored.
- [x] No identity mismatch, lost/duplicated row, verifier failure, parity failure, or silent telemetry blocked the full run.

## Full Mistral Discovery

- [x] Restored production chunking while keeping batching as recorded runtime configuration:

  ```bash
  export BATCH_SIZE=32
  export MAX_BATCH_TOKENS=40000
  export CAPTURE_CHUNK_SIZE=64
  export MAX_ITER=1000
  ```

- [x] Started the full run:

  ```bash
  scripts/control_decision_binding_content_gpu.sh start mistral full \
    artifacts/decision_binding/v3_content/discovery/mistral-7b-instruct-v0.3
  ```

- [x] Monitored without changing settings during the terminal run. Semantic run ID `e32ba1925d2b3362f71a`; no restart; wall time 571.580 seconds; peak VRAM 29,484,389,376 bytes. Training retained 1,765/1,801 evaluable targets; layer selection retained 18,514/18,882 evaluable rows with eligibility hash `4d889b634e94a4813d9605b13b8d75980d58852b49946ee4f73c6fa5462981af`.
- [x] No full-run stop was necessary; stop/resume policy remained identity-bound and unchanged.
- [x] Fetched compact artifacts with `complete` mode and passed the strict local verifier before interpreting the gate.
- [x] Mistral selection result: **FAIL** — the selected layer-31, L2=0.1 reader did not beat both isolated nuisance controls, so `reader_gate` was never opened.
- [x] Mistral Tier 1 result: **FAIL (not opened)**.
- [x] Mistral Tier 2 result: **FAIL (not opened)**.
- [x] Tier 1/selection failure stopped Phi, Qwen, and confirmation.
- [ ] Tier 1 pass plus Tier 2 failure remains descriptive and stops Phi, Qwen, and confirmation.
- [ ] Only verified Tier 2 enables the unchanged discovery method for both Qwen and Phi.

## Phi And Qwen Discovery Lock

- [x] This section remained locked because Mistral did not reach verified Tier 2.
- [ ] Run Qwen discovery, then Phi discovery, through the same full operator and scientific settings; no per-model batching code and no repetitive canary chain.
- [ ] A Qwen failure does not skip Phi; both verified discovery outcomes are retained.
- [ ] Each model independently records Tier 1/Tier 2, eligibility counts/hash, parity, selected layer/L2/control choices, artifact hashes, runtime, peak VRAM, throughput, and cost.
- [ ] Only models independently passing Tier 2 enter the frozen confirmation-eligible roster: `________________`.

## Conditional Confirmation Lock

- [x] No confirmation code or real 599-row bundle was used because Mistral Tier 2 did not pass.
- [ ] Confirmation implementation uses synthetic 599-row fixtures, repeats the required two-reviewer gate, and passes one stable final suite before real access.
- [ ] The confirmation CLI verifies a full non-canary Tier-2 discovery root, model/profile, frozen policy, selection, normalization, rankers, controls, and hashes before opening a confirmation bundle.
- [ ] Confirmation exposes no fitting, selection, threshold, or tuning path and writes separate outputs.
- [ ] Each real confirmation run is separately authorized and executed once only for an independently Tier-2-passing model.

## Fetch, Cost, And Teardown Closeout

- [x] Every fetched startup/full artifact passed the strict content-reader verifier, including manifest-bound SHA-256 checks.
- [x] Startup artifact: `gpu_artifacts/decision_binding_content/mistral-7b-instruct-v0.3/2523de246414badb3924/20260718T234552Z`; full artifact: `gpu_artifacts/decision_binding_content/mistral-7b-instruct-v0.3/e32ba1925d2b3362f71a/20260719T000144Z`. No Qwen, Phi, or confirmation artifact exists because the gate stayed closed.
- [x] Provider billing receipt for task-owned setup/run pods: approximately `$1.31` total (`$1.16` terminal H100 pod plus three approximately `$0.05` failed setup pods); wallet balance after teardown `$2.91`.
- [x] Useful manifests, frozen selection, ranker receipts, score tables, and compact result artifacts were preserved before teardown.
- [x] Task-owned pod `da7e55340a064408adc632e19b50cde3` was terminated only after verified fetch.
- [x] No task-owned persistent disk was created; no disk deletion was required.
- [x] Final provider inventory contained no task-owned pod or disk. The unrelated TWC A100 pod/disk remained active and untouched.
- [x] Final Git status was clean before this receipt-only closeout edit; all implementation commits were pushed to origin.
- [x] Scientific conclusion is bounded: under the frozen candidate-local method, Mistral did not clear nuisance-control selection, so there is no supported reader-gate, cross-model, causal, factual-understanding, or final-confirmation claim. The next paper decision is to report this stopped negative experiment, not tune or expand it post hoc.
