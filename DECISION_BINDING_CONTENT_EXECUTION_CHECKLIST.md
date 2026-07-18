# Decision-Binding Candidate-Local Content Reader Execution Checklist

This is the binding task ledger for the candidate-local content-reader experiment. It does not authorize paid execution. Fill every receipt from live state; do not infer completion from this document.

## Scope And Release Lock

- [ ] Branch is `codex/decision-binding-mechanism` and the exact release commit is recorded: `________________`.
- [ ] `git status --short --branch` is clean before packaging, launch, fetch verification, and closeout.
- [x] No stored prompt, wrapper, audit, split, v2 artifact, or final-set artifact changed.
- [x] The implementation remains inside the existing content module/CLI/operator path; no model-specific batching branch or campaign framework was added.
- [x] Exactly two independent reviewers closed the frozen implementation diff: scientific/data semantics `/root/scientific_diff_review`; runtime/resume/operator `/root/runtime_diff_review`; frozen implementation commit `1e09966`, coordinated fix through `56b118e`.
- [x] One final full local suite passed against the reviewed stable diff: `python3 -m pytest -q`; 318 passed; about 28 seconds on 2026-07-19.

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

- [ ] Prepared Mistral bundle: `artifacts/decision_binding/v3_content/discovery/mistral-7b-instruct-v0.3/`.
- [ ] Prepared manifest hash: `________________`.
- [ ] Candidate-sites hash: `________________`.
- [ ] Rows/items match the frozen discovery contract: 50,335 rows; 1,801 training items; 300 layer-selection items; 300 reader-gate items.
- [ ] Source eligibility receipt matches 49,841 eligible rows and 494 ineligible rows: 473 stored ties and 21 ambiguous-content rows.
- [ ] Training baseline receipt matches 1,801 represented items, including 1,779 source-eligible, 19 stored-tie, and 3 ambiguous-content items before fresh-forward classification.
- [ ] Existing Phi and Qwen discovery source bundles and hashes are recorded without opening any confirmation bundle.
- [ ] The untouched 599-question confirmation outcomes have not been opened or inspected.

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

- [ ] Code-only deploy bundle excludes `.git`, data, caches, generated artifacts, model files, GPU artifacts, and private credentials.
- [ ] Bundle commit/dirty receipt, size, and deploy time are recorded: `________________`.
- [ ] Live provider inventory, H100 80 GB resource, region, disk, hourly price, wallet baseline, teardown reserve, and approved cost ceiling are recorded: `________________`.
- [ ] Paid launch has separate explicit approval; approval receipt: `________________`.
- [ ] SSH proves the expected user, disk capacity, H100 model, CUDA driver, and no conflicting task process.
- [ ] Repository sync and bootstrap complete under the intended working directory.
- [ ] Bootstrap/cache use the validated virtual environment:

  ```bash
  cd /home/ubuntu/interface_formatting_study
  scripts/bootstrap_causal_followup_gpu.sh
  .venv/bin/python scripts/cache_causal_models.py
  ```

- [ ] Runtime receipt records GPU name, compute capability, BF16 support, SDPA, Torch, CUDA, Transformers, tokenizers, Python, model revision, and exact release commit.
- [ ] Task-owned pod ID, disk ID, SSH endpoint, creation time, and live hourly price are recorded without exposing credentials.

## One Changed-Surface Mistral Startup/Resume Check

- [ ] Use only one bounded production-entrypoint check; do not start a chain of functional/profiling canaries.
- [ ] Export the startup configuration:

  ```bash
  export BATCH_SIZE=32
  export MAX_BATCH_TOKENS=40000
  export CAPTURE_CHUNK_SIZE=1
  export CANARY_ITEMS=8
  ```

- [ ] Start with the prepared Mistral discovery bundle:

  ```bash
  scripts/control_decision_binding_content_gpu.sh start mistral functional \
    artifacts/decision_binding/v3_content/discovery/mistral-7b-instruct-v0.3
  ```

- [ ] `status`/`tail` telemetry shows phase, completed/total units, elapsed time, throughput, and peak VRAM; errors remain in the log and interruption/failure reason is verified from the terminal manifest.
- [ ] Stop after durable progress, confirm a clean interrupted manifest, then resume with the identical command and semantic identity.
- [ ] Resume reuses completed activation state without duplicate or lost work keys.
- [ ] H100 identity, eligibility receipts, batch/scalar parity, ranker receipts, and compact artifacts verify strictly.
- [ ] Fetch with `canary` mode and record local path/checksums: `________________`.
- [ ] Startup-check runtime, peak VRAM, utilization snapshot, and actual cost are recorded: `________________`.
- [ ] Any identity mismatch, lost/duplicated row, verifier failure, resolvable parity failure, or silent telemetry blocks the full run.

## Full Mistral Discovery

- [ ] Restore production chunking while keeping batching as recorded runtime configuration:

  ```bash
  export BATCH_SIZE=32
  export MAX_BATCH_TOKENS=40000
  export CAPTURE_CHUNK_SIZE=64
  ```

- [ ] Start the full run:

  ```bash
  scripts/control_decision_binding_content_gpu.sh start mistral full \
    artifacts/decision_binding/v3_content/discovery/mistral-7b-instruct-v0.3
  ```

- [ ] Monitor without changing scientific settings. Record semantic run ID, target-state counts/hash, progress, throughput, peak VRAM, restart count, and running cost.
- [ ] If stopping is necessary, use `stop`, wait for the current shard and interrupted manifest, then resume only with identical identity-bound settings.
- [ ] Fetch compact artifacts with `complete` mode and verify locally before interpreting the gate.
- [ ] Mistral Tier 1 result: `PASS / FAIL`.
- [ ] Mistral Tier 2 result: `PASS / FAIL`.
- [ ] Tier 1 failure stops Phi, Qwen, and confirmation.
- [ ] Tier 1 pass plus Tier 2 failure remains descriptive and stops Phi, Qwen, and confirmation.
- [ ] Only verified Tier 2 enables the unchanged discovery method for both Qwen and Phi.

## Phi And Qwen Discovery Lock

- [ ] This section remains locked until verified Mistral Tier 2.
- [ ] Run Qwen discovery, then Phi discovery, through the same full operator and scientific settings; no per-model batching code and no repetitive canary chain.
- [ ] A Qwen failure does not skip Phi; both verified discovery outcomes are retained.
- [ ] Each model independently records Tier 1/Tier 2, eligibility counts/hash, parity, selected layer/L2/control choices, artifact hashes, runtime, peak VRAM, throughput, and cost.
- [ ] Only models independently passing Tier 2 enter the frozen confirmation-eligible roster: `________________`.

## Conditional Confirmation Lock

- [ ] No confirmation code or real 599-row bundle is used unless Mistral Tier 2 passed and the discovery roster is frozen.
- [ ] Confirmation implementation uses synthetic 599-row fixtures, repeats the required two-reviewer gate, and passes one stable final suite before real access.
- [ ] The confirmation CLI verifies a full non-canary Tier-2 discovery root, model/profile, frozen policy, selection, normalization, rankers, controls, and hashes before opening a confirmation bundle.
- [ ] Confirmation exposes no fitting, selection, threshold, or tuning path and writes separate outputs.
- [ ] Each real confirmation run is separately authorized and executed once only for an independently Tier-2-passing model.

## Fetch, Cost, And Teardown Closeout

- [ ] Every fetched artifact passes the strict content-reader verifier and independent local SHA-256 verification.
- [ ] Local artifact directories and semantic run IDs are recorded for startup, Mistral, Qwen, Phi, and any eligible confirmations.
- [ ] Actual provider cost is reconciled from live receipts, including failed/setup time: `________________`.
- [ ] Useful logs, manifests, frozen specifications, and compact result artifacts are preserved before teardown.
- [ ] Task-owned pod is explicitly terminated only after verified fetch.
- [ ] Task-owned disk is explicitly deleted only after verified fetch and approval.
- [ ] Final provider inventory receipt is `pods: []` and `disks: []` for task-owned resources.
- [ ] Final Git status is clean; verified commits are pushed to origin.
- [ ] Residual scientific limits and the exact next paper/research decision are recorded without expanding the claim.
