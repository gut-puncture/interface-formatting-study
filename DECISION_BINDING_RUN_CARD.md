# Decision-Binding Mechanism Run Card

## Owner Scope Box

- Existing path reused: the current decision-binding ledger, all-layer/two-checkpoint capture, linear probes, patching, semantic identity, atomic shards, and GPU operators.
- New production code: a narrow correction is required because the current single probe is trained only where answer content, physical position, and displayed label coincide. The correction trains three independently identified readers and adds a readout-only verification boundary.
- Expected scope: about 450-650 net new production lines plus focused tests, confined to the existing decision-binding module, CLI, and thin operator/verifier scripts. Pause and simplify before 700 production lines or ten implementation hours before review.
- Review: exactly two independent reviewers, one for scientific/data semantics and one for runtime/resume/operator behavior; maximum two review rounds.
- Final validation: one full local suite after review closes, expected under five minutes; first real GPU proof is one full Mistral readout-only run after local acceptance and explicit rental approval. A smaller run would repeat the full probe-training cost without faithfully testing the changed surface.
- Deviation gate: stop before 700 net new production lines, ten implementation hours before review, two review rounds, or adding generalized orchestration, persistent hidden-state storage, new token positions, nonlinear probes, position patching, head/MLP hooks, or new prompt/audit infrastructure.

## Source Truth Read

- Approved decision-binding implementation plan in the owning Codex task.
- `AGENTS.md`, `CAUSAL_FOLLOWUP_RUN_CARD.md`, `pyproject.toml`, and GPU dependency locks.
- `causal_design.py`, `causal_runner.py`, `causal_cli.py`, `causal_option_maps.py`, `anchors.py`, `hooks.py`, `patching.py`, `shards.py`, `run_identity.py`, and model profiles.
- Existing causal designs/results, conflict-pair artifacts, split manifest, and their tests.

## Approved Behavior

- Preserve every existing plain/wrapped prompt byte-for-byte and build controlled variants only through audited spans.
- Track canonical answer content, physical position, and displayed output label as separate structured coordinates; never infer content identity from normalized or generated answer text.
- Train independent content, position, and displayed-label four-class linear readouts on all seven exact stored plain variants for the 1,801-item training split. Train a baseline-only legacy reader from the same capture for comparison.
- Split the 600 validation items deterministically and subject-stratified into 300 layer-selection items and 300 untouched reader-gate items. Freeze only readers that pass the predeclared gate; keep all 599 test items untouched for later confirmation.
- Use raw model scores for mechanistic targets and causal endpoints. Carry calibrated scores only as secondary continuity with the existing paper.
- Apply probe-subspace, full-residual, norm-matched random, identity, and unpatched conditions on every prespecified validation/test conflict pair plus deterministic stable and label-binding controls.
- Non-goals: new prompts/wrappers, LLM parsing, literal answer-text token lenses, nonlinear probes, attention/head/MLP/neuron searches, broad behavioral reruns, quantization, compile tuning, or paper edits.

## First Vertical Slice

- Behavior/invariant: exact source prompts become a deterministic ledger with content/position/label mappings and tokenizer-safe `format_end`/`answer_prefix_end` checkpoints; no unexpected row can be silently excluded.
- Entry point: `python -m interface_formatting_study.decision_binding_cli prepare`.
- Data/fixture: existing source prompt, audited option mapping, and controlled assignment fixture.
- Proof path: focused public-seam tests fail before implementation, then prove prompt hashes, single-factor edits, duplicate-text flags, mappings, full-token offsets, padding adjustment, and fail-closed invalid inputs.

## Files And Interfaces

- Likely touched areas: new `decision_binding.py` and `decision_binding_cli.py`; narrow extensions to existing anchors/patching only where the current public seam cannot express the approved operation; focused tests; thin scripts and analysis.
- Public entrypoints: CLI subcommands `prepare`, `run-model`, and `analyze`; compact ledgers; `results/decision_binding_runs/<model-slug>/<semantic-run-id>/`.
- Legal states: prepared, probing, frozen, confirming, complete, stopped, or failed. Confirmation requires the exact frozen-mechanism hash. Complete requires every expected work key and a verified merged artifact.
- Protected meanings: answer content is an item-relative canonical ID; position is the physical slot; label is displayed A-D; raw choice is internal model output; calibrated choice is post-hoc adjustment; probe decodability is not causality.
- Compatibility: existing artifacts are read-only inputs. They are neither migrated nor overwritten. New run identity prevents old causal or cross-model shards from satisfying the new experiment.

## Edge Cases And Failure Modes

- Retain literal duplicate answers and mark content identity ambiguous. Exclude them only from content-reader fitting and content-specific metrics; position and label targets remain valid.
- Retain all baseline prompts. Mark only scientifically impossible single-factor transforms not applicable; parsing/anchor failures are fatal rather than row exclusions.
- Keep exact raw ties in artifacts but out of hard-winner probe fitting and classification.
- Resolve checkpoints on the complete tokenized prompt with offset mappings; reject missing offsets, mismatched IDs, instruction-overlapping content tokens, empty prompts, and wrong padding adjustment.
- Reject stale prompt hashes, wrong mappings, wrong model/tokenizer/code identity, conflicting duplicate shards, nonconverged probes, and mismatched frozen specifications.
- SIGINT/SIGTERM completes the current shard and writes progress. Save each completed coordinate reader atomically. Training activation capture may restart because it is bounded; validation/confirmation readout and patching resume by deterministic work key.

## Design Simplicity

- One domain module owns scientific mappings, readout, probes, and patch rows; the CLI owns filesystem orchestration only.
- Existing scorer, residual capture/replacement, identity, and shard modules remain the mechanism owners.
- Hidden activations are preallocated as float32, reused by every reader, converted to float64 only one layer/checkpoint slice at a time, and discarded after fitting/scoring. Only compact probe weights, probabilities, logits, patch effects, manifests, and timings persist.
- Rejected: database, registry/plugin framework, generic campaign system, model-specific hook hierarchy, persistent activation archive, and task-budget architecture.
- Paid-run cost remains task-local: live provider receipt, five-minute monitoring, explicit scale approval, and immediate task-owned teardown.

## Task Slices

1. Exact seven-variant training ledger, coordinate-specific eligibility, and deterministic 300/300 validation roles; focused RED-GREEN-REFACTOR; commit and push.
2. Shared activation capture, three coordinate readers plus legacy comparator, arm-aware selectivity, held-out usability gates, and coordinate-bound patching; focused RED-GREEN-REFACTOR; commit and push.
3. Readout-only status/resume, strict readout verification, partial-fetch correction, dependency/runtime identity, and thin operator mode; focused RED-GREEN-REFACTOR; commit and push.
4. Documentation, two-reviewer batch, coordinated fix/re-review, final suite, commit and push.

## Validation Ladder

- TDD: every behavior-changing slice begins with a focused test through the exported module or CLI and an observed expected RED failure.
- Source/smoke: imports, CLI help, ledger schema, deterministic fixture preparation.
- Real-use micro-proof: exact stored prompts and pinned tokenizers locally where available; one complete Mistral readout-only production run after local acceptance. Phi and Qwen do not start until the corrected Mistral readers are assessed.
- Focused tests: semantic mappings, tokenizer boundaries/padding, synthetic coordinate probes, projection math, target mapping, scalar/batched parity, identity/random controls, semantic identity, interruption/resume, duplicate conflict handling, and frozen confirmation.
- Review gate: freeze after focused tests and owner self-review; open both reviewers together; make no production edit or final-suite run while review is open; reconcile all findings together; apply one fix batch; run affected tests and focused re-review.
- Full gate: run the full suite exactly once after re-review closes and no further production edit is planned.

## Operator / Cost Preflight

- New/changed operator mode: add only the missing decision-binding run, monitor, fetch, and verify surface around existing causal operators.
- Lifecycle DAG: local prepare -> code-only sync -> remote tokenizer/checkpoint preflight -> complete Mistral readout-only run -> shard/manifest validation -> compact fetch -> local checksum verification -> task-owned teardown -> local analysis -> explicit patch/other-model decision.
- Remote mode: BF16, SDPA, inference mode, one model on a non-spot 80 GB A100 with at least 64 GB host RAM. H100 is unnecessary because the corrected probe fitting is expected to be CPU-dominant. No eager-attention reload or tuning campaign.
- Telemetry: phase, completed/total work, last shard, throughput, token/padding ratio, preparation/forward/write time, peak VRAM, errors, ETA, provider spend, and stop reason.
- No rental or full scale begins until local validation, exact operator commands, changed-surface canaries, artifact fetch, and teardown are proven and the owner explicitly approves current spend.

## Review And Approval

- Review batch: frozen base/target; scientific reviewer and runtime reviewer; findings classified as `CONFIRMED_DEFECT`, `MATERIAL_RISK`, `SPECULATIVE`, or `DESIGN_OPTION`; one coordinated accepted-fix batch and one focused re-review.
- Evidence for Shailesh: exact item/block counts, zero silent prompt loss, sample prompt/hash/mapping audit, probe/control acceptance summary, interruption/resume equality, complete local test result, exact GPU commands, runtime/cost forecast, and residual risks.
- Rejection flags: any prompt-byte drift, content/position/label conflation, test leakage, calibration treated as an internal mechanism, uncontrolled patch effect, silent row exclusion, mismatched resume identity, missing stop/fetch/teardown path, or unresolved direct reviewer defect.

## Docs Sync And Stop Rules

- Update reproducibility/operator documentation after final validation. Do not edit the paper until confirmation findings are complete.
- Stop for missing source truth, a user-owned scientific behavior decision, unapproved paid action, or an impossible faithful proof.
- Rebuild rather than keep patching if round two finds another material defect in the same architecture/failure class.

## Mistral Readout Closeout (2026-07-17)

- Completed readout-only semantic run `da8ad9009c3a91841e96` for pinned
  Mistral-7B-Instruct-v0.3. The identity-bound manifest reports
  `readout_complete`, 295/295 shards, and 2,414,592/2,414,592 merged rows.
- Held-out gate macro accuracy was 72.2% for content, 80.6% for position, and
  97.6% for displayed label. Position and label were usable; content failed its
  selectivity lower-bound gate (`-0.0135`) and was not usable. Consequently,
  `patch_eligible=false`; no causal patch or other-model run was started.
- The complete run is under
  `gpu_artifacts/decision_binding/mistral-7b-instruct-v0.3/da8ad9009c3a91841e96/20260717T040748Z/`.
  Strict local verification passed for all 295 readout work keys, and
  `LOCAL_SHA256SUMS.txt` contains 601 independently rechecked file hashes.
- Runtime was 1,990.97 seconds. The successful A100 pod cost $1.3153; including
  the earlier aborted setup pod, this execution attempt cost $1.4501. Final
  Prime inventory was `pods: []` and `disks: []`.
- All 599 test items remain untouched. The next step is an explicit scientific
  decision about whether the failed content-selectivity gate warrants a revised
  preregistered reader or readout-only replication on Phi and Qwen.
