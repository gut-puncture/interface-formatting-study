# Decision-Binding Mechanism Run Card

## Owner Scope Box

- Existing path reused: the causal-follow-up model profiles, exact prompt artifacts, token-aware scorer, residual hooks/patcher, semantic identity, atomic shards, and GPU operators.
- New production code: one focused decision-binding domain module and one thin CLI are necessary because the repository has no all-layer readout, linear-probe, or probe-subspace patch path.
- Expected scope: after the first review exposed missing canonical-source attestation, frozen-test analysis, and artifact completeness proof, about 2,100-2,250 net production lines plus focused tests; still one experiment domain module, one CLI, and narrow operator/analysis additions. Scientific scope is unchanged; the measured increase is fail-closed validation and proof for already approved behavior, not new features.
- Review: exactly two independent reviewers, one for scientific/data semantics and one for runtime/resume/operator behavior; maximum two review rounds.
- Final validation: one full local suite after review closes, expected under five minutes; first real GPU proof is one eight-item canary per model after local acceptance and explicit rental approval.
- Deviation gate: the first 1,750-line gate triggered and the design was rechecked after review; stop again before exceeding 2,300 net production lines, fourteen implementation hours, two review rounds, or adding generalized orchestration, persistent hidden-state storage, head/MLP hooks, or new prompt/audit infrastructure.

## Source Truth Read

- Approved decision-binding implementation plan in the owning Codex task.
- `AGENTS.md`, `CAUSAL_FOLLOWUP_RUN_CARD.md`, `pyproject.toml`, and GPU dependency locks.
- `causal_design.py`, `causal_runner.py`, `causal_cli.py`, `causal_option_maps.py`, `anchors.py`, `hooks.py`, `patching.py`, `shards.py`, `run_identity.py`, and model profiles.
- Existing causal designs/results, conflict-pair artifacts, split manifest, and their tests.

## Approved Behavior

- Preserve every existing plain/wrapped prompt byte-for-byte and build controlled variants only through audited spans.
- Track canonical answer content, physical position, and displayed output label as separate structured coordinates; never infer content identity from normalized or generated answer text.
- Train all-layer four-class linear readouts on the 1,801-item plain training split, choose content/label layers on the 600-item validation split, freeze the mechanism specification, and confirm it on all 599 test items.
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

- Retain literal duplicate answers and mark content identity ambiguous; never force an item-level mechanism label.
- Retain all baseline prompts. Mark only scientifically impossible single-factor transforms not applicable; parsing/anchor failures are fatal rather than row exclusions.
- Keep exact raw ties in artifacts but out of hard-winner probe fitting and classification.
- Resolve checkpoints on the complete tokenized prompt with offset mappings; reject missing offsets, mismatched IDs, instruction-overlapping content tokens, empty prompts, and wrong padding adjustment.
- Reject stale prompt hashes, wrong mappings, wrong model/tokenizer/code identity, conflicting duplicate shards, nonconverged probes, and mismatched frozen specifications.
- SIGINT/SIGTERM completes the current shard and writes progress. Training activation capture may restart because it is bounded; validation/confirmation readout and patching resume by deterministic work key.

## Design Simplicity

- One domain module owns scientific mappings, readout, probes, and patch rows; the CLI owns filesystem orchestration only.
- Existing scorer, residual capture/replacement, identity, and shard modules remain the mechanism owners.
- Hidden activations are bounded in memory and discarded after fitting/scoring. Only compact probe weights, probabilities, logits, patch effects, manifests, and timings persist.
- Rejected: database, registry/plugin framework, generic campaign system, model-specific hook hierarchy, persistent activation archive, and task-budget architecture.
- Paid-run cost remains task-local: live provider receipt, five-minute monitoring, explicit scale approval, and immediate task-owned teardown.

## Task Slices

1. Exact ledgers/mappings/checkpoints; focused RED-GREEN-REFACTOR; commit and push.
2. Batched all-layer capture, deterministic linear probes, coordinate selectivity, usability gates, and frozen layer selection; focused RED-GREEN-REFACTOR; commit and push.
3. Probe-subspace patches, controls, deterministic pair selection, compact shards, interruption/resume, and semantic identity; focused RED-GREEN-REFACTOR; commit and push.
4. CLI, local analysis, operator lifecycle, documentation, two-reviewer batch, coordinated fix/re-review, final suite, commit and push.

## Validation Ladder

- TDD: every behavior-changing slice begins with a focused test through the exported module or CLI and an observed expected RED failure.
- Source/smoke: imports, CLI help, ledger schema, deterministic fixture preparation.
- Real-use micro-proof: exact stored prompts and pinned tokenizers locally where available; eight-item production-entrypoint GPU canary for each pinned model.
- Focused tests: semantic mappings, tokenizer boundaries/padding, synthetic coordinate probes, projection math, target mapping, scalar/batched parity, identity/random controls, semantic identity, interruption/resume, duplicate conflict handling, and frozen confirmation.
- Review gate: freeze after focused tests and owner self-review; open both reviewers together; make no production edit or final-suite run while review is open; reconcile all findings together; apply one fix batch; run affected tests and focused re-review.
- Full gate: run the full suite exactly once after re-review closes and no further production edit is planned.

## Operator / Cost Preflight

- New/changed operator mode: add only the missing decision-binding run, monitor, fetch, and verify surface around existing causal operators.
- Lifecycle DAG: local prepare -> code-only sync -> remote preflight -> eight-item canary/resume proof -> explicit scale decision -> full run -> shard/manifest validation -> compact fetch -> local checksum verification -> task-owned teardown -> local analysis.
- Remote mode: BF16, SDPA, inference mode, one model per non-spot 80 GB H100 by default; non-spot 80 GB A100 fallback. No eager-attention reload or tuning campaign.
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
