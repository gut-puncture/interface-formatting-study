# Decision-Binding Candidate-Local Content Reader Run Card

## Owner Scope Box

- Existing path reused: exact stored prompts, finalized audited option maps, discovery splits, model loaders, all-layer capture, atomic shards, semantic identity, GPU operator, artifact fetch, and checksum verification.
- New production code: one narrow candidate-local ranker is required because the completed common-checkpoint content classifier did not independently identify answer content.
- Expected scope: one focused domain module, a narrow CLI/operator extension, and focused tests; about 350-500 net production lines and four to six local implementation hours.
- Review: exactly two independent reviewers, one for scientific/data semantics and one for runtime/resume/operator behavior; maximum two review rounds.
- Final validation: one full local suite after review closes; first real proof is a small Mistral canary followed by the full Mistral discovery gate only if the canary is exact.
- Deviation gate: pause before 500 net production lines, six implementation hours before review, a second new production module, any change to stored prompts/audits/splits, or any redesign of the existing decision-binding architecture.

## Source Truth Read

- The approved lean content-reader plan in the owning Codex task.
- `AGENTS.md`, `DECISION_BINDING_RUN_CARD.md`, `README.md`, and the completed three-model v2 artifacts.
- `decision_binding.py`, `decision_binding_cli.py`, `causal_option_maps.py`, `causal_option_audit.py`, current GPU scripts, and focused tests.
- Finalized option-audit labels, exact discovery prompts, frozen 1,801/300/300/599 roles, and existing model-specific readers.

## Approved Behavior

- Reuse every existing prompt byte-for-byte. Locate candidate content only from source-hash-bound audited option spans and deterministic existing transforms; regenerated prompt hashes must match stored hashes exactly.
- At every transformer layer, read the hidden state at each candidate answer's final content-bearing token. The token must overlap the audited payload and must not be the option label, delimiter, newline, or answer instruction.
- Fit one shared linear scoring vector per layer. Score the four candidate-local states and train with listwise four-way cross-entropy to rank the model's raw winning answer content.
- Fit only on the 1,801 discovery-training items using the exact plain controlled baseline. Select the layer and L2 penalty only on the existing 300-item `layer_select` role. Use the existing 300-item `reader_gate` role once for the internal Mistral stop/go decision.
- Freeze model ID/revision, prompt and audit hashes, split hash, layer, token-position rule, L2 grid, selected L2, training order/seed, normalization, control definitions, metrics, confidence procedures, and gate thresholds before opening the 599-item final set.
- Mistral is the strict first gate. Phi and Qwen run only if the unchanged method passes the predeclared Mistral gate. Only models passing their internal gate may open their untouched 599-item confirmation set.
- Protected meaning: success demonstrates linear decodability of the model's selected answer content at candidate-local option states. It does not demonstrate factual knowledge, correctness, a unique circuit, origin layer, or causation.

## Non-goals

- No new prompts, wrapper repairs, parser campaign, answer generation, model fine-tuning, nonlinear probe, PCA, ensemble, calibration, attention/head/neuron search, causal patching, or paper edit.
- Do not alter the completed position/label readers, `ProbeBank`, probe-subspace patcher, or v2 artifacts.
- Do not optimize toward near-100% accuracy when the model's own decision is uncertain, tied, or wrong.

## First Vertical Slice

- Behavior/invariant: an exact stored prompt plus its audited source option map deterministically yields four candidate content-token endpoints or fails closed with a diagnostic; no prompt is silently excluded.
- Entry point: exported preparation seam in `interface_formatting_study.decision_binding_content`.
- Data/fixture: representative audited source prompts including prose, CSV, protobuf, shell, escaped text, repeated text, Unicode, and multi-token options.
- Proof path: focused tests first fail for the missing seam, then prove exact prompt/hash reproduction, correct representation choice, tokenizer overlap, and fatal mismatch behavior.

## Files And Interfaces

- Likely touched: new `src/interface_formatting_study/decision_binding_content.py`, narrow additions to `decision_binding_cli.py` and existing GPU/fetch/verify scripts, one focused test module, and standing documentation after validation.
- Public entrypoints: a content-reader preparation/training/gate CLI mode and a matching GPU operator mode; v3 artifacts remain separate from v2.
- Legal states: prepared, capturing, selected, frozen, gated, confirmed, stopped, failed, complete. Final confirmation cannot start without the exact frozen specification hash.
- Trust boundaries: stored prompt bytes, source-hash-bound audits, pinned tokenizer/model revision, split roles, identity-bound shards, provider inventory, compact fetched artifacts, and checksums.
- Compatibility: v2 readers and artifacts are read-only inputs. No migration or overwrite is permitted.

## Scientific Invariants And Failure Modes

- The training target is the model's raw unique winning option, not the reference answer and not the calibrated choice. Exact raw ties remain in artifacts but are ineligible for hard-target fitting/accuracy and are counted explicitly.
- Canonically identical displayed answers remain in artifacts and are excluded only from content-identity metrics, with explicit counts.
- Each option endpoint is the final tokenizer token whose offset overlaps the selected audited content representation. Token crossing an adjacent delimiter is rejected rather than silently accepted.
- For formats with symbolic aliases, select the actual content-bearing representation (for example, protobuf answer text rather than enum number, shell assignment text rather than `$OPTION_A`). If multiple complete content-bearing representations remain, use the last one in prompt order; ambiguity outside that rule is fatal.
- New forward passes must reproduce the stored raw categorical winner for every unique-winner prompt used with prior artifacts. A mismatch stops the run; rows are not dropped.
- Prompt hash, audit/source hash, tokenizer identity, model revision, split hash, selected specification, and work key are validated on prepare, resume, merge, fetch, and confirmation.
- Duplicate identical shards merge once; conflicting duplicates fail. SIGINT/SIGTERM finishes the current shard, writes progress atomically, and resumes by deterministic work key.
- Fail closed on missing spans, missing tokenizer offsets, padding/index mismatch, malformed option maps, stale identity, non-finite loss/weights, failed convergence, or attempted final-set access before freeze.

## Reader And Controls

- Per item/layer, center each candidate state by the four-candidate mean and divide by a training-only scalar layer RMS. Fit one no-intercept vector using listwise cross-entropy plus L2.
- Use BF16 for inference/temporary activation storage and float32 for fitting/scoring. Fixed L2 grid: `1e-4, 1e-3, 1e-2, 1e-1`. Fixed deterministic seed and item ordering.
- Compare against majority, position-only, displayed-label-only, additive position-plus-label, and answer-length controls. Fit three deterministic random-target rankers using the same activations.
- Selection data: choose only among candidates that beat isolated position and label controls on both isolated transformation arms. Maximize worst-arm top-1 accuracy; break ties by within-question AUC, then stronger regularization, then earlier layer.
- Statistical proof: item-cluster bootstrap with 5,000 replicates and 1,000 within-item target permutations. Report overall, isolated-position, isolated-label, stable-pair preference, incorrect-decision stratum, conflict stratum, and wrapper strata with sample sizes.

## Frozen Internal Gates

- Tier 1, meaningful decodability: overall and both isolated arms exceed their permutation null with lower 95% bootstrap bounds above 25%; stable-pair content preference lower bound exceeds 50%; paired improvement over the strongest nuisance control has lower bound above zero; random-target readers remain at chance; incorrect-decision stratum exceeds chance.
- Tier 2, strong operational readout: overall point accuracy at least 90% with lower bound at least 85%; each isolated arm at least 85% with lower bound at least 75%; both stable-pair preferences at least 90% with lower bounds at least 80%; at least 10 percentage points above the strongest nuisance control; conflict point accuracy at least 85% and above chance; adequately sized wrapper strata have point accuracy at least 75% and lower bounds above 50%; save/load, batch, and seed parity pass.
- Tier 1 failure stops after verified Mistral artifacts. Tier 1 pass but Tier 2 failure remains descriptive and stops before Phi/Qwen or final confirmation. Tier 2 enables the unchanged method for Phi and Qwen; it does not authorize causal patching.

## Task Slices And Validation Ladder

1. Audited candidate endpoint preparation and compact sidecar; RED-GREEN-REFACTOR; focused tests; commit and push.
2. Batched candidate-local capture, shared-vector ranker, normalization, and nuisance/random controls; RED-GREEN-REFACTOR; focused tests; commit and push.
3. Layer/L2 selection, frozen specification, bootstrap/permutation metrics, and Tier 1/Tier 2 gate; RED-GREEN-REFACTOR; focused tests; commit and push.
4. Narrow run/resume/fetch/verify operator path and separate v3 artifacts; RED-GREEN-REFACTOR; focused tests; commit and push.
5. Owner self-review; freeze the diff; open the scientific and operator reviewers together; classify all findings together; apply at most one coordinated fix batch; focused re-review; run the full suite once.

## Operator And Cost Preflight

- Lifecycle: local prepare -> thin code/data sync -> remote import/CUDA/model/tokenizer/endpoint micro-check -> small Mistral changed-surface canary -> verified canary artifacts -> full Mistral discovery capture/fit/select/gate -> compact fetch/checksum verification -> stop/go decision -> immediate task-owned teardown or unchanged Phi/Qwen continuation.
- Target hardware: one non-spot 80 GB A100-class GPU on Ubuntu, BF16 and SDPA. H100 is out of scope unless A100 supply is unavailable and a live cost/throughput comparison justifies it.
- Canary: about 32 items spanning wrappers, transformations, formats, multi-token/escaped/Unicode/repeated content, exact winner reproduction, shard/resume, and artifact verification. It proves functionality, not scientific efficacy.
- Telemetry: phase, work completed/total, throughput, padding, preparation/forward/write time, peak VRAM, fitting time, last shard, ETA, errors, provider spend, and stop reason.
- Cost forecast before launch: Mistral target 15-30 paid minutes and under $1; all three discovery runs about $2-3; eligible confirmations about $1-2 more. Pause if the measured forecast materially exceeds this or implementation needs new architecture.
- No scale run begins until the exact command, checkpoints, monitor, stop/resume, fetch, strict verifier, and task-owned pod/disk teardown are locally proven.

## Review, Docs, And Stop Rules

- Review rejection flags: prompt-byte drift; source/audit mismatch; position/label leakage presented as content; data leakage across roles; final set read before freeze; silent exclusions; checkpoint mismatch; non-reproducible selection; stale resume; missing teardown; or any causal claim.
- Evidence for Shailesh: exact row/item counts, zero silent prompt loss, sample endpoint audit, selected layer/L2 and frozen hash, control comparison, gate report with confidence intervals, exact cost/runtime, checksums, and final empty task-owned inventory.
- Update `README.md` and operator/reproducibility documentation only after final validation and real-run outcome. Do not edit the paper in this task.
- Stop for missing source truth, an irreproducible stored winner, any required architecture redesign, more than the scoped production size/time, a second material defect in the same design class, materially higher paid forecast, or inability to monitor/fetch/terminate safely.
