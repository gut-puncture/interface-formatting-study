# Mistral Two-Contract Logit-Lens Run Card

## Owner Scope Box

- Existing path reused: authenticated Mistral causal prompts and audits, pinned
  model/tokenizer loader, transformer-block hooks, semantic identity, atomic
  shards, Ubuntu bootstrap, model cache, thin sync, PID operator, and strict
  local artifact verification.
- New production code is limited to one Mistral logit-lens scorer, its CLI and
  analysis, three thin operator scripts, focused tests, and these durable run
  documents. The causal and candidate-reader systems remain unchanged.
- The original 1,400-2,000-line estimate was rechecked when the implementation
  exceeded it. The final bounded scorer, authenticated CLI/verifier, and frozen
  analysis are approximately 3,300 non-test lines because every eligibility,
  resume, parity, and paper-output rule is explicit. No shared loader/shard
  redesign or additional subsystem is authorized.
- Review tier: exactly two independent reviewers in one batch, one for
  scientific identity/scoring/analysis and one for cache/resume/operator
  behavior. One coordinated correction and focused re-review are allowed.
- Validation: focused tests during implementation, then one full suite after
  review closes. First paid proof is one eight-block Mistral startup and resume
  check, followed directly by the full run if it passes.

## Scientific Contract

This is a paired two-contract by two-readout descriptive measurement grid, not
a causal factorial experiment. The manipulated variable is the terminal output
instruction. Letter versus candidate scoring is a measurement choice.

The experiment covers exactly the existing 2,401 internal train/validation
items: 1,801 train and 600 validation, with nine formats per item. The 599
internal final-test items remain inaccessible. There is no training, layer
selection, threshold fitting, or result-dependent scoring choice.

For every authenticated item-format block, measure:

1. the exact existing letter-only prompt and the matched prompt requesting the
   exact answer text;
2. raw A/B/C/D continuations under both prompt contracts, mapped through the
   audited displayed-position-to-content identity;
3. exact displayed candidate-answer continuations under both contracts; and
4. the existing layerwise content-free letter calibration as a secondary,
   separately reported analysis.

The text-output contract is a counterfactual task. It is not privileged access
to what the letter-output prompt was thinking. The lens is an output-head
projection of intermediate states, not literal access to thought and not proof
of causation or a controlling circuit.

## Protected Inputs And Observation Point

- Preparation accepts only authenticated Mistral causal baseline, matched
  exact-answer-text, and content-free calibration rows from train/validation.
- It must account for 2,401 unique items and 21,609 item-format blocks, with
  nine formats per item. Unknown or protected splits fail closed.
- Prompt bytes, prompt hashes, causal semantic identity, design/applicability
  hashes, option/choice audits, prior readout identities, work-key order, and
  governing source hashes are bound into a separate v4 bundle.
- The only scientific observation point is the final non-padding token of the
  exact stored prompt ending in `Answer:` or `Answer: ` (the source-preserved
  trailing-space difference is retained). Transformer blocks are indexed 0-31;
  no embedding pseudo-layer is added.
- End-of-question, option endpoints, and end-of-options are not scientific
  measurements. A shared pre-instruction boundary may be checked only as a
  technical causal-prefix invariant.

Every source block remains represented. Incomplete four-way identity proof or
wrapper-versus-plain surface drift makes the affected cross-wrapper comparison
ineligible; it does not delete the block or its local scores.

## Frozen Lens And Continuation Semantics

At every block output, apply Mistral's existing final RMSNorm and language-model
head, then compute log-softmax. Retain only declared continuation-token scores;
never persist hidden states or full-vocabulary logits.

Letter scores use the exact contextual A/B/C/D continuations. Tokenizer audit
must prove four distinct single-token continuations. Raw letter scores are
primary. Existing content-free bias is subtracted layer by layer only for the
separate calibrated-letter sensitivity.

Candidate scoring preserves the exact audited displayed bytes:

- no trimming, Unicode normalization, case or punctuation variants, synonyms,
  paraphrases, chat template, BOS, EOS, or alternative-tokenization sum;
- `add_special_tokens=False`; the exact prompt string and exact resulting root
  token IDs are both hash-bound. Mistral's implicit standalone Metaspace prefix
  is disabled only for continuation encoding because the fixed root already
  contains every real boundary byte;
- the appended path must decode to the decoded fixed root plus the exact
  candidate surface. Prompt-only round-trip differences such as a tokenizer
  dropping an initial space are counted and hashed, not silently repaired;
- pinned tokenizer identity and no registered special token in a candidate;
- canonical token IDs, decoded suffix, UTF-8 hash, token count, byte count, and
  eligibility reason are persisted before model scoring;
- a failed/empty/overflow continuation remains in the ledger and is explicitly
  ineligible. It is never truncated or repaired.

For candidate tokens `c_1 ... c_m`, layer `l` stores:

`candidate_path_total_logp = sum_t log p_l(c_t | prompt, c_<t)`

The first-token term comes from the common `Answer:` state. Later terms use
candidate-specific teacher-forced positions. The total is complete path-prefix
compatibility, not stopping probability and not information contained entirely
in the original root state.

Frozen score families are total path log probability (primary), mean token log
probability (length sensitivity), and first-token log probability (root lexical
diagnostic where all four first tokens are distinct). Four-way restricted
softmax values are for readability; unnormalized scores remain authoritative.
No byte-normalized score or result-dependent scoring rule is permitted.

## Prefix Tree, Parity, And Eligibility

Production scoring performs one root prefill per exact prompt batch, builds a
four-candidate token trie, reuses shared prefixes and Mistral `DynamicCache`,
batches equal-depth divergent branches, and discards temporary states after
projection. An uncached scalar scorer is the correctness oracle, not the
production full-run path.

Frozen tolerances:

- final block lens/native target log probabilities: maximum absolute coordinate
  difference `0.02`;
- cached/uncached and scalar/batched target-token log probabilities: `0.02`;
- candidate total difference: at most `0.02 * token_count`;
- token IDs, prompt/source hashes, shapes, layer count, and finite values have
  zero tolerance.

Historical stored/fresh winner changes do not invalidate a faithful run.
Same-forward lens/native disagreement beyond tolerance does.

Duplicate displayed answers, identical token sequences, strict candidate-prefix
collisions, label-like candidates, shared first tokens, incomplete semantic
mapping, and changed paired surfaces are retained with explicit flags. They are
excluded only from the comparisons whose identity assumption they violate.
Exact ties store the full argmax set and zero margin; no order-based tie-break
is allowed.

## Frozen Analysis

All 2,401 permitted items are measured. Pooled train/validation is the primary
descriptive population because no result is fitted; train and validation are
also reported separately as a fixed replication check.

Within each item, average the eight eligible wrapped-versus-plain comparisons
before aggregating items. Winner agreement is Jaccard overlap of complete
argmax sets. Layers 0-30 form the pre-final trajectory summary; layer 31 remains
visible as the native endpoint and parity anchor but is excluded from AUC.

The two co-primary contrasts are candidate-total plain/wrapped agreement AUC
minus raw-letter agreement AUC, separately under the letter-output and
answer-text contracts, using the same eligible population. Secondary results
include full agreement/Jensen-Shannon/margin trajectories, token-mean and
first-token sensitivities, calibrated-letter trajectories, and fixed-contract
comparisons. Absolute candidate-versus-letter probability magnitudes are not
compared.

Winner-stability layer is the earliest layer 0-30 whose unique winner remains
the same through layer 31. Plain/wrapped separation onset is defined only when
both final winners are unique and different, and is the earliest layer from
which both remain fixed to their respective final winners. The numerical
ambiguity reference is `0.04`; it flags sensitivity but does not exclude rows.

Prior strata come only from the authenticated causal artifact: wrapper,
stable/conflict status, calibrated confidence tertile, correctness transition,
position/label susceptibility and overlap, existing answer-text/letter
agreement, incorrect wrapped decisions, surface match, and audit provenance.
They cannot select layers or populations.

Use 5,000 deterministic item-cluster bootstrap replicates, preserving every
wrapper/contract/readout/layer within an item. Report 95% intervals, Holm-adjust
only the two co-primary contrasts, and use simultaneous trajectory bands rather
than 31 separate discoveries. Every output reports item/block/exclusion counts.

Interpretation is limited to: consistent with later output/binding effects,
consistent with wrapper-dependent formation, contract-sensitive,
`predeclared_stratum_heterogeneous`, or uninformative. Before any label other
than uninformative is allowed, all four frozen quality gates must pass:

1. Each contract has exactly one finite primary estimate with a finite,
   ordered 95% interval, Holm value, positive item count, and positive pair
   count.
2. Each contract represents at least 200 unique items and at least 80% of the
   2,401-item permitted population (therefore at least 1,921 items here).
3. Candidate-total and candidate-token-mean AUC contrasts have the same
   nonzero direction separately under both prompt contracts.
4. At layer 31, the `0.04` ambiguity rate is at most 20% separately for plain
   and wrapped rows for both raw-letter and candidate-total readouts under both
   contracts.

These thresholds are frozen before GPU outcomes. They do not remove rows or
change the primary estimates; they only prevent a fragile, weakly covered, or
internally inconsistent result from receiving a positive mechanism label.

## Runtime, Resume, And Artifact Contract

Public CLI commands are `prepare`, `audit-tokenizer`, `run-model`, `analyze`,
and `verify`. The operator fixes `--profile mistral`. Batch size, maximum batch
tokens, and capture chunk size remain runtime configuration and are recorded in
identity; no model-profile batching branch exists.

One atomic work unit contains both prompt contracts, its calibration prompt,
all 32 layers, and all four candidates for one item-format block. Shards use
deterministic work keys and atomic writes. Exact duplicate shards merge once;
conflicting overlaps fail. `--max-chunks-this-invocation` is an execution-only
stop/resume seam and does not change scientific identity.

The eight-block startup uses four-block persistence chunks, so the first
max-chunks invocation leaves exactly four blocks for a genuine resume. The
full run uses 64-block persistence chunks to avoid thousands of tiny files.
Both are identity-bound runtime choices; batch/scalar parity is established in
startup before the larger full-run grouping is used.

Identity binds the source/tokenization/scoring/analysis policies; model and
tokenizer revisions; GPU name and compute capability; Torch/CUDA/Transformers/
tokenizers versions; BF16/SDPA; and runtime batch/token/chunk configuration.
A100 and H100 shards therefore cannot mix. Resume must authenticate every
existing shard before skipping work.

Required outputs are semantic identity, tokenization manifest, continuation
audit, work plan, progress and attempt receipts, score shards, merged layerwise
scores, parity report, frozen analysis specification, analysis summary,
deterministic `quality_gates.json`, bounded `interpretation_memo.md`, paper-ready
tables/figures, run manifest, and independent local checksums.

## Operator Lifecycle And Paid Gate

Lifecycle DAG:

`local prepare -> reviewed release -> thin sync -> Ubuntu bootstrap -> pinned
Mistral cache -> complete tokenizer audit -> one startup chunk -> partial fetch
and verify -> identical resume -> startup verify -> full run -> complete fetch
and local verify -> local analysis -> task-owned teardown proof`

Target one non-spot H100 80 GB on Ubuntu with Torch 2.7.1/CUDA 12.6, BF16, and
SDPA. The paid-time ceiling is three H100-hours including setup, startup,
full-run, fetch, and teardown reserve. Provider availability, rate, wallet,
task-owned pod/disk inventory, and exact release are refreshed immediately
before creation. Never terminate an unrelated resource.

The sole startup sample is eight blocks selected from tokenizer structure, not
model outcomes, covering one- and multi-token candidates, shared and immediate
branches, Unicode/punctuation, a long path, and a retained ineligible collision
where available. It runs through the production entrypoint. Scale immediately
after source, token, cache/scalar, batch/scalar, final-native, resume, shape,
finiteness, telemetry, and fetched-artifact verification pass. Do not run a
second canary.

Monitor completed/total blocks, root and branch tokens, throughput, padding,
root/branch GPU utilization, peak VRAM, phase timing, last durable shard, ETA,
provider price times elapsed lifetime, current charges, and teardown reserve.
Unexpected exits are diagnosed from the canonical log before any restart.

## Validation And Stop Rules

Implementation uses RED -> GREEN -> REFACTOR through public seams. After all
focused tests pass, the owner self-reviews and freezes the diff. Both independent
reviewers run together; no implementation edit or final suite occurs while the
batch is open. Findings are reconciled once, accepted fixes receive focused
re-review, and the complete suite runs exactly once on the final stable diff.

Stop before paid scale for any protected-split access, prompt/audit/hash drift,
silent row loss, unresolved identity ambiguity in a primary comparison, token
round-trip failure without an explicit persisted state, parity failure,
non-finite value, stale/mixed resume, missing telemetry, unverifiable fetch,
unknown task-owned teardown, or cost forecast beyond the approved ceiling.
Do not widen tolerances, change score rules, add a new probe, or inspect the 599
final items in response to an unattractive result.
