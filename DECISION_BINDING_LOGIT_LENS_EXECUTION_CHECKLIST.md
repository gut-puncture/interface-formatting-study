# Mistral Two-Contract Logit-Lens Execution Checklist

This is the durable execution ledger for the approved Mistral-only logit-lens
experiment. Check boxes require live receipts. This document is not evidence
that a step happened, and it never authorizes access to the final 599 items.

## Release And Scope Lock

- [ ] Record branch, local HEAD, origin HEAD, and clean task diff.
- [ ] Record the reviewed release commit and push receipt.
- [ ] Confirm `SCIENTIFIC_NORTH_STAR.md` and this Run Card are committed and
  record their SHA-256 hashes.
- [ ] Confirm no stored prompt, wrapper, audit, split, causal artifact,
  candidate-reader artifact, or final-test artifact changed.
- [ ] Record the two independent reviewers, frozen diff, finding
  classifications, coordinated fix commit, and focused re-review closure.
- [ ] Record exactly one final full-suite command, pass count, elapsed time, and
  final stable commit.

## Frozen Scientific Decisions

- [ ] Population is exactly 1,801 train plus 600 validation items and nine
  formats per item; final-test count remains inaccessible to the runner.
- [ ] Scientific position is only the final non-padding token of each exact
  stored `Answer:` or `Answer: ` prompt; blocks are 0-31.
- [ ] Prompt contracts differ only in the authenticated terminal instruction.
- [ ] Raw letters are primary; content-free letter calibration is separate and
  secondary; no candidate calibration exists.
- [ ] Candidate surfaces are exact bytes with no normalization, variants,
  synonyms, chat template, BOS, EOS, or alternative-token mass aggregation.
- [ ] Candidate total path log probability is primary; token mean is length
  sensitivity; first token is a root lexical diagnostic; byte normalization is
  absent.
- [ ] Ties, duplicates, prefix collisions, shared first tokens, label-like
  candidates, surface drift, and incomplete mappings follow the Run Card's
  retained-but-explicit eligibility policy.
- [ ] Same-forward final/native tolerance is fixed at `0.02`; cached/full-prefix
  BF16 differences are finite recorded sensitivity values with no acceptance
  threshold; `0.04` remains only the analysis ambiguity flag.
- [ ] Primary contrasts, layers 0-30 AUC, bootstrap count 5,000, Holm scope,
  prior strata, timing algorithms, and interpretation categories are frozen.
- [ ] Freeze the positive-interpretation gates: resolved primary estimates;
  at least 200 and at least 80% of 2,401 unique items per contract; same nonzero
  candidate-total/token-mean direction under both contracts; and no more than
  20% layer-31 `0.04`-ambiguity in either plain or wrapped rows for raw-letter
  or candidate-total readouts under either contract.

## Authenticated Preparation

- [ ] Record authenticated Mistral causal run root and semantic identity.
- [ ] Record causal design, applicability, v2 readout, v3 prepared-manifest,
  option-audit, and choice-audit hashes.
- [ ] Run the committed CLI preparation command from the project root and paste
  its stdout receipt here:

  ```bash
  .venv/bin/python -m interface_formatting_study.decision_binding_logit_lens_cli prepare \
    --causal-run gpu_artifacts/causal_followup/mistral-7b-instruct-v0.3/c5bebfc04bc812355a2e/20260716T065500Z \
    --v2-bundle artifacts/decision_binding/v2/discovery/mistral-7b-instruct-v0.3 \
    --v3-bundle artifacts/decision_binding/v3_content/discovery/mistral-7b-instruct-v0.3 \
    --design-manifest artifacts/causal_followup/v3_source_preserving/design_train_validation.parquet.manifest.json \
    --output-dir artifacts/decision_binding/v4_logit_lens/discovery/mistral-7b-instruct-v0.3
  ```
- [ ] Record v4 bundle path, manifest hash, ordered work-key hash, item counts,
  block counts, prompt-contract counts, calibration count, and split counts.
- [ ] Verify the split set is exactly `{train, validation}` and the complete
  ledger contains no unknown/final/confirmation row.
- [ ] Record exact-surface-match, complete-content-map, override-provenance, and
  every preparation ineligibility count.

## Tokenizer Audit Before Model Load

- [ ] Cache only pinned Mistral revision after the reviewed release is on the
  host.
- [ ] Run the committed CLI tokenizer audit over the complete v4 bundle and
  paste its stdout receipt here:

  ```bash
  .venv/bin/python -m interface_formatting_study.decision_binding_logit_lens_cli audit-tokenizer \
    --profile mistral \
    --bundle artifacts/decision_binding/v4_logit_lens/discovery/mistral-7b-instruct-v0.3 \
    --output artifacts/decision_binding/v4_logit_lens/discovery/mistral-7b-instruct-v0.3/tokenization \
    --local-files-only
  ```

- [ ] Verify the token-audit directory contains
  `continuation_audit.parquet` and `tokenization_manifest.json`; record both
  hashes. Pass the directory, not either file, to `--token-audit`.
- [ ] Record tokenizer revision/hash, token-audit path/hash, total candidates,
  one/multi-token counts, maximum and percentile path lengths, Unicode,
  punctuation, label-like, duplicate, identical-token, shared-first-token,
  strict-prefix, decode-failure, special-token, and context-overflow counts.
- [ ] Verify contextual A/B/C/D are distinct single tokens.
- [ ] Verify the fixed-root Metaspace-without-implicit-prefix policy, count
  prompt-only round-trip mismatches, and verify every eligible appended path
  decodes to the decoded root plus the exact candidate surface. Confirm no row
  was silently removed.

## Local Implementation Proof

- [ ] Record RED evidence for input/final-set rejection, exact contextual
  tokenization, normalized lens, scalar oracle, teacher-forced sums, prefix
  tree/cache, parity, explicit eligibility, resume, manifest, hardware identity,
  analysis, and operator behavior.
- [ ] Record focused GREEN command and elapsed time.
- [ ] Verify shell syntax for all three logit-lens operator scripts.
- [ ] Verify runner uses `.venv/bin/python`, fixed Mistral, runtime batch knobs,
  token-audit binding, and optional max-chunks seam.
- [ ] Verify controller owns both bundle and token-audit arguments before
  signaling and surfaces the latest progress JSON.
- [ ] Verify fetch modes `partial`, `startup`, and `complete` all call the
  production verifier.
- [ ] Verify thin sync includes the new scripts and excludes local results,
  tests, caches, `.git`, and unrelated artifacts.

## Live Provider And Cost Receipt

- [ ] Refresh wallet, pods, disks, H100 80 GB availability, provider, region,
  non-spot status, hourly rate, storage, and Ubuntu image immediately before
  creation.
- [ ] Confirm no existing task-owned pod is live and identify every unrelated
  resource that must not be modified.
- [ ] Record selected resource ID, exact rate, three-hour maximum forecast,
  creation time, baseline wallet/charges, and teardown reserve.
- [ ] Create one task-named non-spot H100 80 GB Ubuntu pod only after all local
  gates above are checked.
- [ ] Record task-owned pod ID, SSH endpoint, key path presence (never contents),
  disk IDs, and provider creation receipt.

## Thin Deploy And Runtime Preflight

```bash
scripts/sync_interface_formatting_study_to_gpu.sh \
  ubuntu@<host> /home/ubuntu/interface_formatting_study <key> <port> \
  artifacts/decision_binding/v4_logit_lens/discovery/mistral-7b-instruct-v0.3/prompt_ledger.parquet

ssh -i <key> -p <port> ubuntu@<host> \
  'cd /home/ubuntu/interface_formatting_study && scripts/bootstrap_causal_followup_gpu.sh'

ssh -i <key> -p <port> ubuntu@<host> \
  'cd /home/ubuntu/interface_formatting_study && .venv/bin/python scripts/cache_causal_models.py --profile mistral'
```

- [ ] Record thin payload bytes and remote file inventory.
- [ ] Record Ubuntu, Python, Torch `2.7.1`, CUDA `12.6`, Transformers,
  tokenizers, BF16, SDPA, GPU name, compute capability, VRAM, disk-free, and
  pinned model/tokenizer revision receipts.
- [ ] Verify the remote git/source receipt matches the reviewed release bundle.
- [ ] Verify the real detached launch receives variable presence, correct cwd,
  and `.venv/bin/python`; do not print secret values.

## Single Startup, Partial Fetch, And Resume

Choose exactly eight blocks by tokenizer structure only and record their work
keys and coverage reasons. Do not inspect model outcomes to choose them.

```bash
ssh -i <key> -p <port> ubuntu@<host> \
  'cd /home/ubuntu/interface_formatting_study && \
   MAX_CHUNKS_THIS_INVOCATION=1 \
   scripts/control_decision_binding_logit_lens_gpu.sh start startup \
   <v4-bundle> <token-audit>'

ssh -i <key> -p <port> ubuntu@<host> \
  'cd /home/ubuntu/interface_formatting_study && \
   scripts/control_decision_binding_logit_lens_gpu.sh status'

ssh -i <key> -p <port> ubuntu@<host> \
  'cd /home/ubuntu/interface_formatting_study && \
   scripts/control_decision_binding_logit_lens_gpu.sh tail'
```

- [ ] Record startup semantic run ID, runtime configuration, selected work keys,
  first completed atomic chunk, elapsed time, throughput, utilization, peak
  VRAM, padding, and last durable shard.
- [ ] Fetch in `partial` mode and verify the partial receipt locally.
- [ ] Restart the identical startup command without
  `MAX_CHUNKS_THIS_INVOCATION`; record that completed work was skipped rather
  than recomputed.
- [ ] Fetch in `startup` mode and verify all eight blocks.
- [ ] Verify cumulative telemetry covers all eight exact work keys and the
  authenticated attempts prove `0 -> 4 interrupted` then
  `4 -> 8 startup_complete`; reject a one-shot startup receipt.
- [ ] Record the combined cached-batched versus scalar-full-prefix sensitivity
  maxima and per-layer argmax disagreement counts, plus per-token/path-total
  values, layer count, same-forward final/native parity, shape, finiteness,
  source hash, and resume receipts.
- [ ] Verify shard-local candidate eligibility against the authenticated
  continuation audit and require the exact canonical analysis artifact set,
  paths, and hashes before accepting a public `complete` verification.
- [ ] Record measured shutdown/flush/fetch duration and a full-run runtime/cost
  forecast from useful-block throughput.
- [ ] If any hard startup gate fails, stop before scale and diagnose that exact
  surface. Do not tune a cross-forward threshold or start a second canary.

Fetch form:

```bash
scripts/fetch_decision_binding_logit_lens_artifacts.sh \
  ubuntu@<host> <semantic-run-id> \
  /home/ubuntu/interface_formatting_study \
  gpu_artifacts/decision_binding_logit_lens \
  <key> <port> <partial|startup|complete>
```

## Full Mistral Run

Freeze and record `BATCH_SIZE` and `MAX_BATCH_TOKENS` from the successful
startup. The startup chunk size is predeclared as 4 so one invocation leaves
half of the exact eight-block sample for a real resume; the full-run chunk size
is predeclared as 64 to avoid thousands of tiny Parquet shards. Both values are
identity-bound. The startup cached-batched versus scalar-full-prefix comparison
must be complete, finite, and recorded before the larger full-run grouping is
allowed; its BF16 coordinate and argmax differences are sensitivity data, not
a fatal threshold.

```bash
ssh -i <key> -p <port> ubuntu@<host> \
  'cd /home/ubuntu/interface_formatting_study && \
   BATCH_SIZE=<n> MAX_BATCH_TOKENS=<n> CAPTURE_CHUNK_SIZE=<n> \
   scripts/control_decision_binding_logit_lens_gpu.sh start full \
   <v4-bundle> <token-audit>'
```

- [ ] Record full semantic run ID and prove its identity cannot consume startup,
  A100, different-runtime, different-token-audit, or different-source shards.
- [ ] At each monitor point record completed/total blocks, root/branch tokens,
  throughput, padding, root/branch utilization, peak VRAM, phase timing, last
  shard, ETA, elapsed lifetime, live charges, forecast total, and reserve.
- [ ] Diagnose an unexpected exit from the owned log before restarting; never
  start a possible duplicate writer.
- [ ] Stop gracefully if source/parity/finiteness/identity/telemetry fails or if
  the full forecast cannot fit inside the three-hour ceiling plus teardown.
- [ ] Record final remote complete status, block/layer/contract counts, parity
  maxima, elapsed time, peak VRAM, and total provider cost.

## Fetch, Verification, And Analysis

- [ ] Fetch the full run with mode `complete`.
- [ ] Record local destination, manifest hash, shard count, merged row count,
  exact Cartesian reconciliation, exclusion counts, parity report, and verifier
  exit receipt.
- [ ] Generate and independently verify `LOCAL_SHA256SUMS.txt`.
- [ ] Run analysis locally from only the verified merged artifact and frozen
  `analysis_spec.json`; record the exact command and deterministic seed.
- [ ] Record both co-primary estimates, 95% item-cluster intervals, Holm result,
  total/mean agreement, first-token diagnostic, train/validation replication,
  predeclared strata, and coverage counts.
- [ ] Verify and hash `quality_gates.json`; no positive interpretation is valid
  unless every frozen gate passes. Record every failed gate without changing a
  threshold or dropping affected rows.
- [ ] Record paths/hashes for the token census, 2x2 trajectory figure,
  sensitivity table, timing distributions, strata table, parity/reproducibility
  table, analysis summary, and bounded interpretation memo.
- [ ] Select exactly one declared interpretation category, using the exact
  artifact label `predeclared_stratum_heterogeneous` for qualified
  heterogeneity, and copy its bounded claim from `interpretation_memo.md`. Do
  not open the final 599, tune another score, or invent another probe.

## Teardown And Closeout

- [ ] Confirm complete local verification before normal teardown.
- [ ] Terminate only the recorded task-owned pod and any task-owned ephemeral
  disk; never act on an unrelated resource.
- [ ] Refresh provider pods/disks/wallet and record empty task-owned inventory,
  termination time, final posted charge, rate-times-lifetime estimate, and the
  larger accounting value.
- [ ] Confirm no credential value, model cache, protected final data, or private
  artifact was committed or included in the thin payload.
- [ ] Push the final verified code/document checkpoint and record origin receipt.
- [ ] Record the exact next paper action. Phi/Qwen, final-599 confirmation,
  another probe, or paper-wide claim expansion requires a separate decision.
