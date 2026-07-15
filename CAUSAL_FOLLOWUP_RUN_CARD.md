# Causal Follow-up Run Card

## Scientific contract

This is an exploratory causal follow-up on the fixed train and validation
partitions. The 599 internal-test items remain untouched until the explanation
and analysis are frozen.

- Active design: 2,401 items, eight wrappers, 96,040 scored rows per model.
- Letter arm: four randomized-block assignments per item/wrapper. Every answer
  text appears once in every physical position and once under every answer
  letter. Position and letter schedules are randomized independently from a
  stable item/wrapper hash. Variant 0 is the unchanged identity assignment.
- Text arm: one identity-assignment prompt per item/wrapper scored against the
  four exact answer texts. Total sequence log-likelihood is primary; mean token
  log-likelihood is a declared length-sensitivity analysis.
- Models: pinned Qwen2.5-1.5B, Phi-3.5-mini, and Mistral-7B-v0.3 profiles.
- Active design SHA-256:
  `435d9662f9c1f06e3bf4e7f5334f12fc86f287b6d7ab6794ae02c1dd4fe4f8f7`.
  The launch operator must re-read the committed manifest rather than trust
  this copied value if the design is regenerated.

The older wrapper audit is outcome-blind. Its first pass covered all 24,000
rows exactly; an independent second pass adjudicated all 134 initially
questionable rows. Final labels are 22,081 formatting-only, 1,808
meaning-preserving rewrites, 107 content changes, and four ambiguous rows. The
causal design does not reuse those generated prompt strings: it renders directly
from the canonical question and four canonical choices. This is deliberate so
the new causal evidence is not contaminated by the 111 non-preserving or
uncertain old rows.

## Local readiness gate

Before rental, all of the following must be green:

1. Full local tests from the project root.
2. Causal design checksum and 96,040-row/2,401-item contract.
3. Functional fake-model run, interruption/resume equality, shard conflict
   rejection, verified fetch, and local analysis consumer tests.
4. Thin payload inventory: `src/`, `configs/`, three causal operator scripts,
   packaging metadata, README, and the 13 MB active design plus manifest only.
5. Independent review wave reconciled; no unresolved correctness blocker.
6. No unexpected live Prime pod. Never terminate another project's pod.

## Resource and budget

- Campaign target: $6.00; absolute cap: $7.00.
- Forced stop begins by $6.75 to retain $0.25 for billing lag and teardown.
- Accounting starts at pod creation and uses the larger of provider-posted
  campaign charges and launch-price multiplied by elapsed pod lifetime.
- Read-only inventory on 2026-07-15 found a non-spot A6000 48 GB candidate at
  $0.54/hour (`ea69b8`, Massed Compute, US, 256 GB local disk). Resource IDs,
  stock, and price must be refreshed immediately before creation.
- Use one GPU sequentially. Do not attach or create a persistent disk for the
  non-spot run. Fetch after every completed model.
- Use Prime image `cuda_12_6_pytorch_2_7` and reuse its Torch. If the chosen
  provider rejects that image, do not silently fall back or reinstall Torch on
  paid time; return to availability and select the cheapest compatible 48 GB
  non-spot resource.

At the last inventory there was one active, unrelated TWC RTX 6000 Ada pod.
That pod is not part of this campaign and must not be modified. A new Keyhole
pod may be created only after the owner confirms the existing pod is expected
or it is independently gone.

## Exact launch sequence

Values in angle brackets are filled from the refreshed Prime response. Create
a dedicated Keyhole SSH key before the pod; do not reuse the TWC key currently
configured in Prime.

```bash
prime --plain wallet --output json
prime --plain pods list --output json
prime --plain disks list --output json
prime --plain availability list --output json
prime config set-ssh-key-path <keyhole-private-key>
prime --plain pods create \
  --id <fresh-48gb-nonspot-resource-id> \
  --name keyhole-causal-followup-<utc-stamp> \
  --disk-size 200 \
  --image cuda_12_6_pytorch_2_7 \
  --yes
```

After ACTIVE status, record pod ID, exact hourly price, creation time, baseline
wallet, and SSH endpoint. Prove SSH and CUDA before syncing.

```bash
ssh -i <key> -p <port> ubuntu@<host> 'whoami; nvidia-smi; df -h'
scripts/sync_interface_formatting_study_to_gpu.sh \
  ubuntu@<host> /home/ubuntu/interface_formatting_study <key> <port> \
  artifacts/causal_followup/v1/design_train_validation.parquet
ssh -i <key> -p <port> ubuntu@<host> \
  'cd /home/ubuntu/interface_formatting_study && scripts/bootstrap_causal_followup_gpu.sh'
ssh -i <key> -p <port> ubuntu@<host> \
  'cd /home/ubuntu/interface_formatting_study && python scripts/cache_causal_models.py'
```

## Canary, forecast, and full-run gates

Run functional then profiling canaries for Mistral, Phi, and Qwen. Mistral is
first because it is the slowest and highest-memory checkpoint.

```bash
scripts/run_causal_followup_gpu.sh mistral functional
scripts/run_causal_followup_gpu.sh phi functional
scripts/run_causal_followup_gpu.sh qwen functional
scripts/run_causal_followup_gpu.sh mistral profiling
scripts/run_causal_followup_gpu.sh phi profiling
scripts/run_causal_followup_gpu.sh qwen profiling
```

The profiling canary is 32 items = 1,280 durable rows. For each model record
wall time, forward time, forward calls, peak VRAM, padding ratio, completed
rows/second, and input preparation time. Forecast each full run as measured
seconds per durable row times 96,040, then apply a 1.25 P90 multiplier and add
measured setup time. Start full scale only if combined P90 cost fits below
$6.00. Otherwise perform at most one batch/token-budget optimization canary;
accept it only with exact categorical parity and at least 10% end-to-end gain.

Full order:

```bash
scripts/run_causal_followup_gpu.sh mistral full
# fetch and verify Mistral
scripts/run_causal_followup_gpu.sh phi full
# fetch and verify Phi
scripts/run_causal_followup_gpu.sh qwen full
# fetch and verify Qwen
```

SIGINT/SIGTERM finishes the current shard and records progress. Repeating the
same command resumes only the same semantic identity. Do not alter batch size
or token budget after a full semantic run begins.

## Monitoring and teardown

- Check cost after setup, every canary, every model, and at least every five
  minutes; check every minute after $5.50.
- At $6.00, continue only if less than 10% of canary-weighted GPU work remains
  and its P90 cost plus shutdown fits below $6.75.
- At $6.75 or if monitoring fails: send SIGTERM, wait for shard flush, fetch in
  partial mode, and terminate the Keyhole pod.
- Never terminate the unrelated TWC pod.

Fetch command per model:

```bash
scripts/fetch_causal_followup_artifacts.sh \
  ubuntu@<host> <model-slug> <semantic-run-id> \
  /home/ubuntu/interface_formatting_study gpu_artifacts/causal_followup \
  <key> <port> complete
```

Only after all required model fetches validate:

```bash
prime --plain pods terminate <keyhole-pod-id>
prime --plain pods list --output json
```

Local analysis uses `analysis/analyze_causal_followup.py`; plotting, paper
editing, compilation, and archiving stay on the Mac.
