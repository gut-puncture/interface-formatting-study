# Causal Follow-up Run Card

## Scientific contract

This is an exploratory causal follow-up on the fixed train and validation
partitions. The 599 internal-test items remain untouched until the explanation
and analysis are frozen.

- Active design: all 2,401 source items are retained. Eight exact stored wrapper
  baselines plus one matched plain MCQ yield 21,609 item-format blocks and
  172,506 durable rows per model. Both intervention arms are available for
  21,548 blocks. The 61 blocks with no independent answer labels keep their
  baseline and answer-text rows but do not invent position or label treatments.
- Letter arm: one controlled baseline, three position-only rotations (the
  displayed letters stay attached to their answer texts), and three
  letter-only rotations (physical order stays fixed) per item/format. The two
  causes are never changed together. Raw and content-free-calibrated outcomes
  are both saved; calibrated accuracy is primary because it matches the paper.
- Text arm: one controlled-baseline prompt per item/format. Deterministic
  generated exact-answer match is primary. Total and mean candidate
  log-likelihood are explicitly secondary length-sensitive diagnostics.
- This is a controlled causal experiment over the exact prompts used in the
  observational runs. Baselines are byte-identical. Counterfactuals change only
  the unambiguous stored answer texts/labels or the terminal readout instruction.
  When one exact prompt has no independent labels, only its inapplicable
  position/label rows are absent; no item or baseline is excluded.
- Models: pinned Qwen2.5-1.5B, Phi-3.5-mini, and Mistral-7B-v0.3 profiles.
- Active design SHA-256:
  `2a48fa95dd7e22d354e4597212accf0a9f5774107a386a082a0aee5a8b80d4dd`.
  The launch operator must re-read the committed manifest rather than trust
  this copied value if the design is regenerated.

The older model-assisted wrapper audit remains stored. Its 111 non-formatting
rows identify the bounded set whose displayed option order/text needs explicit
source-bound bookkeeping; it never changes prompt bytes or decides whether an
item is retained. The applicability ledger records the exact 61 block-level
position/label omissions.

## Local readiness gate

Before rental, all of the following must be green:

1. Full local tests from the project root.
2. Causal design checksum and 172,506-row/2,401-item contract, plus the
   checksum-bound 21,609-row applicability ledger.
3. Functional fake-model run, interruption/resume equality, shard conflict
   rejection, verified fetch, and local analysis consumer tests.
4. Thin payload inventory: `src/`, `configs/`, causal operator scripts, pinned
   non-Torch environment, packaging metadata, README, and the 34 MB active
   design, manifest, and applicability ledger only.
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
  artifacts/causal_followup/v3_source_preserving/design_train_validation.parquet
ssh -i <key> -p <port> ubuntu@<host> \
  'cd /home/ubuntu/interface_formatting_study && scripts/bootstrap_causal_followup_gpu.sh'
ssh -i <key> -p <port> ubuntu@<host> \
  'cd /home/ubuntu/interface_formatting_study && python scripts/cache_causal_models.py'
```

## Canary, forecast, and full-run gates

Run functional then profiling canaries for Mistral, Phi, and Qwen. Mistral is
first because it is the slowest and highest-memory checkpoint.

All starts below run on the rented host under a durable PID, log, and exclusive
per-model lock. Replace `<ssh>` with `ssh -i <key> -p <port> ubuntu@<host>`.

```bash
<ssh> 'cd /home/ubuntu/interface_formatting_study && scripts/control_causal_followup_gpu.sh start mistral functional'
<ssh> 'cd /home/ubuntu/interface_formatting_study && scripts/control_causal_followup_gpu.sh status mistral'
<ssh> 'cd /home/ubuntu/interface_formatting_study && scripts/control_causal_followup_gpu.sh tail mistral functional'
# Repeat start/status/tail for phi, then qwen, then profiling in the same order.
```

The profiling canary is 32 items and at most 2,304 durable rows. For each model record
model-load seconds separately from scoring seconds, forward time/calls, peak
VRAM, padding ratio, completed rows/second, and input-preparation time. Forecast
each full run as `one model load + scoring_seconds / completed_rows * 172506`,
then apply a 1.25 P90 multiplier and add measured setup time. Start full scale
only if combined P90 cost fits below $6.00. Otherwise perform at most one
batch/token-budget optimization canary; accept only if categorical outputs are
identical, selected scores remain within tolerance, and end-to-end scoring
throughput improves at least 10%.

Before full scale, prove stop/resume on one profiling canary: issue `stop`, time
until `status` says stopped, fetch in partial mode, restart the identical
command, and verify the final complete artifact. Record the maximum measured
flush+fetch+termination duration as `shutdown_seconds`.

Canary fetches pass the canary name as the final argument:

```bash
scripts/fetch_causal_followup_artifacts.sh \
  ubuntu@<host> <model-slug> <semantic-run-id> \
  /home/ubuntu/interface_formatting_study gpu_artifacts/causal_followup \
  <key> <port> partial profiling
```

Full order:

```bash
<ssh> 'cd /home/ubuntu/interface_formatting_study && scripts/control_causal_followup_gpu.sh start mistral full'
# status, monitor, fetch and verify Mistral; then repeat for Phi and Qwen.
```

SIGINT/SIGTERM finishes the current shard and records progress. Repeating the
same command resumes only the same semantic identity. Do not alter batch size
or token budget after a full semantic run begins.

## Monitoring and teardown

- Check cost after setup, every canary, every model, and at least every five
  minutes; check every minute after $5.50.
- At $6.00, continue only if less than 10% of canary-weighted GPU work remains
  and its P90 cost plus shutdown fits below $6.75.
- Compute the normal wind-down trigger as
  `$6.75 - hourly_rate * shutdown_seconds / 3600`. At that trigger send SIGTERM,
  wait only the measured reserve, fetch partial, and terminate.
- At $6.75 or if monitoring fails: terminate the Keyhole pod immediately even
  if flush or fetch has not finished. The last already-atomic shard remains the
  recovery boundary.
- Never terminate the unrelated TWC pod.

Fetch command per model:

```bash
scripts/fetch_causal_followup_artifacts.sh \
  ubuntu@<host> <model-slug> <semantic-run-id> \
  /home/ubuntu/interface_formatting_study gpu_artifacts/causal_followup \
  <key> <port> complete
```

Remote stop and emergency kill commands are concrete:

```bash
<ssh> 'cd /home/ubuntu/interface_formatting_study && scripts/control_causal_followup_gpu.sh stop <profile>'
<ssh> 'cd /home/ubuntu/interface_formatting_study && scripts/control_causal_followup_gpu.sh status <profile>'
<ssh> 'cd /home/ubuntu/interface_formatting_study && scripts/control_causal_followup_gpu.sh kill <profile>'
```

During every paid run, collect utilization next to progress and cost checks:

```bash
<ssh> 'nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv,noheader'
<ssh> 'cd /home/ubuntu/interface_formatting_study && find results/causal_runs -name progress.json -print -exec tail -n +1 {} \;'
prime --plain wallet --output json
prime --plain pods list --output json
```

Only after all required model fetches validate:

```bash
prime --plain pods terminate <keyhole-pod-id>
prime --plain pods list --output json
```

Local analysis uses `analysis/analyze_causal_followup.py`; plotting, paper
editing, compilation, and archiving stay on the Mac.
