# Paper Story and Camera-Ready Plan

Status: living camera-ready memo, started 19 July 2026. This document records
the scientific story, verified evidence, claim boundaries, reviewer-response
plan, visual design, and CAISc submission requirements. Numerical claims must
remain bound to the cited artifacts; update the cross-model logit-lens section
only after the live Phi and Qwen runs pass complete local verification.

## The paper in one sentence

Changing only an interface wrapper can change a model's answer because the
final choice reflects not only answer-compatible content but also uncertainty,
physical position, displayed labels, and a late mapping from content to the
required output symbol; controlled interventions show that position and label
matter causally, while Mistral's layerwise trajectories show that candidate
answer compatibility remains much more stable across wrappers than A/B/C/D
preferences until late in the network.

This is a heterogeneous-mechanism account, not a claim that every wrapper
failure has one cause or that the logit lens exposes literal thought.

## The complete narrative arc

### Beginning: establish the problem

The same question and the same four answer choices should not change their
meaning when placed in CSV, HTML, INI, TOML, GraphQL, shell, protobuf, or
key-value syntax. Yet the tested instruction-tuned models frequently cross the
correct/incorrect boundary when only that interface wrapper changes.

The behavioral finding is large and replicated:

- Qwen2.5-1.5B: 48.4% calibrated accuracy and 62.0% item-level conflict.
- Phi-3.5-mini: 56.8% calibrated accuracy and 55.1% item-level conflict.
- Mistral-7B-v0.3: 50.3% calibrated accuracy and 66.2% item-level conflict.

These are 3,000 MMLU items per model and eight wrapped prompts per item. The
audit sensitivity analysis preserves the conclusion after identified wrapper
defects are removed. The paper should lead with this concrete reliability
failure, not with interpretability machinery.

### Middle: separate plausible causes

Uncertainty is part of the story but not the whole story. Low-confidence items
are more unstable, yet controlled rotations change answers even when question
content is held fixed:

- Mistral position rotations reduce calibrated accuracy by 6.87 percentage
  points (95% CI -7.84 to -5.95) and flip 43.2% of rows.
- Mistral displayed-label rotations reduce calibrated accuracy by 6.08 points
  (95% CI -6.88 to -5.29) and flip 44.6% of rows.
- Both effects remain negative in every predeclared confidence tercile.

This is the causal core. The model's answer is affected by where candidate
content appears and which displayed letter names it. Therefore at least some
wrapper sensitivity is a binding/output problem rather than merely a loss of
factual or semantic information.

The internal readouts reinforce the distinction without overclaiming it.
Held-out Mistral readouts recover physical position at 80.6% and displayed
label at 97.6%. The content readout reaches 72.2% but fails its predeclared
selectivity criterion, so it does not independently identify a clean,
wrapper-invariant content state. The candidate-local reader is therefore a
diagnostic dead end, not evidence against content-before-label computation: its
target was reconstructed from the emitted label and its token locations were
not clean completed-decision anchors.

### End: locate the strongest defensible internal pattern

The Mistral two-contract logit lens avoids training another semantic classifier.
It compares, at every transformer block, two readouts under two matched prompt
contracts:

- raw A/B/C/D continuation scores, mapped back to exact candidate identities;
- exact candidate-answer continuation-path scores;
- under the original letter-only instruction and a matched answer-text
  counterfactual instruction.

The observation point is the final `Answer:` prefix. Candidate paths use exact
audited answer text and correct multi-token teacher forcing. The first token is
read from the common answer-prefix state; later tokens are conditioned on the
candidate prefix. Thus the full path measures continuation compatibility, not
information residing entirely in one original hidden state.

Across 2,401 permitted train/validation items, nine formats, and 17,626
eligible wrapped/plain pairs per contract, candidate paths are markedly more
wrapper-stable over network depth than mapped A/B/C/D preferences:

| Contract | Raw-letter agreement AUC | Candidate-total agreement AUC | Difference (95% CI) |
|---|---:|---:|---:|
| Letter output | 0.419 | 0.830 | +0.411 (+0.404 to +0.418) |
| Answer-text output | 0.458 | 0.842 | +0.384 (+0.377 to +0.391) |

Both Holm-adjusted p-values are 0.00080. All frozen quality gates pass. The
effect replicates between the discovery-train and validation partitions: the
letter-contract contrast is +0.411 in train and +0.412 in validation.

The result is not a length artifact. Token-mean candidate sensitivities remain
positive (+0.296 for letter output and +0.260 for answer-text output). The
root-only first-token effects are smaller but still positive (+0.069 and
+0.051), which is exactly why the paper must distinguish root lexical evidence
from teacher-forced full-path compatibility.

Timing completes the descriptive picture. Raw letter winners stabilize at
median layer 19. Candidate-total winners stabilize at median layer 28 under
the letter contract and layer 27 under the text contract. Where final
plain/wrapped candidate winners differ, persistent separation usually appears
only at layer 30; raw-letter separation appears earlier (median layer 24 for
letter output and 27 for text output). Candidate evidence is therefore not
perfectly invariant, but its wrapper-specific divergence is concentrated late.

The frozen Mistral conclusion is
`consistent_with_later_output_binding_effects`. A safe paper sentence is:

> Candidate-answer continuation compatibility remains comparatively stable
> across interface wrappers through most of Mistral's computation, whereas
> content-mapped A/B/C/D preferences are substantially more wrapper-sensitive;
> together with causal position and label rotations, this is consistent with
> late binding and output mapping as an important source of wrapper failures.

Do not replace this with “the model first knows the answer and then maps it to
a letter.” The experiment does not establish a discrete internal answer,
causal mediation, literal thought, or one universal circuit.

## What the evidence does and does not resolve

### Supported

- Interface wrappers materially change answer correctness across three model
  families on this MMLU setup.
- Confidence is associated with instability but does not absorb the controlled
  position and label effects.
- Physical position and displayed answer letters causally affect Mistral's
  choices under the declared rotations.
- Candidate-answer compatibility is linearly visible through the final
  normalization/output head and is more wrapper-stable across layers than the
  required A/B/C/D output mapping in Mistral.
- The effect is strongest on wrapper-conflict items (+0.477 versus +0.390 on
  stable items) and when plain/wrapped correctness differs (+0.491 for
  plain-only-correct and +0.467 for wrapped-only-correct), while remaining
  positive in every predeclared major stratum.
- The evidence supports a mixture of answer-formation and binding/output
  failures, with a strong Mistral population-level signature of late binding.

### Not supported

- A claim that the model literally “knows” or “thinks” a specific answer before
  producing the letter.
- A claim that the full candidate-path score is contained at the original
  `Answer:` state. Later candidate tokens are teacher-forced.
- A claim that applying the final output head to intermediate states is a
  privileged or faithful decoder of every internal representation.
- A claim that late binding is the only mechanism, or that every conflict has
  been causally assigned to one bucket.
- A claim that the logit-lens trajectory itself is causal.
- A claim about free-form answer quality beyond exact audited candidate paths.
- A claim about the protected final 599 items; they remain unopened.

## How to restructure the camera-ready paper

The current manuscript tells a behavioral-positive, mechanism-negative story.
The camera-ready should instead tell a progressive explanation story. Preserve
the verified behavioral work, shorten the old coarse-probe narrative, and make
the controlled rotations plus logit lens the scientific resolution.

### Abstract

Use five moves in order:

1. State the reliability problem: formatting-only wrappers change answers.
2. Give the three-model accuracy/conflict headline.
3. State that confidence does not fully explain the effect.
4. Give the causal position/label intervention result.
5. Give the bounded Mistral trajectory result and interpretation.

The abstract must not say merely that “probes did not isolate a state.” It
should end with the positive bounded account: wrapper failures arise from a
mixture of uncertainty and interface-sensitive binding, and Mistral shows a
strong late-output-compatible internal signature.

### 1. Introduction

- Open with one concrete same-question/different-wrapper flip.
- Explain why structured interfaces are operationally common.
- Pose the two rival failure families in plain language:
  answer formation versus answer-to-output binding.
- Preview the evidence ladder rather than listing methods.
- End with three contributions:
  replicated behavior, causal factor separation, and layerwise internal
  evidence with explicit limitations.

### 2. Experimental design and audit

- Models, MMLU population, eight wrappers, exact next-letter scoring, and
  content-free calibration.
- Outcome-blind wrapper audit and sensitivity population.
- Split policy, item-clustered inference, and protected final boundary.
- A compact diagram should show one content identity moving through position,
  displayed letter, and output token.

### 3. Wrappers change answers across models

- Keep the strongest cross-model behavioral figure and table.
- Report calibrated accuracy, conflict, all-correct/all-wrong, and audit
  sensitivity.
- Avoid spending main-text space on every wrapper-specific number.

### 4. Uncertainty is insufficient

- Show the association between confidence and conflicts.
- Then show why this is not a complete explanation: both controlled rotation
  effects remain negative in every confidence tercile.
- This section is the bridge from observation to mechanism.

### 5. Position and displayed labels causally change decisions

- Explain rotations visually before equations.
- Report accuracy deltas, row flips, intervals, and overlap strata.
- Separate marginal causal effects from mutually exclusive explanations; do
  not allocate arbitrary percentages of conflicts to causes.
- Use answer-text/letter disagreement as supporting evidence of an imperfect
  content-to-symbol map.

### 6. Candidate compatibility remains stable until late in Mistral

- Introduce the two-contract by two-readout grid and say explicitly that only
  contract is manipulated; readout is measurement.
- Explain exact contextual tokenization and the root-first-token versus
  teacher-forced-full-path distinction in one small schematic.
- Put the 2x2 trajectory figure first, then the two primary AUC contrasts.
- Report token-mean and first-token sensitivities immediately after the primary
  result, not in a distant appendix.
- Report winner-stability and separation-onset timing as descriptive timing,
  not cognitive commitment.
- End with the frozen bounded interpretation.

### 7. Prior probes and what they ruled out

Compress the old attention/cosine/patching material into one section. Its role
is now to rule out an overly simple global “de-formatting vector,” not to serve
as the paper's ending. Keep the scientifically useful control that
same-answer-letter cross-item donors can outperform same-item successful
donors. Move detailed per-layer plots and donor tables to the appendix if the
page budget is tight.

### 8. Discussion

- Present the heterogeneous account: uncertainty, answer formation, position,
  label, and late output binding can all contribute.
- Explain how the causal rotations and descriptive lens complement rather than
  duplicate each other.
- State why the two prompt contracts do not reveal a privileged latent truth.
- Distinguish exact candidate continuation compatibility from free generation.
- Discuss one dataset/eight-wrapper limits and model-scale limits.
- State the practical implication: systems should test semantic equivalence
  across interface renderings, not assume it.

### 9. Conclusion

End with the answer to the opening puzzle: formatting can alter both evidence
formation and how surviving answer evidence is bound to the required output;
in Mistral, the strongest measured population-level pattern points to the
latter emerging late. The conclusion should not retreat to “we found no
mechanism,” and should not claim a universal circuit.

## Reviewer-response map

The camera-ready has no formal response field, so address the reviews through
the manuscript itself.

| Reviewer concern | Camera-ready resolution |
|---|---|
| Only one small model | Lead with the completed Qwen/Phi/Mistral behavioral replication. Add cross-model logit-lens replication only if the live artifacts verify and the estimand remains comparable. |
| No confidence intervals/statistical testing | Report item-clustered intervals for causal rotations and 5,000-draw item-clustered bootstrap intervals for logit-lens contrasts. Update the reproducibility checklist answer accordingly. |
| Mechanistic probes are coarse/negative | Reframe old probes as ruling out a simple global vector; add the exact answer-prefix logit lens as a more targeted but still descriptive analysis. |
| Mechanism remains unexplained | Present the causal position/label effects and late candidate-versus-letter trajectory as a bounded positive account, not a complete causal decomposition. |
| Wrapper verification unclear | Put the audit procedure, 107 content-changing rows, four ambiguous rows, and removal sensitivity in the main text or a prominent table. |
| Only answer-letter scoring | Add the matched answer-text contract and exact candidate-path analysis, while clearly stating that it remains constrained candidate continuation scoring rather than open-ended generation. |
| Prior work positioning weak | Tighten related work around prompt sensitivity, option-order/label bias, calibration, logit lens limitations, and causal intervention controls. Do not inflate this into a broad literature survey. |

## Figures and visual system

Match the restrained visual language of *Semantic Gravity Wells: Why Negative
Constraints Backfire* (arXiv:2601.08070): white background, compact serif paper
typography, black axes and annotation, sparse grid use, direct captions, and a
small consistent diverging palette.

The dominant sampled colors in the reference PDF are:

- blue `#2166AC` for stable/baseline/content-preserving or success series;
- red `#B2182B` for wrapper-sensitive/divergent/failure series;
- ochre `#D1771E` for a third intervention or calibration series;
- neutral dark gray `#4D4D4D` for reference lines and text;
- light gray `#E6E6E6` for confidence bands or late-layer shading.

Freeze semantic color meanings across all figures. Do not use blue for success
in one panel and candidate scores in another unless those concepts coincide.
For the central logit-lens figure, use blue for candidate compatibility and red
for A/B/C/D binding because the scientific contrast is stability versus
wrapper-sensitive output mapping; keep the same mapping in every model panel.
Use line style, marker shape, and direct labels in addition to color so the
figures remain interpretable in grayscale and for color-vision deficiencies.

Main-text visual plan:

1. One overview schematic: same content, different wrapper, content identity,
   position, displayed label, emitted token.
2. One cross-model behavioral figure: accuracy plus item-level conflicts.
3. One causal-controls figure: position and label rotation effects with 95% CI.
4. One Mistral 2x2 trajectory figure: candidate versus mapped-letter agreement
   across layers under both contracts, with the late region lightly shaded.
5. One compact synthesis figure or table: behavioral effect, controlled cause,
   internal timing, claim boundary.

Avoid rainbow palettes, decorative gradients, oversized titles, and separate
legends that force the reader to decode every panel repeatedly.

## Verified CAISc camera-ready requirements

Source: CAISc acceptance email for paper 168, received 10 July 2026, inspected
in the authenticated mailbox on 19 July 2026.

- Deadline: 18 July 2026, Anywhere on Earth (AoE).
- Upload through the same OpenReview submission used for the original paper.
- Use the same CAISc LaTeX template, changing
  `\usepackage{caisc_2026}` to `\usepackage[final]{caisc_2026}`.
- Restore author names, affiliations, and acknowledgements; the paper is no
  longer anonymous.
- Incorporate reviewer comments and suggestions as much as possible.
- There is no planned presentation or poster session. The only current
  requirement is the camera-ready paper.
- Accepted papers will be visible on OpenReview and the CAISc website.
- Best-paper awards will be announced later; no separate action is currently
  required.

The live OpenReview `Camera Ready Revision` form requires:

- title;
- authors;
- keywords;
- abstract;
- PDF.

TL;DR is present but not marked required. The form also exposes edit history,
readers, and signatures. The current OpenReview abstract and TL;DR still
describe the old single-Qwen, negative-mechanism submission, so update the
metadata to match the final PDF before submitting. Do not submit until the PDF,
metadata, author list, and final checks agree exactly.

## Camera-ready execution order

1. Finish, fetch, analyze, and completely verify the live Phi and Qwen
   logit-lens runs without opening the protected 599 items.
2. Decide the cross-model sentence from verified outcomes:
   - same direction: report replication and a compact cross-model panel;
   - mixed direction: report heterogeneity without averaging it away;
   - invalid/uninformative: retain the bounded Mistral result and omit an
     unsupported general internal claim.
3. Rewrite the paper in the section order above, starting with abstract,
   introduction, and the causal/logit-lens results rather than polishing the
   old negative narrative.
4. Regenerate all figures with the frozen palette and consistent semantic
   color mapping.
5. Update statistical language, denominators, uncertainty, compute details,
   limitations, AI-involvement disclosures, and the reproducibility checklist.
6. Switch the CAISc package to `final`; add author, affiliation, acknowledgement,
   and the non-anonymous AI-system details required by the checklist.
7. Compile and inspect the final PDF page by page for overflow, unreadable
   plots, stale anonymous text, broken references, and mismatched numbers.
8. Make OpenReview title/authors/keywords/TL;DR/abstract match the PDF, attach
   the final PDF, then pause for action-time confirmation before pressing
   Submit.

## Artifact and source provenance

- Governing scientific policy: `SCIENTIFIC_NORTH_STAR.md`.
- Logit-lens contract: `DECISION_BINDING_LOGIT_LENS_RUN_CARD.md`.
- Current manuscript: `paper/main.tex`.
- Verified Mistral artifact:
  `gpu_artifacts/decision_binding_logit_lens/mistral-7b-instruct-v0.3/0ad54c71e58305b4be63/20260719T092800Z/`.
- Mistral primary results:
  `analysis/primary_contrasts.csv` within that artifact.
- Mistral sensitivities and timing:
  `analysis/secondary_contrasts.csv` and
  `analysis/timing_distributions.csv`.
- Mistral frozen gates and interpretation:
  `analysis/quality_gates.json` and `analysis/interpretation_memo.md`.
- OpenReview submission: `https://openreview.net/forum?id=1tSaLHyz13`.
- Visual reference: `https://arxiv.org/pdf/2601.08070`.

## Pending entries

- Phi logit-lens outcome: live run in progress; add only after local complete
  verification.
- Qwen logit-lens outcome: live run in progress; add only after local complete
  verification.
- Final cross-model internal-mechanism sentence and figure layout.
- Exact final paper page count after the new results replace the old narrative.
- Final acknowledgements, affiliation wording, and public repository URL.

