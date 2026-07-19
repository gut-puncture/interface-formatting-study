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

The prescribed CAISc template is authoritative for typography, margins, page
layout, section styling, and bibliography. Do not copy the layout or paper
format of *Semantic Gravity Wells: Why Negative Constraints Backfire*
(arXiv:2601.08070). Use that paper only as the requested chart-color reference.
Within the CAISc format, keep plots restrained: white backgrounds, black axes
and annotation, sparse grids, direct captions, and a small consistent diverging
palette.

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

## Complete paper worklist

This is the working camera-ready checklist. A box is complete only when the
compiled PDF and its bound source/artifact support the statement.

### Evidence and result finalization

- [ ] Finish, fetch, and completely verify the Phi logit-lens artifact.
- [ ] Finish, fetch, and completely verify the Qwen logit-lens artifact.
- [ ] Record each model's final quality-gate result, population, exclusions,
      primary contrasts, sensitivities, timing, and parity receipt.
- [ ] Decide the cross-model internal conclusion from the frozen rules; do not
      average away opposite or uninformative outcomes.
- [ ] Preserve the unopened final-599 boundary in methods and limitations.
- [ ] Reconcile every number used in prose, tables, figures, abstract,
      OpenReview metadata, and checklists against one canonical artifact.
- [ ] Record exact compute environment, GPU, runtime, and cost for the new runs.

### Scientific framing and claims

- [ ] Rewrite the one-sentence thesis around heterogeneous answer formation and
      answer-to-output binding.
- [ ] Replace the old “modest negative” ending with the bounded causal-plus-
      trajectory account.
- [ ] State explicitly that the logit lens is descriptive, not causal or
      literal thought access.
- [ ] State explicitly that full candidate paths are teacher-forced continuation
      compatibility, not information contained wholly at the answer prefix.
- [ ] Keep first-token, total-path, and token-mean estimands separate.
- [ ] Avoid claiming that late binding is universal, unique, or sufficient to
      explain every wrapper conflict.
- [ ] Avoid claiming free-form generation robustness from constrained
      candidate scoring.
- [ ] Explain that the text-output prompt is a counterfactual contract, not a
      privileged view of the letter-output model's hidden answer.

### Abstract, title, and introduction

- [ ] Reassess whether the existing title remains the narrowest accurate title
      after the new evidence; change it only if the final story requires it.
- [ ] Rewrite the abstract with three-model behavior, confidence limitation,
      causal rotations, logit-lens result, and bounded conclusion.
- [ ] Update the abstract's model, prompt, item, and uncertainty counts.
- [ ] Open the introduction with one concrete wrapper-induced answer flip.
- [ ] Define answer-formation failure and binding/output failure in plain
      language.
- [ ] End the introduction with three precise contributions and claim limits.

### Experimental setup and audit

- [ ] Retain exact model revisions, MMLU population, wrappers, scoring, and
      content-free calibration.
- [ ] Explain the outcome-blind row audit in the main text, including 107
      content-changing rows, four ambiguous rows, and removal sensitivity.
- [ ] Explain item-level conflicts and why wrapper rows are clustered by item.
- [ ] Add the controlled position-rotation and displayed-label-rotation design.
- [ ] Add the two-contract/two-readout logit-lens design without calling it a
      causal 2x2 factorial.
- [ ] Document exact candidate identities, contextual tokenization,
      multi-token teacher forcing, prefix collisions, duplicates, and explicit
      eligibility.
- [ ] Document the final-layer/native parity gate and numerical ambiguity policy.
- [ ] Explain train/validation pooling, fixed split replication, bootstrap,
      Holm correction, and predeclared strata.

### Results sections

- [ ] Consolidate the cross-model behavioral results into one lead figure and
      one compact table.
- [ ] Add the confidence analysis and show that rotation effects persist in all
      confidence terciles.
- [ ] Add position and displayed-label causal effects with item-clustered 95%
      intervals, row-flip rates, and overlap strata.
- [ ] Add answer-text/letter disagreement as bounded supporting evidence.
- [ ] Add the Mistral primary AUC contrasts with item and pair denominators.
- [ ] Add token-mean and first-token sensitivity results beside the primary
      result.
- [ ] Add winner-stability and separation-onset results with careful timing
      language.
- [ ] Add train-versus-validation replication and predeclared stratum results.
- [ ] Add Phi/Qwen layerwise replication only if their complete artifacts and
      frozen gates support it.
- [ ] Compress old attention, cosine, and patching results into a “what simple
      accounts were ruled out” section; move excess layer detail to appendix.

### Statistical reporting

- [ ] Add confidence intervals to every comparative mechanistic claim that
      currently has only a point estimate.
- [ ] Give deterministic rates denominators even where no sampling claim is
      made.
- [ ] Use item-clustered uncertainty wherever wrappers share an underlying
      question.
- [ ] Describe the 5,000-draw bootstrap and seed.
- [ ] Report Holm adjustment only for the two declared co-primary contrasts.
- [ ] Do not turn each layer into a separate significance claim; use complete
      trajectories and simultaneous bands.
- [ ] Update the reproducibility checklist's statistical-significance answer
      from `No` if and only if the final manuscript actually reports the
      declared intervals correctly.

### Related work and bibliography

- [ ] Add a concise related-work subsection covering prompt sensitivity and
      format dependence, option-order and answer-label bias, content-free
      calibration, logit-lens limitations, and causal intervention controls.
- [ ] Connect each related work citation to the exact claim it supports; avoid
      citation lists detached from prose.
- [ ] Verify every existing and new reference against an authoritative
      publisher, DOI, OpenReview, arXiv, or official model record.
- [ ] Correct title, authors, venue, year, identifier, and URL where the current
      BibTeX differs from the authoritative record.
- [ ] Confirm every citation key used in `main.tex` exists exactly once and
      every bibliography entry is cited or intentionally retained.
- [ ] Remove any invented, unverifiable, duplicate, or claim-mismatched entry.
- [ ] Recompile and inspect the rendered bibliography under the prescribed
      CAISc style.

### Figures and tables

- [ ] Regenerate all main figures with the fixed `#2166AC` blue, `#B2182B`
      red, `#D1771E` ochre, and neutral gray palette.
- [ ] Keep semantic color assignments identical across every model and panel.
- [ ] Add non-color encodings: line style, markers, labels, or hatching.
- [ ] Add an overview schematic of content identity, position, displayed label,
      and output token.
- [ ] Add the causal rotation figure with intervals.
- [ ] Add the logit-lens 2x2 trajectory figure with a lightly shaded late layer
      region and no implication of a tuned transition point.
- [ ] Ensure every axis, unit, denominator, layer convention, uncertainty band,
      and exclusion is explained in its caption.
- [ ] Verify legibility at the prescribed 5.5-inch single-column size and in grayscale.
- [ ] Remove redundant figures rather than shrinking them below legibility.

### Discussion, limitations, and conclusion

- [ ] Explain the heterogeneous mechanism account and unresolved population.
- [ ] Separate causal evidence from descriptive internal evidence.
- [ ] Retain one-dataset/eight-wrapper and imperfect-wrapper limitations.
- [ ] Retain model-family/scale limits even if all three logit-lens runs agree.
- [ ] Add constrained-candidate versus open-generation limitations.
- [ ] Add tokenizer/surface-form and intermediate-unembedding limitations.
- [ ] State positive reliability uses and possible adversarial misuse.
- [ ] End with the bounded answer to the opening question, not a generic null.

### Appendices and reproducibility

- [ ] Move detailed wrapper audit, old probe layers, control donor tables,
      continuation eligibility census, parity details, and additional strata to
      appendices when they do not fit the main narrative.
- [ ] Update the reproducibility bundle with the exact released code commit,
      compact verified artifacts, commands, manifests, and checksums.
- [ ] Update model/tokenizer revisions, package versions, runtime batching, GPU
      provenance, elapsed time, and cost.
- [ ] Ensure no final-599 outcomes, private credentials, GPU host details, or
      irrelevant raw caches enter the public bundle.
- [ ] Update the AI Involvement Checklist to name the actual systems in the
      non-anonymous camera-ready as required by its own instructions.
- [ ] Update every checklist explanation whose scientific story or statistical
      evidence changed.
- [ ] Rebuild the PDF, page previews, source archive, reproducibility bundle,
      bundle README, and SHA-256 manifest together from one frozen source.
- [ ] Delete or replace stale generated submission artifacts only through the
      established packaging command; do not leave the May single-Qwen bundle
      beside a July three-model camera-ready under an ambiguous name.
- [ ] Reassess the checklist's current “open access to data and code: Yes” after
      the new causal and logit-lens files enter the paper; it is supportable only
      if the refreshed public bundle actually contains the required code,
      compact evidence, and instructions.

### Camera-ready formatting and metadata

- [ ] Keep the prescribed `caisc_2026` template and switch only to its `final`
      option.
- [ ] Restore author name, affiliation, contact details if required by the
      template, and acknowledgements.
- [ ] Check the venue's page limit and any template diagnostics before final
      upload; do not infer layout rules from the visual-reference paper.
- [ ] Remove all anonymous-submission wording and stale “one model” language.
- [ ] Compile from a clean state and inspect every page for overflow, floats,
      broken links, bad citations, missing glyphs, and illegible figures.
- [ ] Confirm PDF opens, fonts are embedded, links work, and no tracked changes,
      comments, hidden annotations, or local paths remain.
- [ ] Make OpenReview title, author list, keywords, TL;DR, and abstract match the
      final PDF.
- [ ] Upload only the final verified PDF and pause for explicit action-time
      confirmation before pressing OpenReview Submit.

## Reference audit and related-work plan

The current bibliography was audited entry by entry on 19 July 2026 against
authoritative OpenReview, arXiv, Hugging Face, NeurIPS, PMLR, ACL Anthology, and
publisher records. All seven cited works are real. There are no duplicate keys,
missing keys, unused entries, or existing citation claims contradicted by their
sources.

| Existing key | Audit result | Action |
|---|---|---|
| `hendrycks2021mmlu` | Verified: MMLU, ICLR 2021 | Keep. |
| `qwen2025qwen25` | Real report; prior entry mismatched the current 2025 revision | Corrected to Qwen Team, 2025, arXiv:2412.15115. |
| `microsoft2024phi35` | Verified official Microsoft model card | Keep; optionally pin the model revision in prose or manifest, not necessarily the citation. |
| `mistralai2024mistral7bv03` | Verified official Mistral AI model card | Keep; optionally normalize the title wording. |
| `zhang2024activationpatching` | Verified ICLR 2024 paper | Keep; directly supports patching-method sensitivity. |
| `meng2022locating` | Verified NeurIPS 2022 paper | Keep; optional page/publisher metadata only. |
| `wang2022interpretability` | Verified 2022 arXiv preprint | Keep or upgrade to the ICLR 2023 record; either is real. |

The reviewer was correct that this real bibliography is too narrow. It has no
focused Related Work section and does not cite the closest precedents for
prompt formatting or content-free calibration. The minimum verified additions
are:

1. Sclar, Choi, Tsvetkov, and Suhr, “Quantifying Language Models' Sensitivity
   to Spurious Features in Prompt Design or: How I Learned to Start Worrying
   About Prompt Formatting,” ICLR 2024, OpenReview `RIu5lyNXjT`, arXiv:2310.11324.
   Use for meaning-preserving prompt-format sensitivity and state our distinct
   contribution rather than implying the phenomenon is wholly new.
2. Mizrahi, Kaplan, Malkin, Dror, Shahaf, and Stanovsky, “State of What Art? A
   Call for Multi-Prompt LLM Evaluation,” TACL 12 (2024), 933–949,
   DOI `10.1162/tacl_a_00681`, arXiv:2401.00595. Use for multi-prompt benchmark
   brittleness and the need to report variation across prompt realizations.
3. Zhao, Wallace, Feng, Klein, and Singh, “Calibrate Before Use: Improving
   Few-shot Performance of Language Models,” ICML 2021, 12697–12706. Use for
   content-free contextual calibration; describe this paper's wrapper-specific
   log-score subtraction as an adaptation, not an identical procedure.

The new causal and logit-lens story additionally needs a few targeted, verified
citations rather than a broad survey:

4. Pezeshkpour and Hruschka, “Large Language Models Sensitivity to The Order of
   Options in Multiple-Choice Questions,” Findings of NAACL 2024, 2006–2017,
   DOI `10.18653/v1/2024.findings-naacl.130`. Use for option-position effects.
5. Zhou, Wang, Xu, Chen, and Duan, “Revisiting the Self-Consistency Challenges
   in Multi-Choice Question Formats for Large Language Model Evaluation,”
   LREC-COLING 2024, 14103–14110. Use for knowledge-equivalent position and
   label variants in multiple-choice evaluation.
6. Belrose et al., “Eliciting Latent Predictions from Transformers with the
   Tuned Lens,” arXiv:2303.08112. Use specifically to explain that a raw logit
   lens can be brittle and intermediate states were not necessarily trained for
   direct final-head decoding.

Two further real works are optional if the corresponding prose remains:

- Wei et al., “Unveiling Selection Biases: Exploring Order and Token
  Sensitivity in Large Language Models,” Findings of ACL 2024,
  DOI `10.18653/v1/2024.findings-acl.333`.
- Holtzman et al., “Surface Form Competition: Why the Highest Probability
  Answer Isn't Always Right,” EMNLP 2021,
  DOI `10.18653/v1/2021.emnlp-main.564`.

Add only references used by a specific sentence. Before final compile, copy
BibTeX metadata from these authoritative records, then repeat the key,
duplication, citation-use, and rendered-bibliography audit.

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
