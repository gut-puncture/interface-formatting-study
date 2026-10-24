# Scientific North Star

## Goal

Explain why presenting the same multiple-choice question and answer texts through different interface wrappers can change a model's answer.

The paper should form a complete argument:

1. Establish the phenomenon across models.
2. Separate the plausible causes with controlled evidence.
3. Quantify how much each cause explains, including overlaps.
4. Connect the behavioral findings to the strongest defensible internal evidence.
5. Bound the remaining unexplained cases instead of forcing one universal mechanism.

The intended endpoint is not merely another positive or negative experiment. It is a coherent account of the observed behavior with a clear beginning, middle, and end.

## Working Explanation

Wrapper sensitivity is probably heterogeneous. A model's final answer can reflect several competing influences:

- semantic evidence about the answer;
- uncertainty and question difficulty;
- physical-position preferences;
- displayed-letter preferences, including a strong preference for A in some settings;
- the process that maps answer content to an A/B/C/D output;
- wrapper-induced changes in attention or computation.

These influences imply two broad failure families:

1. **Answer-formation failure:** the wrapper changes or degrades the model's substantive answer preference.
2. **Binding/output failure:** useful answer information survives, but position, letter, or interface priors alter how it is expressed.

Both may occur. The project must measure mixtures and overlaps rather than protect either explanation as the conclusion.

## Established Evidence

- Wrapped and plain presentations can produce different answers across Qwen, Phi, and Mistral.
- Conflicts are associated with uncertainty, but uncertainty does not explain every conflict.
- Controlled position and displayed-letter rotations materially affect choices.
- Answer-text and answer-letter readouts sometimes diverge.
- Position and displayed letter are strongly decodable from internal states.
- The original global content reader was not independently identified.
- The candidate-local reader found moderate candidate-related signal, but a displayed-label shortcut predicted the final choice better.

The candidate-local result is narrow. It shows that the final emitted choice was not cleanly decodable as candidate content from those early option-token states. It does not disprove a content-before-label process.

## Why Another Semantic Classifier Is Not The Current Path

There is no clean observed label for "the answer content the model was internally thinking" in the original letter-output prompt.

- Using the emitted A/B/C/D choice as the semantic target is circular. It defines the model's internal content as whatever content the emitted letter denotes, so it cannot detect "preferred Berlin but emitted A/Madrid."
- Using the objectively correct answer measures whether correct information is decodable; it does not identify the model's actual preference when the model is wrong.
- Asking for answer text supplies an independent behavioral readout, but it changes the terminal instruction. Hidden states after that changed instruction are not directly comparable with late states from the letter-output prompt.

The previous candidate-token locations are also poorly aligned with the intended claim. In a causal language model, the state after an early option has not seen later options. It is also strongly involved in predicting list continuation, such as `B.` after the first candidate or `C.` after the second. Those states may contain useful information, but they are not clean measurements of a completed answer decision.

Therefore, do not train another semantic classifier merely by increasing capacity, data, or token locations. Resume classifier work only if a non-circular target and comparable measurement context are established first.

## Current Internal-Evidence Policy

Use the logit lens as a direct, lightweight diagnostic rather than inventing semantic labels. Under the exact original letter-output prompt:

- inspect A/B/C/D logits at the final answer-prefix token across layers;
- inspect candidate-answer tokens only where tokenization permits a predeclared, interpretable comparison;
- compare matched plain and wrapped trajectories;
- treat intermediate-layer unembedding as an imperfect descriptive window, not literal access to thought;
- never infer that semantic information is absent merely because raw candidate-token logits are weak.

The frozen extension applies this same diagnostic to the pinned Mistral, Phi,
and Qwen profiles. Model identity and revision determine the complete layer
range and final parity layer; scoring, eligibility, interpretation, and claim
boundaries do not change by profile.

Any causal claim still requires an intervention. A logit-lens trajectory can locate a candidate transition; it cannot by itself prove that the decoded signal controls the answer.

## Immediate Scientific Sequence

1. Analyze the existing train-and-validation evidence deeply before another paid run.
2. Build a quantitative decomposition of conflicts using confidence, position rotations, letter rotations, answer-text versus letter outcomes, wrapper identity, and their overlaps.
3. Report both marginal explanatory coverage and overlap. Do not create arbitrary mutually exclusive causal percentages.
4. Identify the largest scientifically important unexplained population.
5. Freeze one small pinned-profile internal diagnostic targeted at that population, using the logit lens rather than another semantic classifier.
6. Consider one targeted causal intervention only if the diagnostic reveals a clear, repeatable internal transition.

The untouched final 599 questions remain closed until the method and interpretation are frozen. The failed candidate-local gate does not authorize opening them.

## Claim Boundaries

The project may support claims about behavioral sensitivity, controlled position or letter effects, descriptive layerwise decodability, and causal effects of interventions actually performed.

It must not equate a probe or logit lens with literal thought, treat correlation as causation, claim that one mechanism explains every conflict, or interpret a negative readout as proof that the corresponding information does not exist internally.
