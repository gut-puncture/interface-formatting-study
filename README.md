# When Wrappers Hide What a Language Model Knows

Code, data and results for the paper *When Wrappers Hide What a Language Model Knows:
The Question Gets Through, the Answer Does Not* (Shailesh Rana). The paper is in
[`paper/main.pdf`](paper/main.pdf).

We ask four open models (Qwen3-4B-Instruct-2507, Qwen3-8B, OLMo-3-7B-Instruct, Gemma-4-12B-it)
1,606 short questions about facts, arithmetic and translation, keep for each model only the
questions it answers correctly when asked plainly, and place each question inside five kinds of
wrapper (tool result, web page, server log, email thread, plain prose) at three lengths.

- Wrappers hide known answers. Server logs hide the most in two of the four models, mostly by
  turning the reply into a description of the log.
- The other wrappers mostly turn weakly held answers into wrong answers, or into claims that the
  text holds no answer.
- Activation patching shows the model still reads the question inside the wrapper. Swapping in
  another question's states makes the model answer that question. The answer is lost later, at
  the last position, after the question's content reaches it (about 0.6–0.7 of the depth).

## Layout

| Path | What it holds |
|---|---|
| `keyhole/frozen/` | Frozen generators for items and wrappers (hashes in `SHA256SUMS`) |
| `data/` | Items, wrapper texts, filler assignments, token checks and the vendored source datasets |
| `keyhole/` | Prompt building, the prefix-cache engine with patch hooks, the behaviour and patching passes, analysis, figures and paper numbers |
| `scripts/` | GPU pod setup, run, watchdog and helper scripts |
| `tests/` | Unit tests on tiny random models (`pytest tests`) |
| `results/results.json` | Every number in the paper |
| `results/run/` | Per-model behaviour and patching outputs (parquet), manifest and run log |
| `paper/` | LaTeX sources, generated numbers and tables, figures and the PDF |

## Reproduce

1. On one 80 GB GPU: `bash scripts/pod_setup.sh`, then `bash scripts/pod_run.sh`
   (writes `runs/main/`; about 6.5 GPU hours on an A100).
2. Analysis on CPU: `python -m keyhole.analyze --run runs/main --out results`.
3. Figures and numbers: `python -m keyhole.figures --results results --out paper/figures` and
   `python -m keyhole.numbers --results results --run runs/main --out paper/generated`.
4. Paper: `cd paper && latexmk -pdf main.tex`.

The residual-state arrays used for one secondary check (`states_*.npy`, 6 GB) are not in git.

## Data sources

Country capitals from Todd et al. (function vectors), relations from Hernandez et al. (linear
relational embeddings), translation word lists from Wendler et al.; licences are in `data/raw/`.

## Earlier version

The first version of this study, the CAISc 2026 paper *When Interface Formatting Changes Model
Decisions*, is preserved under the git tag `v1-caisc2026`.
