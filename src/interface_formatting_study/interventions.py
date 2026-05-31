from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, Mapping

import torch

from .anchors import resolve_anchor
from .calibration import calibrate_scores
from .hooks import ResidualEdit
from .scoring import correct_answer_margin, score_labels
from .utils import LABELS


def bias_scores_from_row(row: Mapping[str, object], *, prefix: str | None = None) -> dict[str, float]:
    scores: dict[str, float] = {}
    for label in LABELS:
        column = f"{prefix}_bias_score_{label}" if prefix else f"bias_score_{label}"
        if column not in row:
            raise KeyError(f"Missing required calibration column {column!r}")
        scores[label] = float(row[column])
    return scores


def score_prompt_with_optional_edit(
    model,
    tokenizer,
    prompt: str,
    *,
    correct_label: str,
    bias_scores: Mapping[str, float],
    layer: int | None = None,
    anchor: str | None = None,
    edit_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    batch_size: int = 40,
    device=None,
) -> dict[str, object]:
    if edit_fn is None:
        context_factory = None
    else:
        if layer is None or anchor is None:
            raise ValueError("layer and anchor are required when edit_fn is supplied")
        position = resolve_anchor(tokenizer, prompt, anchor)
        if position is None:
            raise ValueError(f"Anchor {anchor!r} is unavailable for prompt")

        def context_factory():
            return ResidualEdit(model, layer=layer, position=position, edit_fn=edit_fn)

    raw_scores = score_labels(
        model,
        tokenizer,
        prompt,
        batch_size=batch_size,
        device=device,
        forward_context_factory=context_factory,
    )
    cal_scores = calibrate_scores(raw_scores, bias_scores)
    raw_margin = correct_answer_margin(raw_scores, correct_label)
    cal_margin = correct_answer_margin(cal_scores, correct_label)

    out: dict[str, object] = {}
    for label in LABELS:
        out[f"raw_score_{label}"] = raw_scores[label]
        out[f"cal_score_{label}"] = cal_scores[label]
    out.update(
        {
            "raw_pred_label": raw_margin.pred_label,
            "raw_correct": raw_margin.correct,
            "raw_margin": raw_margin.margin,
            "cal_pred_label": cal_margin.pred_label,
            "cal_correct": cal_margin.correct,
            "cal_margin": cal_margin.margin,
            "cal_tie": cal_margin.is_tie,
        }
    )
    return out

