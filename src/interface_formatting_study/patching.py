from __future__ import annotations

import pandas as pd
import torch
from tqdm import tqdm

from .anchors import anchor_components, anchor_is_leaky, resolve_anchor, resolve_anchor_positions
from .calibration import calibrate_scores
from .hooks import ResidualCapture, ResidualLayerRowMultiEdit, ResidualMultiEdit, find_transformer_blocks, replace_with
from .interventions import bias_scores_from_row, score_prompt_with_optional_edit
from .scoring import (
    correct_answer_margin,
    label_variants,
    score_labels,
    single_token_label_ids,
    tokenize_text,
)
from .utils import LABELS


def hidden_state_at(
    model,
    tokenizer,
    prompt: str,
    *,
    layer: int,
    anchor: str,
    question: object = None,
    choices: object = None,
    device=None,
) -> torch.Tensor:
    position = resolve_anchor(tokenizer, prompt, anchor, question=question, choices=choices)
    if position is None:
        raise ValueError(f"Anchor {anchor!r} is unavailable for prompt")
    prompt_ids = tokenize_text(tokenizer, prompt)
    device_obj = device or next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device_obj)
    attention_mask = torch.ones_like(input_ids)
    with ResidualCapture(model, layer=layer, position=position) as capture:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    if capture.value is None:
        raise RuntimeError("Residual capture hook did not run")
    return capture.value[0]


def hidden_states_at_layers(
    model,
    tokenizer,
    prompt: str,
    *,
    layers: list[int],
    anchor: str,
    device=None,
) -> dict[int, torch.Tensor]:
    position = resolve_anchor(tokenizer, prompt, anchor)
    if position is None:
        raise ValueError(f"Anchor {anchor!r} is unavailable for prompt")
    prompt_ids = tokenize_text(tokenizer, prompt)
    device_obj = device or next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device_obj)
    attention_mask = torch.ones_like(input_ids)
    blocks = find_transformer_blocks(model)
    captures: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captures[layer] = hidden[:, position, :].detach().float().cpu()[0]
            return output

        return hook

    try:
        for layer in layers:
            handles.append(blocks[layer].register_forward_hook(make_hook(layer)))
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for handle in handles:
            handle.remove()
    missing = set(layers) - set(captures)
    if missing:
        raise RuntimeError(f"Residual capture hooks did not run for layers: {sorted(missing)}")
    return captures


def hidden_states_at_layers_and_positions(
    model,
    tokenizer,
    prompt: str,
    *,
    layers: list[int],
    positions: list[int],
    device=None,
) -> dict[int, dict[int, torch.Tensor]]:
    if not positions:
        raise ValueError("positions must not be empty")
    prompt_ids = tokenize_text(tokenizer, prompt)
    device_obj = device or next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device_obj)
    attention_mask = torch.ones_like(input_ids)
    blocks = find_transformer_blocks(model)
    wanted_positions = sorted({int(position) for position in positions})
    captures: dict[int, dict[int, torch.Tensor]] = {}
    handles = []

    def make_hook(layer: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captures[layer] = {
                position: hidden[:, position, :].detach().float().cpu()[0]
                for position in wanted_positions
            }
            return output

        return hook

    try:
        for layer in layers:
            handles.append(blocks[layer].register_forward_hook(make_hook(layer)))
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for handle in handles:
            handle.remove()
    missing = set(layers) - set(captures)
    if missing:
        raise RuntimeError(f"Residual capture hooks did not run for layers: {sorted(missing)}")
    return captures


def score_with_replacement_patch(
    model,
    tokenizer,
    prompt: str,
    *,
    correct_label: str,
    layer: int,
    anchor: str,
    replacement_vector: torch.Tensor,
    batch_size: int = 40,
    device=None,
) -> dict[str, object]:
    position = resolve_anchor(tokenizer, prompt, anchor)
    if position is None:
        raise ValueError(f"Anchor {anchor!r} is unavailable for prompt")

    bias_scores = {label: 0.0 for label in ("A", "B", "C", "D")}
    scored = score_prompt_with_optional_edit(
        model,
        tokenizer,
        prompt,
        correct_label=correct_label,
        bias_scores=bias_scores,
        layer=layer,
        anchor=anchor,
        edit_fn=replace_with(replacement_vector),
        batch_size=batch_size,
        device=device,
    )
    out: dict[str, object] = {}
    for key, value in scored.items():
        out[f"patched_{key}"] = value
    return out


def score_prompt_with_position_replacements(
    model,
    tokenizer,
    prompt: str,
    *,
    correct_label: str,
    bias_scores: dict[str, float],
    layer: int,
    replacements: dict[int, torch.Tensor],
    batch_size: int = 40,
    device=None,
) -> dict[str, object]:
    if not replacements:
        raise ValueError("replacements must not be empty")
    edits = {position: replace_with(vector) for position, vector in replacements.items()}

    def context_factory():
        return ResidualMultiEdit(model, layer=layer, edits=edits)

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


def _pad_id(tokenizer) -> int:
    value = getattr(tokenizer, "pad_token_id", None)
    if value is not None:
        return int(value)
    value = getattr(tokenizer, "eos_token_id", None)
    if value is not None:
        return int(value)
    return 0


def _scored_label_result(
    raw_scores: dict[str, float],
    bias_scores: dict[str, float],
    correct_label: str,
) -> dict[str, object]:
    cal_scores = calibrate_scores(raw_scores, bias_scores)
    raw_margin = correct_answer_margin(raw_scores, correct_label)
    cal_margin = correct_answer_margin(cal_scores, correct_label)
    scored: dict[str, object] = {}
    for label in LABELS:
        scored[f"raw_score_{label}"] = raw_scores[label]
        scored[f"cal_score_{label}"] = cal_scores[label]
    scored.update(
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
    return scored


def score_layers_with_position_replacements(
    model,
    tokenizer,
    prompt: str,
    *,
    correct_label: str,
    bias_scores: dict[str, float],
    replacements_by_layer: dict[int, dict[int, torch.Tensor]],
    device=None,
) -> dict[int, dict[str, object]]:
    """Score independent layer patches for one prompt in a single batched forward pass."""

    if not replacements_by_layer:
        raise ValueError("replacements_by_layer must not be empty")
    if any(not replacements for replacements in replacements_by_layer.values()):
        raise ValueError("each layer must have at least one replacement")

    prompt_ids = tokenize_text(tokenizer, prompt)
    if not prompt_ids:
        raise ValueError("Prompt must contain at least one token to score completions")
    device_obj = device or next(model.parameters()).device
    label_ids = single_token_label_ids(tokenizer)
    if label_ids is not None:
        layers = sorted(int(layer) for layer in replacements_by_layer)
        input_ids = torch.tensor(
            [prompt_ids for _ in layers], dtype=torch.long, device=device_obj
        )
        attention_mask = torch.ones_like(input_ids)
        edits_by_layer = {
            layer: {row: replacements_by_layer[layer]}
            for row, layer in enumerate(layers)
        }
        with ResidualLayerRowMultiEdit(model, edits_by_layer=edits_by_layer):
            with torch.inference_mode():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = torch.log_softmax(logits[:, len(prompt_ids) - 1], dim=-1)
        return {
            layer: _scored_label_result(
                {
                    label: float(log_probs[row, label_ids[label]].detach().cpu())
                    for label in LABELS
                },
                bias_scores,
                correct_label,
            )
            for row, layer in enumerate(layers)
        }

    completions: list[str] = []
    label_for_completion: list[str] = []
    for label in LABELS:
        variants = label_variants(label)
        completions.extend(variants)
        label_for_completion.extend([label] * len(variants))
    completion_ids = [tokenize_text(tokenizer, completion) for completion in completions]
    if any(len(ids) == 0 for ids in completion_ids):
        raise ValueError("Completion variants must contain at least one token")

    pad_token_id = _pad_id(tokenizer)
    entries = [
        (int(layer), completion_index)
        for layer in sorted(replacements_by_layer)
        for completion_index in range(len(completion_ids))
    ]
    sequences = [prompt_ids + completion_ids[completion_index] for _, completion_index in entries]
    max_len = max(len(seq) for seq in sequences)
    input_ids = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long, device=device_obj)
    attention_mask = torch.zeros_like(input_ids)
    for row_index, seq in enumerate(sequences):
        seq_tensor = torch.tensor(seq, dtype=torch.long, device=device_obj)
        input_ids[row_index, : len(seq)] = seq_tensor
        attention_mask[row_index, : len(seq)] = 1

    edits_by_layer: dict[int, dict[int, dict[int, torch.Tensor]]] = {}
    for row_index, (layer, _) in enumerate(entries):
        edits_by_layer.setdefault(layer, {})[row_index] = replacements_by_layer[layer]

    with ResidualLayerRowMultiEdit(model, edits_by_layer=edits_by_layer):
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            log_probs = torch.log_softmax(outputs.logits, dim=-1)

    scores_by_layer: dict[int, dict[str, list[float]]] = {
        int(layer): {label: [] for label in LABELS}
        for layer in replacements_by_layer
    }
    for row_index, (layer, completion_index) in enumerate(entries):
        ids = completion_ids[completion_index]
        total = torch.zeros((), dtype=log_probs.dtype, device=log_probs.device)
        for offset, token_id in enumerate(ids):
            absolute_pos = len(prompt_ids) + offset
            total = total + log_probs[row_index, absolute_pos - 1, token_id]
        scores_by_layer[int(layer)][label_for_completion[completion_index]].append(float(total.detach().cpu()))

    out: dict[int, dict[str, object]] = {}
    for layer, label_scores in scores_by_layer.items():
        raw_scores: dict[str, float] = {}
        for label in LABELS:
            values = torch.tensor(label_scores[label], dtype=torch.float64)
            raw_scores[label] = float(torch.logsumexp(values, dim=0) - torch.log(torch.tensor(float(len(values)))))
        out[layer] = _scored_label_result(raw_scores, bias_scores, correct_label)
    return out


def recovery(patched_margin: float, corrupt_margin: float, clean_margin: float) -> float | None:
    denom = clean_margin - corrupt_margin
    if denom <= 1e-6:
        return None
    return float((patched_margin - corrupt_margin) / denom)


def select_best_location(results: pd.DataFrame, *, allow_leaky_anchors: bool = False) -> dict[str, object]:
    if results.empty:
        raise ValueError("Cannot select a layer/anchor from empty patching results")
    candidates = results.copy()
    if "skipped" in candidates.columns:
        candidates = candidates[~candidates["skipped"].astype(bool)].copy()
    if not allow_leaky_anchors:
        candidates = candidates[~candidates["anchor"].astype(str).map(anchor_is_leaky)].copy()
    if candidates.empty:
        raise ValueError("Cannot select a non-leaky layer/anchor from empty patching results")
    if "patched_correct" not in candidates.columns:
        candidates["patched_correct"] = float("nan")
    if "n_positions" not in candidates.columns:
        candidates["n_positions"] = 1
    grouped = (
        candidates.groupby(["layer", "anchor"], as_index=False)
        .agg(
            mean_margin_improvement=("margin_improvement", "mean"),
            mean_recovery=("recovery", "mean"),
            patched_accuracy=("patched_correct", "mean"),
            mean_positions=("n_positions", "mean"),
            n_items=("item_id", "nunique"),
        )
        .sort_values(["mean_margin_improvement", "mean_recovery"], ascending=[False, False])
    )
    best = grouped.iloc[0]
    best_improvement = float(best["mean_margin_improvement"])
    tolerance = abs(best_improvement) * 0.05
    near = grouped[grouped["mean_margin_improvement"] >= best_improvement - tolerance].copy()
    near["anchor_rank"] = near["anchor"].map(
        {
            "question_end": 0,
            "question_end+options_end": 1,
            "options_end": 2,
            "all_option_ends": 3,
            "option_A_end": 4,
            "option_B_end": 4,
            "option_C_end": 4,
            "option_D_end": 4,
            "content_end": 5,
            "answer_anchor": 99,
        }
    ).fillna(50)
    near = near.sort_values(["layer", "anchor_rank", "mean_recovery"], ascending=[True, True, False])
    chosen = near.iloc[0]
    return {
        "selected_layer": int(chosen["layer"]),
        "selected_anchor": str(chosen["anchor"]),
        "selected_anchor_components": list(anchor_components(str(chosen["anchor"]))),
        "selection_split": "validation",
        "selection_metric": "mean_calibrated_margin_improvement",
        "mean_margin_improvement": float(chosen["mean_margin_improvement"]),
        "mean_recovery": float(chosen["mean_recovery"]),
        "patched_accuracy": float(chosen["patched_accuracy"]),
        "n_items": int(chosen["n_items"]),
        "leaky_anchors_allowed": bool(allow_leaky_anchors),
        "candidate_anchor_policy": "all_anchors" if allow_leaky_anchors else "non_leaky_semantic_anchors_only",
    }


def available_layers(model) -> list[int]:
    return list(range(len(find_transformer_blocks(model))))


def _cap_pairs(pairs: pd.DataFrame, *, cap: int | None, seed: int) -> pd.DataFrame:
    if cap is None or len(pairs) <= cap:
        return pairs
    return pairs.sample(n=cap, random_state=seed).sort_values("item_id").reset_index(drop=True)


def _positions_to_string(positions: list[int]) -> str:
    return ",".join(str(int(position)) for position in positions)


def _skip_rows(
    *,
    row_dict: dict[str, object],
    layers: list[int],
    anchor: str,
    reason: str,
) -> list[dict[str, object]]:
    out = []
    for layer in layers:
        out.append(
            {
                "item_id": row_dict["item_id"],
                "subject": row_dict.get("subject"),
                "clean_wrapper": row_dict["clean_wrapper"],
                "corrupt_wrapper": row_dict["corrupt_wrapper"],
                "split": row_dict["split"],
                "layer": int(layer),
                "anchor": anchor,
                "anchor_components": ",".join(anchor_components(anchor)),
                "patch_kind": "grouped" if len(anchor_components(anchor)) > 1 else "single",
                "n_positions": 0,
                "clean_positions": "",
                "corrupt_positions": "",
                "clean_margin": float(row_dict["clean_margin"]),
                "corrupt_margin": float(row_dict["corrupt_margin"]),
                "patched_margin": float("nan"),
                "patched_correct": False,
                "patched_pred_label": None,
                "margin_improvement": float("nan"),
                "recovery": float("nan"),
                "skipped": True,
                "skip_reason": reason,
            }
        )
    return out


def run_validation_patching_sweep(
    model,
    tokenizer,
    conflict_pairs: pd.DataFrame,
    *,
    layers: list[int] | None = None,
    anchors: tuple[str, ...] = ("question_end", "options_end"),
    split: str = "validation",
    cap: int | None = 300,
    seed: int = 1729,
    batch_size: int = 40,
    device=None,
) -> pd.DataFrame:
    pairs = conflict_pairs[conflict_pairs["split"] == split].copy()
    pairs = _cap_pairs(pairs, cap=cap, seed=seed)
    if layers is None:
        layers = available_layers(model)

    rows: list[dict[str, object]] = []
    for _, pair in tqdm(pairs.iterrows(), total=len(pairs), desc="patching pairs"):
        row_dict = pair.to_dict()
        correct_label = str(row_dict["correct_label"])
        corrupt_bias = bias_scores_from_row(row_dict, prefix="corrupt")
        question = row_dict.get("question")
        choices = row_dict.get("choices")
        for anchor in anchors:
            clean_positions = resolve_anchor_positions(
                tokenizer,
                str(row_dict["clean_prompt"]),
                anchor,
                question=question,
                choices=choices,
            )
            if clean_positions is None:
                rows.extend(_skip_rows(row_dict=row_dict, layers=layers, anchor=anchor, reason="clean_anchor_unresolved"))
                continue
            corrupt_positions = resolve_anchor_positions(
                tokenizer,
                str(row_dict["corrupt_prompt"]),
                anchor,
                question=question,
                choices=choices,
            )
            if corrupt_positions is None:
                rows.extend(_skip_rows(row_dict=row_dict, layers=layers, anchor=anchor, reason="corrupt_anchor_unresolved"))
                continue
            clean_position_values = [position.position for position in clean_positions]
            corrupt_position_values = [position.position for position in corrupt_positions]
            clean_hidden_by_layer = hidden_states_at_layers_and_positions(
                model,
                tokenizer,
                str(row_dict["clean_prompt"]),
                layers=layers,
                positions=clean_position_values,
                device=device,
            )
            replacements_by_layer = {}
            for layer in layers:
                replacements_by_layer[int(layer)] = {
                    corrupt_position.position: clean_hidden_by_layer[layer][clean_position.position]
                    for clean_position, corrupt_position in zip(clean_positions, corrupt_positions, strict=True)
                }
            scored_by_layer = score_layers_with_position_replacements(
                model,
                tokenizer,
                str(row_dict["corrupt_prompt"]),
                correct_label=correct_label,
                bias_scores=corrupt_bias,
                replacements_by_layer=replacements_by_layer,
                device=device,
            )
            for layer in layers:
                replacements = replacements_by_layer[int(layer)]
                scored = scored_by_layer[int(layer)]
                patched_margin = float(scored["cal_margin"])
                corrupt_margin = float(row_dict["corrupt_margin"])
                clean_margin = float(row_dict["clean_margin"])
                rows.append(
                    {
                        "item_id": row_dict["item_id"],
                        "subject": row_dict.get("subject"),
                        "clean_wrapper": row_dict["clean_wrapper"],
                        "corrupt_wrapper": row_dict["corrupt_wrapper"],
                        "split": row_dict["split"],
                        "layer": int(layer),
                        "anchor": anchor,
                        "anchor_components": ",".join(position.name for position in clean_positions),
                        "patch_kind": "grouped" if len(clean_positions) > 1 else "single",
                        "n_positions": len(replacements),
                        "clean_positions": _positions_to_string(clean_position_values),
                        "corrupt_positions": _positions_to_string(corrupt_position_values),
                        "clean_margin": clean_margin,
                        "corrupt_margin": corrupt_margin,
                        "patched_margin": patched_margin,
                        "patched_correct": bool(scored["cal_correct"]),
                        "patched_pred_label": scored["cal_pred_label"],
                        "margin_improvement": patched_margin - corrupt_margin,
                        "recovery": recovery(patched_margin, corrupt_margin, clean_margin),
                        "skipped": False,
                        "skip_reason": "",
                    }
                )
    return pd.DataFrame(rows)
