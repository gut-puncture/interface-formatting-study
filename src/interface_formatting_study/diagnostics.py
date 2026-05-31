from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd
import torch
from tqdm import tqdm

from .anchors import resolve_anchor_positions
from .patching import hidden_states_at_layers_and_positions
from .scoring import tokenize_text
from .token_roles import (
    CONTENT_ROLES,
    OPTION_ROLE_BY_LABEL,
    ROLE_INSTRUCTION,
    ROLE_QUESTION,
    ROLE_WRAPPER_SYNTAX,
    TOKEN_ROLES,
    label_token_roles,
)
from .utils import LABELS
from .vanilla import build_vanilla_prompt_from_row


@dataclass(frozen=True)
class PromptVariant:
    run_kind: str
    prompt: str
    wrapper: str
    margin: float | None
    pred_label: str | None


def safe_cosine_similarity(left: torch.Tensor, right: torch.Tensor, *, eps: float = 1.0e-12) -> float:
    left = left.detach().float().flatten()
    right = right.detach().float().flatten()
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if float(left_norm) <= eps or float(right_norm) <= eps:
        return float("nan")
    return float(torch.dot(left, right) / (left_norm * right_norm))


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _query_attention(layer_attention: torch.Tensor, query_positions: Iterable[int]) -> torch.Tensor:
    attention = layer_attention.detach().float()
    if attention.ndim == 4:
        if attention.shape[0] != 1:
            raise ValueError("attention diagnostics expect one prompt at a time")
        attention = attention[0]
    if attention.ndim != 3:
        raise ValueError(f"expected attention tensor with shape [heads, seq, seq], got {tuple(attention.shape)}")
    seq_len = int(attention.shape[-1])
    valid_positions = [int(position) for position in query_positions if 0 <= int(position) < seq_len]
    if not valid_positions:
        raise ValueError("no valid query positions for attention aggregation")
    return attention[:, valid_positions, :].mean(dim=1)


def attention_role_metrics(
    layer_attention: torch.Tensor,
    query_positions: Iterable[int],
    token_roles: list[str],
) -> pd.DataFrame:
    query_attention = _query_attention(layer_attention, query_positions)
    seq_len = int(query_attention.shape[-1])
    if len(token_roles) != seq_len:
        raise ValueError(f"role length {len(token_roles)} does not match attention sequence length {seq_len}")
    rows: list[dict[str, object]] = []
    for head in range(query_attention.shape[0]):
        weights = query_attention[head].clamp_min(0)
        total = float(weights.sum())
        if total > 0:
            weights = weights / weights.sum()
        masses: dict[str, float] = {}
        for role in TOKEN_ROLES:
            positions = [index for index, observed in enumerate(token_roles) if observed == role]
            masses[role] = float(weights[positions].sum()) if positions else 0.0
        option_mass = sum(masses[OPTION_ROLE_BY_LABEL[label]] for label in LABELS)
        content_mass = masses[ROLE_QUESTION] + option_mass
        entropy = float(-(weights * weights.clamp_min(1.0e-12).log()).sum())
        rows.append(
            {
                "head": int(head),
                "attention_mass_question": masses[ROLE_QUESTION],
                "attention_mass_options": option_mass,
                "attention_mass_question_options": content_mass,
                "attention_mass_wrapper_syntax": masses[ROLE_WRAPPER_SYNTAX],
                "attention_mass_instruction": masses[ROLE_INSTRUCTION],
                "attention_entropy": entropy,
                "content_vs_wrapper_focus_ratio": content_mass / (masses[ROLE_WRAPPER_SYNTAX] + 1.0e-12),
            }
        )
    return pd.DataFrame(rows)


def _forward_attentions(model, tokenizer, prompt: str, *, device=None):
    token_ids = tokenize_text(tokenizer, prompt)
    if not token_ids:
        raise ValueError("prompt has no tokens")
    device_obj = torch.device(device) if device is not None else _model_device(model)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device_obj)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
    attentions = getattr(outputs, "attentions", None)
    if not attentions or any(attention is None for attention in attentions):
        raise RuntimeError("model did not return attention tensors; try an eager attention implementation on the GPU host")
    return token_ids, attentions


def _variants_for_pair(pair: Mapping[str, object]) -> list[PromptVariant]:
    vanilla = build_vanilla_prompt_from_row(pair)
    return [
        PromptVariant(
            run_kind="clean_correct_wrapped",
            prompt=str(pair["clean_prompt"]),
            wrapper=str(pair["clean_wrapper"]),
            margin=float(pair["clean_margin"]),
            pred_label=None if pair.get("clean_pred_label") is None else str(pair.get("clean_pred_label")),
        ),
        PromptVariant(
            run_kind="corrupt_wrong_wrapped",
            prompt=str(pair["corrupt_prompt"]),
            wrapper=str(pair["corrupt_wrapper"]),
            margin=float(pair["corrupt_margin"]),
            pred_label=None if pair.get("corrupt_pred_label") is None else str(pair.get("corrupt_pred_label")),
        ),
        PromptVariant(
            run_kind="vanilla_same_item",
            prompt=vanilla,
            wrapper="vanilla",
            margin=None,
            pred_label=None,
        ),
    ]


def _cap_frame(frame: pd.DataFrame, *, cap: int | None, seed: int) -> pd.DataFrame:
    if cap is None or len(frame) <= cap:
        return frame.reset_index(drop=True)
    return frame.sample(n=cap, random_state=seed).sort_values("item_id").reset_index(drop=True)


def run_attention_diagnostics(
    model,
    tokenizer,
    conflict_pairs: pd.DataFrame,
    *,
    split: str = "validation",
    cap: int | None = 300,
    anchors: tuple[str, ...] = ("options_end", "all_option_ends"),
    seed: int = 1729,
    device=None,
) -> pd.DataFrame:
    source = _cap_frame(conflict_pairs[conflict_pairs["split"] == split].copy(), cap=cap, seed=seed)
    rows: list[dict[str, object]] = []
    for _, pair in tqdm(source.iterrows(), total=len(source), desc="attention diagnostics"):
        row = pair.to_dict()
        question = row.get("question")
        choices = row.get("choices")
        for variant in _variants_for_pair(row):
            labeling = label_token_roles(tokenizer, variant.prompt, question=question, choices=choices)
            token_ids, attentions = _forward_attentions(model, tokenizer, variant.prompt, device=device)
            if len(token_ids) != len(labeling.token_roles):
                raise RuntimeError("tokenization changed between role labeling and attention forward pass")
            for anchor in anchors:
                positions = resolve_anchor_positions(
                    tokenizer,
                    variant.prompt,
                    anchor,
                    question=question,
                    choices=choices,
                )
                if positions is None:
                    continue
                query_positions = [position.position for position in positions]
                for layer, layer_attention in enumerate(attentions):
                    metrics = attention_role_metrics(layer_attention, query_positions, labeling.token_roles)
                    for metric in metrics.to_dict("records"):
                        rows.append(
                            {
                                "item_id": row["item_id"],
                                "subject": row.get("subject"),
                                "split": row.get("split"),
                                "correct_label": row.get("correct_label"),
                                "run_kind": variant.run_kind,
                                "wrapper": variant.wrapper,
                                "anchor": anchor,
                                "anchor_components": ",".join(position.name for position in positions),
                                "query_positions": ",".join(str(position) for position in query_positions),
                                "prompt_tokens": len(token_ids),
                                "layer": int(layer),
                                "margin": variant.margin,
                                "pred_label": variant.pred_label,
                                **metric,
                            }
                        )
    return pd.DataFrame(rows)


def summarize_attention_diagnostics(attention: pd.DataFrame) -> pd.DataFrame:
    if attention.empty:
        return pd.DataFrame()
    grouped = (
        attention.groupby(["anchor", "layer", "run_kind"], as_index=False)
        .agg(
            n_items=("item_id", "nunique"),
            n_rows=("item_id", "size"),
            attention_mass_question=("attention_mass_question", "mean"),
            attention_mass_options=("attention_mass_options", "mean"),
            attention_mass_question_options=("attention_mass_question_options", "mean"),
            attention_mass_wrapper_syntax=("attention_mass_wrapper_syntax", "mean"),
            attention_mass_instruction=("attention_mass_instruction", "mean"),
            attention_entropy=("attention_entropy", "mean"),
            content_vs_wrapper_focus_ratio=("content_vs_wrapper_focus_ratio", "mean"),
        )
        .sort_values(["anchor", "layer", "run_kind"])
    )
    return grouped


def _mean_vectors_by_layer(
    model,
    tokenizer,
    prompt: str,
    *,
    layers: list[int],
    positions: list[int],
    device=None,
) -> dict[int, torch.Tensor]:
    captured = hidden_states_at_layers_and_positions(
        model,
        tokenizer,
        prompt,
        layers=layers,
        positions=positions,
        device=device,
    )
    out: dict[int, torch.Tensor] = {}
    for layer in layers:
        vectors = [captured[int(layer)][position] for position in positions]
        out[int(layer)] = torch.stack(vectors).mean(dim=0)
    return out


def run_vanilla_convergence(
    model,
    tokenizer,
    conflict_pairs: pd.DataFrame,
    *,
    layers: list[int],
    split: str = "validation",
    cap: int | None = 300,
    anchors: tuple[str, ...] = ("options_end", "all_option_ends"),
    seed: int = 1729,
    device=None,
) -> pd.DataFrame:
    source = _cap_frame(conflict_pairs[conflict_pairs["split"] == split].copy(), cap=cap, seed=seed)
    rows: list[dict[str, object]] = []
    for _, pair in tqdm(source.iterrows(), total=len(source), desc="vanilla convergence"):
        row = pair.to_dict()
        question = row.get("question")
        choices = row.get("choices")
        vanilla_prompt = build_vanilla_prompt_from_row(row)
        prompts = {
            "clean": str(row["clean_prompt"]),
            "corrupt": str(row["corrupt_prompt"]),
            "vanilla": vanilla_prompt,
        }
        for anchor in anchors:
            positions = {
                name: resolve_anchor_positions(tokenizer, prompt, anchor, question=question, choices=choices)
                for name, prompt in prompts.items()
            }
            if any(value is None for value in positions.values()):
                continue
            vectors = {
                name: _mean_vectors_by_layer(
                    model,
                    tokenizer,
                    prompts[name],
                    layers=layers,
                    positions=[position.position for position in positions[name] or []],
                    device=device,
                )
                for name in prompts
            }
            for layer in layers:
                clean_cos = safe_cosine_similarity(vectors["clean"][int(layer)], vectors["vanilla"][int(layer)])
                corrupt_cos = safe_cosine_similarity(vectors["corrupt"][int(layer)], vectors["vanilla"][int(layer)])
                rows.append(
                    {
                        "item_id": row["item_id"],
                        "subject": row.get("subject"),
                        "split": row.get("split"),
                        "correct_label": row.get("correct_label"),
                        "clean_wrapper": row.get("clean_wrapper"),
                        "corrupt_wrapper": row.get("corrupt_wrapper"),
                        "anchor": anchor,
                        "anchor_components": ",".join(position.name for position in positions["clean"] or []),
                        "layer": int(layer),
                        "clean_positions": ",".join(str(position.position) for position in positions["clean"] or []),
                        "corrupt_positions": ",".join(str(position.position) for position in positions["corrupt"] or []),
                        "vanilla_positions": ",".join(str(position.position) for position in positions["vanilla"] or []),
                        "cos_clean_vanilla": clean_cos,
                        "cos_corrupt_vanilla": corrupt_cos,
                        "delta_clean_minus_corrupt": clean_cos - corrupt_cos,
                    }
                )
    return pd.DataFrame(rows)


def summarize_vanilla_convergence(convergence: pd.DataFrame) -> pd.DataFrame:
    if convergence.empty:
        return pd.DataFrame()
    return (
        convergence.groupby(["anchor", "layer"], as_index=False)
        .agg(
            n_items=("item_id", "nunique"),
            cos_clean_vanilla=("cos_clean_vanilla", "mean"),
            cos_corrupt_vanilla=("cos_corrupt_vanilla", "mean"),
            delta_clean_minus_corrupt=("delta_clean_minus_corrupt", "mean"),
        )
        .sort_values(["anchor", "layer"])
    )
