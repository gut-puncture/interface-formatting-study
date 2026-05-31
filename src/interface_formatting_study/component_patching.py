from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping

import pandas as pd
import torch
from tqdm import tqdm

from .anchors import anchor_components, resolve_anchor_positions
from .interventions import bias_scores_from_row
from .patching import hidden_states_at_layers_and_positions, recovery, score_layers_with_position_replacements
from .utils import LABELS, validate_label
from .vanilla import build_vanilla_prompt_from_row

FOCUSED_CONTROL_CONDITIONS = (
    "same_item_clean_to_corrupt",
    "vanilla_to_corrupt",
    "cross_item_clean_to_corrupt",
    "same_label_cross_item_clean_to_corrupt",
    "different_label_cross_item_clean_to_corrupt",
)


@dataclass(frozen=True)
class DonorSelection:
    condition: str
    donor_item_id: str
    donor_label: str
    donor_wrapper: str
    donor_prompt: str
    donor_question: object
    donor_choices: object
    donor_margin: float | None


def _stable_pick(frame: pd.DataFrame, *, key: str) -> Mapping[str, object] | None:
    if frame.empty:
        return None
    ordered = frame.sort_values(["item_id", "clean_wrapper", "corrupt_wrapper"]).reset_index(drop=True)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    index = int(digest[:16], 16) % len(ordered)
    return ordered.iloc[index].to_dict()


def select_control_donor(
    donor_pool: pd.DataFrame,
    target: Mapping[str, object],
    *,
    condition: str,
    seed: int = 1729,
) -> DonorSelection | None:
    if condition not in FOCUSED_CONTROL_CONDITIONS:
        raise ValueError(f"unknown focused control condition: {condition}")
    target_item_id = str(target["item_id"])
    target_label = validate_label(str(target["correct_label"]), name="target correct_label")
    if condition == "same_item_clean_to_corrupt":
        return DonorSelection(
            condition=condition,
            donor_item_id=target_item_id,
            donor_label=target_label,
            donor_wrapper=str(target["clean_wrapper"]),
            donor_prompt=str(target["clean_prompt"]),
            donor_question=target.get("question"),
            donor_choices=target.get("choices"),
            donor_margin=float(target["clean_margin"]),
        )
    if condition == "vanilla_to_corrupt":
        return DonorSelection(
            condition=condition,
            donor_item_id=target_item_id,
            donor_label=target_label,
            donor_wrapper="vanilla",
            donor_prompt=build_vanilla_prompt_from_row(target),
            donor_question=target.get("question"),
            donor_choices=target.get("choices"),
            donor_margin=None,
        )

    eligible = donor_pool[donor_pool["item_id"].astype(str) != target_item_id].copy()
    if condition == "same_label_cross_item_clean_to_corrupt":
        eligible = eligible[eligible["correct_label"].astype(str).str.upper() == target_label]
    elif condition == "different_label_cross_item_clean_to_corrupt":
        eligible = eligible[eligible["correct_label"].astype(str).str.upper() != target_label]
    donor = _stable_pick(eligible, key=f"{seed}:{condition}:{target_item_id}:{target_label}")
    if donor is None:
        return None
    return DonorSelection(
        condition=condition,
        donor_item_id=str(donor["item_id"]),
        donor_label=validate_label(str(donor["correct_label"]), name="donor correct_label"),
        donor_wrapper=str(donor["clean_wrapper"]),
        donor_prompt=str(donor["clean_prompt"]),
        donor_question=donor.get("question"),
        donor_choices=donor.get("choices"),
        donor_margin=float(donor["clean_margin"]),
    )


def _cap_frame(frame: pd.DataFrame, *, cap: int | None, seed: int) -> pd.DataFrame:
    if cap is None or len(frame) <= cap:
        return frame.reset_index(drop=True)
    return frame.sample(n=cap, random_state=seed).sort_values("item_id").reset_index(drop=True)


def _positions_to_string(positions: list[int]) -> str:
    return ",".join(str(int(position)) for position in positions)


def _skip_rows(
    *,
    target: Mapping[str, object],
    donor: DonorSelection | None,
    layers: list[int],
    anchor: str,
    condition: str,
    reason: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    target_label = validate_label(str(target["correct_label"]), name="target correct_label")
    for layer in layers:
        rows.append(
            {
                "item_id": target["item_id"],
                "subject": target.get("subject"),
                "split": target.get("split"),
                "condition": condition,
                "anchor": anchor,
                "anchor_components": ",".join(anchor_components(anchor)),
                "layer": int(layer),
                "target_item_id": target["item_id"],
                "target_label": target_label,
                "target_corrupt_wrapper": target.get("corrupt_wrapper"),
                "donor_item_id": None if donor is None else donor.donor_item_id,
                "donor_label": None if donor is None else donor.donor_label,
                "donor_wrapper": None if donor is None else donor.donor_wrapper,
                "donor_item_equals_target": None if donor is None else donor.donor_item_id == str(target["item_id"]),
                "donor_label_relation": None if donor is None else ("same" if donor.donor_label == target_label else "different"),
                "donor_margin": None if donor is None else donor.donor_margin,
                "clean_margin": float(target["clean_margin"]),
                "corrupt_margin": float(target["corrupt_margin"]),
                "patched_margin": float("nan"),
                "patched_correct": False,
                "patched_pred_label": None,
                "margin_improvement": float("nan"),
                "recovery": float("nan"),
                "n_positions": 0,
                "donor_positions": "",
                "target_positions": "",
                "skipped": True,
                "skip_reason": reason,
            }
        )
    return rows


def _replacement_vectors_by_layer(
    model,
    tokenizer,
    donor_prompt: str,
    target_prompt: str,
    *,
    donor_question: object,
    donor_choices: object,
    target_question: object,
    target_choices: object,
    anchor: str,
    layers: list[int],
    device=None,
) -> tuple[dict[int, dict[int, torch.Tensor]], list[int], list[int]] | None:
    donor_positions = resolve_anchor_positions(
        tokenizer,
        donor_prompt,
        anchor,
        question=donor_question,
        choices=donor_choices,
    )
    target_positions = resolve_anchor_positions(
        tokenizer,
        target_prompt,
        anchor,
        question=target_question,
        choices=target_choices,
    )
    if donor_positions is None or target_positions is None:
        return None
    if len(donor_positions) != len(target_positions):
        return None
    donor_position_values = [position.position for position in donor_positions]
    target_position_values = [position.position for position in target_positions]
    donor_hidden = hidden_states_at_layers_and_positions(
        model,
        tokenizer,
        donor_prompt,
        layers=layers,
        positions=donor_position_values,
        device=device,
    )
    replacements_by_layer: dict[int, dict[int, torch.Tensor]] = {}
    for layer in layers:
        replacements_by_layer[int(layer)] = {
            target_position.position: donor_hidden[int(layer)][donor_position.position]
            for donor_position, target_position in zip(donor_positions, target_positions, strict=True)
        }
    return replacements_by_layer, donor_position_values, target_position_values


def run_focused_patching_controls(
    model,
    tokenizer,
    conflict_pairs: pd.DataFrame,
    *,
    layers: list[int],
    split: str = "validation",
    cap: int | None = 300,
    anchors: tuple[str, ...] = ("options_end", "all_option_ends"),
    conditions: tuple[str, ...] = FOCUSED_CONTROL_CONDITIONS,
    seed: int = 1729,
    device=None,
) -> pd.DataFrame:
    donor_pool = conflict_pairs[conflict_pairs["split"] == split].copy()
    source = _cap_frame(donor_pool.copy(), cap=cap, seed=seed)
    rows: list[dict[str, object]] = []
    for _, pair in tqdm(source.iterrows(), total=len(source), desc="focused patching controls"):
        target = pair.to_dict()
        target_label = validate_label(str(target["correct_label"]), name="target correct_label")
        corrupt_bias = bias_scores_from_row(target, prefix="corrupt")
        question = target.get("question")
        choices = target.get("choices")
        target_prompt = str(target["corrupt_prompt"])
        for condition in conditions:
            donor = select_control_donor(donor_pool, target, condition=condition, seed=seed)
            if donor is None:
                for anchor in anchors:
                    rows.extend(
                        _skip_rows(
                            target=target,
                            donor=None,
                            layers=layers,
                            anchor=anchor,
                            condition=condition,
                            reason="no_eligible_donor",
                        )
                    )
                continue
            for anchor in anchors:
                replacement_payload = _replacement_vectors_by_layer(
                    model,
                    tokenizer,
                    donor.donor_prompt,
                    target_prompt,
                    donor_question=donor.donor_question,
                    donor_choices=donor.donor_choices,
                    target_question=question,
                    target_choices=choices,
                    anchor=anchor,
                    layers=layers,
                    device=device,
                )
                if replacement_payload is None:
                    rows.extend(
                        _skip_rows(
                            target=target,
                            donor=donor,
                            layers=layers,
                            anchor=anchor,
                            condition=condition,
                            reason="anchor_unresolved",
                        )
                    )
                    continue
                replacements_by_layer, donor_positions, target_positions = replacement_payload
                scored_by_layer = score_layers_with_position_replacements(
                    model,
                    tokenizer,
                    target_prompt,
                    correct_label=target_label,
                    bias_scores=corrupt_bias,
                    replacements_by_layer=replacements_by_layer,
                    device=device,
                )
                for layer in layers:
                    scored = scored_by_layer[int(layer)]
                    patched_margin = float(scored["cal_margin"])
                    corrupt_margin = float(target["corrupt_margin"])
                    clean_margin = float(target["clean_margin"])
                    rows.append(
                        {
                            "item_id": target["item_id"],
                            "subject": target.get("subject"),
                            "split": target.get("split"),
                            "condition": condition,
                            "anchor": anchor,
                            "anchor_components": ",".join(anchor_components(anchor)),
                            "layer": int(layer),
                            "target_item_id": target["item_id"],
                            "target_label": target_label,
                            "target_corrupt_wrapper": target.get("corrupt_wrapper"),
                            "donor_item_id": donor.donor_item_id,
                            "donor_label": donor.donor_label,
                            "donor_wrapper": donor.donor_wrapper,
                            "donor_item_equals_target": donor.donor_item_id == str(target["item_id"]),
                            "donor_label_relation": "same" if donor.donor_label == target_label else "different",
                            "donor_margin": donor.donor_margin,
                            "clean_margin": clean_margin,
                            "corrupt_margin": corrupt_margin,
                            "patched_margin": patched_margin,
                            "patched_correct": bool(scored["cal_correct"]),
                            "patched_pred_label": scored["cal_pred_label"],
                            "margin_improvement": patched_margin - corrupt_margin,
                            "recovery": recovery(patched_margin, corrupt_margin, clean_margin),
                            "n_positions": len(target_positions),
                            "donor_positions": _positions_to_string(donor_positions),
                            "target_positions": _positions_to_string(target_positions),
                            "skipped": False,
                            "skip_reason": "",
                        }
                    )
    return pd.DataFrame(rows)


def summarize_focused_controls(controls: pd.DataFrame) -> pd.DataFrame:
    if controls.empty:
        return pd.DataFrame()
    source = controls.copy()
    if "skipped" in source.columns:
        source = source[~source["skipped"].astype(bool)].copy()
    if source.empty:
        return pd.DataFrame()
    return (
        source.groupby(["condition", "anchor", "layer"], as_index=False)
        .agg(
            n_items=("item_id", "nunique"),
            n_rows=("item_id", "size"),
            n_positions=("n_positions", "mean"),
            patched_accuracy=("patched_correct", "mean"),
            mean_margin_improvement=("margin_improvement", "mean"),
            mean_recovery=("recovery", "mean"),
            donor_item_equals_target_rate=("donor_item_equals_target", "mean"),
            same_label_rate=("donor_label_relation", lambda values: float((values == "same").mean())),
        )
        .sort_values(["anchor", "layer", "condition"])
    )
