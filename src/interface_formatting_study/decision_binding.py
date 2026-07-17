from __future__ import annotations

import hashlib
import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .hooks import find_transformer_blocks
from .patching import _scored_label_result, score_layers_with_position_replacements
from .scoring import predict_from_scores, single_token_label_ids, tokenize_text


LABELS = ("A", "B", "C", "D")
TERMINAL_LETTER_INSTRUCTION = "Return only the letter (A, B, C, or D)."


@dataclass(frozen=True)
class WinnerCoordinates:
    label_index: int
    position: int
    content_id: int


@dataclass(frozen=True)
class CheckpointIndices:
    format_end: int
    answer_prefix_end: int
    sequence_length: int

    def padded(self, *, padded_length: int, padding_side: str) -> tuple[int, int]:
        if padded_length < self.sequence_length:
            raise ValueError("padded_length is shorter than the unpadded sequence")
        if padding_side not in {"left", "right"}:
            raise ValueError("padding_side must be 'left' or 'right'")
        shift = padded_length - self.sequence_length if padding_side == "left" else 0
        return self.format_end + shift, self.answer_prefix_end + shift


def _sequence(value: object, *, name: str) -> list[object]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} is not a JSON sequence") from exc
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray, str)):
        raise ValueError(f"{name} must be a sequence")
    return list(value)


def winner_coordinates(
    *,
    label: str,
    labels_by_position: object,
    content_ids_by_position: object,
) -> WinnerCoordinates:
    labels = [str(value) for value in _sequence(labels_by_position, name="labels_by_position")]
    contents = [int(value) for value in _sequence(content_ids_by_position, name="content_ids_by_position")]
    if sorted(labels) != list(LABELS):
        raise ValueError("labels_by_position must be a permutation of A, B, C, D")
    if sorted(contents) != list(range(4)):
        raise ValueError("content_ids_by_position must be a permutation of 0, 1, 2, 3")
    normalized = str(label)
    if normalized not in LABELS:
        raise ValueError(f"Unknown winner label: {normalized!r}")
    position = labels.index(normalized)
    return WinnerCoordinates(
        label_index=LABELS.index(normalized),
        position=position,
        content_id=contents[position],
    )


def _flat_tokenizer_field(value: object, *, name: str) -> list[object]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray, str)):
        raise ValueError(f"Tokenizer {name} is not a sequence")
    values = list(value)
    if len(values) == 1 and isinstance(values[0], Sequence) and not isinstance(
        values[0], (bytes, bytearray, str)
    ):
        values = list(values[0])
    return values


def resolve_readout_checkpoints(tokenizer, prompt: str) -> CheckpointIndices:
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Prompt must be a non-empty string")
    marker_start = prompt.rfind(TERMINAL_LETTER_INSTRUCTION)
    if marker_start < 0:
        raise ValueError("Prompt is missing the terminal letter instruction")
    content_prefix = prompt[:marker_start].rstrip()
    if not content_prefix:
        raise ValueError("Prompt has no content before the terminal letter instruction")

    try:
        encoded = tokenizer(
            prompt,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (TypeError, NotImplementedError) as exc:
        raise ValueError("Tokenizer does not provide offset mappings") from exc
    if "input_ids" not in encoded or "offset_mapping" not in encoded:
        raise ValueError("Tokenizer did not return token IDs and offset mappings")
    input_ids = [int(value) for value in _flat_tokenizer_field(encoded["input_ids"], name="input_ids")]
    offsets_raw = _flat_tokenizer_field(encoded["offset_mapping"], name="offset_mapping")
    offsets = [tuple(int(part) for part in value) for value in offsets_raw]
    if len(input_ids) != len(offsets) or not input_ids:
        raise ValueError("Tokenizer token IDs and offsets have inconsistent lengths")
    if input_ids != [int(value) for value in tokenize_text(tokenizer, prompt)]:
        raise ValueError("Tokenizer offset token IDs differ from the model prompt token IDs")

    content_character = len(content_prefix) - 1
    matching = [
        index
        for index, (start, end) in enumerate(offsets)
        if start <= content_character < end
    ]
    if len(matching) != 1:
        raise ValueError("Could not resolve the final content character to exactly one token")
    format_end = matching[0]
    if offsets[format_end][1] > marker_start:
        raise ValueError("The format-end token overlaps the response instruction")
    return CheckpointIndices(
        format_end=format_end,
        answer_prefix_end=len(input_ids) - 1,
        sequence_length=len(input_ids),
    )


_SCORE_COLUMNS = tuple(
    f"{kind}_score_{label}"
    for kind in ("raw", "bias", "cal")
    for label in LABELS
)
_REQUIRED_SCORE_COLUMNS = {
    "work_key",
    "item_id",
    "subject",
    "split",
    "wrapper_name",
    "arm",
    "variant",
    "manipulation",
    "position_shift",
    "label_shift",
    "source_prompt_sha256",
    "prompt",
    "prompt_sha256",
    "content_ids_by_position",
    "labels_by_position",
    "candidate_texts",
    "correct_content_id",
    "raw_predicted_label",
    "raw_predicted_position",
    "raw_predicted_content_id",
    "raw_tie",
    *_SCORE_COLUMNS,
}
_REQUIRED_APPLICABILITY_COLUMNS = {
    "item_id",
    "subject",
    "split",
    "wrapper_name",
    "source_prompt_sha256",
    "baseline_applicable",
    "position_applicable",
    "label_applicable",
    "not_applicable_reason",
}

_CANONICAL_SEMANTIC_COLUMNS = (
    "item_id",
    "subject",
    "split",
    "wrapper_name",
    "arm",
    "variant",
    "manipulation",
    "position_shift",
    "label_shift",
    "template_scope",
    "source_prompt_sha256",
    "prompt",
    "prompt_sha256",
    "content_ids_by_position",
    "labels_by_position",
    "candidate_texts",
    "correct_content_id",
    "correct_position",
    "correct_label",
)


def _canonical_cell(value: object) -> object:
    if isinstance(value, np.ndarray):
        return [_canonical_cell(child) for child in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_canonical_cell(child) for child in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def validate_canonical_sources(
    scored: pd.DataFrame,
    applicability: pd.DataFrame,
    canonical_design: pd.DataFrame,
    *,
    stage: Literal["discovery", "confirmation"],
    expected_items: Mapping[str, int],
) -> dict[str, int]:
    """Fail closed unless scored rows exactly preserve the canonical design semantics."""
    allowed_splits = ("train", "validation") if stage == "discovery" else ("test",)
    if set(expected_items) != set(allowed_splits):
        raise ValueError("Expected item denominators do not match the experiment stage")
    canonical = canonical_design[
        (canonical_design["arm"] == "letter_intervention")
        & canonical_design["split"].astype(str).isin(allowed_splits)
    ].copy()
    observed = scored[
        (scored["arm"] == "letter_intervention")
        & scored["split"].astype(str).isin(allowed_splits)
    ].copy()
    for name, frame in (("canonical", canonical), ("scored", observed)):
        missing = {"work_key", *_CANONICAL_SEMANTIC_COLUMNS} - set(frame.columns)
        if missing:
            raise ValueError(f"{name} rows are missing canonical columns: {sorted(missing)}")
        if frame["work_key"].duplicated().any():
            raise ValueError(f"{name} rows contain duplicate work keys")

    expected_keys = set(canonical["work_key"].astype(str))
    observed_keys = set(observed["work_key"].astype(str))
    if observed_keys != expected_keys:
        raise ValueError(
            "Scored rows do not match canonical work keys; "
            f"missing={sorted(expected_keys - observed_keys)[:3]} "
            f"extra={sorted(observed_keys - expected_keys)[:3]}"
        )
    left = observed.set_index("work_key").loc[sorted(expected_keys)]
    right = canonical.set_index("work_key").loc[sorted(expected_keys)]
    mismatches: list[str] = []
    for column in _CANONICAL_SEMANTIC_COLUMNS:
        unequal = [
            key
            for key, observed_value, canonical_value in zip(
                left.index,
                left[column],
                right[column],
                strict=True,
            )
            if _canonical_cell(observed_value) != _canonical_cell(canonical_value)
        ]
        if unequal:
            mismatches.append(f"{column}:{unequal[0]}")
    if mismatches:
        raise ValueError(f"Scored rows differ in canonical semantic columns: {mismatches[:3]}")

    counts = {
        split: int(canonical.loc[canonical["split"].astype(str) == split, "item_id"].nunique())
        for split in allowed_splits
    }
    if counts != {str(key): int(value) for key, value in expected_items.items()}:
        raise ValueError(f"Canonical split denominators differ: observed={counts} expected={expected_items}")
    expected_blocks = {
        tuple(str(value) for value in row)
        for row in canonical[["item_id", "wrapper_name", "split"]].drop_duplicates().itertuples(
            index=False, name=None
        )
    }
    observed_blocks = {
        tuple(str(value) for value in row)
        for row in applicability[
            applicability["split"].astype(str).isin(allowed_splits)
        ][["item_id", "wrapper_name", "split"]].itertuples(index=False, name=None)
    }
    if observed_blocks != expected_blocks:
        raise ValueError("Applicability rows do not match canonical item/format blocks")
    return counts


def _validate_block(
    block: pd.DataFrame,
    applicability: pd.Series,
) -> None:
    key = f"{applicability['item_id']}::{applicability['wrapper_name']}"
    baseline = block[block["manipulation"] == "controlled_baseline"]
    if len(baseline) != 1:
        raise ValueError(f"{key} expected 1 controlled_baseline row, found {len(baseline)}")
    if not bool(applicability["baseline_applicable"]):
        raise ValueError(f"{key} marks its required baseline not applicable")
    baseline_row = baseline.iloc[0]
    if str(baseline_row["prompt_sha256"]) != str(applicability["source_prompt_sha256"]):
        raise ValueError(f"{key} baseline does not match the applicability source prompt")
    if str(baseline_row["source_prompt_sha256"]) != str(baseline_row["prompt_sha256"]):
        raise ValueError(f"{key} baseline does not match its source prompt")

    for manipulation, applicability_column in (
        ("position_only", "position_applicable"),
        ("label_only", "label_applicable"),
    ):
        observed = len(block[block["manipulation"] == manipulation])
        expected = 3 if bool(applicability[applicability_column]) else 0
        if observed != expected:
            raise ValueError(f"{key} expected {expected} {manipulation} rows, found {observed}")


def _stable_digest(seed: str, *parts: object) -> str:
    payload = "|".join((seed, *(str(part) for part in parts)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _balanced_confirmation_rows(
    frame: pd.DataFrame,
    *,
    manipulation: Literal["position_only", "label_only"],
    variants: tuple[int, int, int],
    seed: str,
) -> pd.DataFrame:
    subset = frame[frame["manipulation"] == manipulation]
    if subset.empty:
        return subset.copy()
    groups: list[tuple[str, tuple[str, str], pd.DataFrame]] = []
    for (item_id, wrapper_name), group in subset.groupby(
        ["item_id", "wrapper_name"], sort=False
    ):
        groups.append(
            (
                _stable_digest(seed, manipulation, item_id, wrapper_name),
                (str(item_id), str(wrapper_name)),
                group,
            )
        )
    selected: list[pd.Series] = []
    for index, (_, _, group) in enumerate(sorted(groups, key=lambda value: value[:2])):
        variant = variants[index % len(variants)]
        match = group[group["variant"].astype(int) == variant]
        if len(match) != 1:
            raise ValueError(
                f"Confirmation block does not contain exactly one {manipulation} variant {variant}"
            )
        selected.append(match.iloc[0])
    return pd.DataFrame(selected, columns=frame.columns)


def _validation_readout_roles(frame: pd.DataFrame, *, seed: str) -> dict[str, str]:
    items = frame[["item_id", "subject"]].drop_duplicates()
    if items.groupby("item_id", sort=False)["subject"].nunique().gt(1).any():
        raise ValueError("A validation item cannot belong to multiple subjects")
    items = items.drop_duplicates("item_id")
    layer_select_target = len(items) // 2
    selected_by_subject: dict[str, int] = {}
    odd_subjects: list[str] = []
    subject_items: dict[str, list[str]] = {}
    for subject, group in items.groupby("subject", sort=False):
        normalized_subject = str(subject)
        ordered = sorted(
            group["item_id"].astype(str),
            key=lambda item_id: (
                _stable_digest(seed, "validation-role", normalized_subject, item_id),
                item_id,
            ),
        )
        subject_items[normalized_subject] = ordered
        selected_by_subject[normalized_subject] = len(ordered) // 2
        if len(ordered) % 2:
            odd_subjects.append(normalized_subject)
    remaining = layer_select_target - sum(selected_by_subject.values())
    for subject in sorted(
        odd_subjects,
        key=lambda value: (_stable_digest(seed, "validation-role-extra", value), value),
    )[:remaining]:
        selected_by_subject[subject] += 1

    roles: dict[str, str] = {}
    for subject, item_ids in subject_items.items():
        selected = set(item_ids[: selected_by_subject[subject]])
        roles.update(
            {
                item_id: "layer_select" if item_id in selected else "reader_gate"
                for item_id in item_ids
            }
        )
    return roles


def prepare_readout_ledger(
    scored: pd.DataFrame,
    applicability: pd.DataFrame,
    *,
    stage: Literal["discovery", "confirmation"],
    seed: str = "decision-binding-v1",
) -> pd.DataFrame:
    if stage not in {"discovery", "confirmation"}:
        raise ValueError("stage must be 'discovery' or 'confirmation'")
    missing_scores = _REQUIRED_SCORE_COLUMNS - set(scored.columns)
    if missing_scores:
        raise ValueError(f"Scored rows are missing columns: {sorted(missing_scores)}")
    missing_applicability = _REQUIRED_APPLICABILITY_COLUMNS - set(applicability.columns)
    if missing_applicability:
        raise ValueError(
            f"Applicability rows are missing columns: {sorted(missing_applicability)}"
        )

    allowed_splits = {"train", "validation"} if stage == "discovery" else {"test"}
    rows = scored[
        (scored["arm"] == "letter_intervention")
        & scored["split"].astype(str).isin(allowed_splits)
    ].copy()
    statuses = applicability[applicability["split"].astype(str).isin(allowed_splits)].copy()
    if rows.empty or statuses.empty:
        raise ValueError(f"No {stage} rows were supplied")
    if rows["work_key"].duplicated().any():
        raise ValueError("Scored letter rows contain duplicate work keys")
    status_keys = ["item_id", "wrapper_name", "split"]
    if statuses.duplicated(status_keys).any():
        raise ValueError("Applicability rows contain duplicate item/format/split keys")

    transformed_records: list[dict[str, object]] = []
    for record in rows.to_dict("records"):
        prompt = str(record["prompt"])
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if str(record["prompt_sha256"]) != digest:
            raise ValueError(f"{record['work_key']} has a prompt checksum mismatch")
        source_digest = str(record["source_prompt_sha256"])
        if len(source_digest) != 64 or any(character not in "0123456789abcdef" for character in source_digest):
            raise ValueError(f"{record['work_key']} has an invalid source prompt checksum")
        coordinates = winner_coordinates(
            label=str(record["raw_predicted_label"]),
            labels_by_position=record["labels_by_position"],
            content_ids_by_position=record["content_ids_by_position"],
        )
        if (
            int(record["raw_predicted_position"]) != coordinates.position
            or int(record["raw_predicted_content_id"]) != coordinates.content_id
        ):
            raise ValueError(f"{record['work_key']} has inconsistent stored raw winner coordinates")
        candidate_texts = [str(value) for value in _sequence(record["candidate_texts"], name="candidate_texts")]
        if len(candidate_texts) != 4 or any(not value for value in candidate_texts):
            raise ValueError(f"{record['work_key']} must have four non-empty candidate texts")
        if any(not math.isfinite(float(record[column])) for column in _SCORE_COLUMNS):
            raise ValueError(f"{record['work_key']} has a non-finite A-D score")
        expected_label, expected_tie = predict_from_scores(
            {label: float(record[f"raw_score_{label}"]) for label in LABELS}
        )
        if (
            str(record["raw_predicted_label"]) != expected_label
            or bool(record["raw_tie"]) != expected_tie
        ):
            raise ValueError(f"{record['work_key']} raw winner does not match raw scores")
        transformed_records.append(
            {
                **record,
                "content_ids_by_position": [
                    int(value)
                    for value in _sequence(
                        record["content_ids_by_position"], name="content_ids_by_position"
                    )
                ],
                "labels_by_position": [
                    str(value)
                    for value in _sequence(record["labels_by_position"], name="labels_by_position")
                ],
                "candidate_texts": candidate_texts,
                "winner_label_index": coordinates.label_index,
                "winner_position": coordinates.position,
                "winner_content_id": coordinates.content_id,
                "winner_unique": not bool(record["raw_tie"]),
                "text_identity_ambiguous": len(set(candidate_texts)) != 4,
                "content_evaluable": (
                    not bool(record["raw_tie"]) and len(set(candidate_texts)) == 4
                ),
                "position_evaluable": not bool(record["raw_tie"]),
                "label_evaluable": not bool(record["raw_tie"]),
            }
        )
    normalized = pd.DataFrame(transformed_records)

    block_columns = ["item_id", "wrapper_name", "split"]
    normalized_keys = {
        tuple(str(value) for value in key)
        for key in normalized[block_columns].itertuples(index=False, name=None)
    }
    status_key_set = {
        tuple(str(value) for value in key)
        for key in statuses[block_columns].itertuples(index=False, name=None)
    }
    if normalized_keys != status_key_set:
        missing = sorted(status_key_set - normalized_keys)
        extra = sorted(normalized_keys - status_key_set)
        raise ValueError(f"Scored/applicability block mismatch; missing={missing[:3]} extra={extra[:3]}")
    for _, status in statuses.iterrows():
        block = normalized[
            (normalized["item_id"].astype(str) == str(status["item_id"]))
            & (normalized["wrapper_name"].astype(str) == str(status["wrapper_name"]))
            & (normalized["split"].astype(str) == str(status["split"]))
        ]
        if not block["source_prompt_sha256"].astype(str).eq(str(status["source_prompt_sha256"])).all():
            raise ValueError(
                f"{status['item_id']}::{status['wrapper_name']} source prompt checksum drift"
            )
        _validate_block(block, status)

    if stage == "discovery":
        train = normalized[
            (normalized["split"] == "train")
            & (normalized["wrapper_name"] == "plain")
        ].copy()
        train_variants = train.groupby("item_id", sort=False)["variant"].agg(
            lambda values: {int(value) for value in values}
        )
        if not train_variants.empty and not train_variants.map(
            lambda variants: variants == set(range(7))
        ).all():
            raise ValueError("Discovery requires all seven plain training variants per item")
        validation = normalized[normalized["split"] == "validation"].copy()
        train["readout_role"] = "probe_train"
        validation_roles = _validation_readout_roles(validation, seed=seed)
        validation["readout_role"] = (
            validation["item_id"].astype(str).map(validation_roles)
        )
        if validation["readout_role"].isna().any():
            raise AssertionError("Validation role assignment omitted an item")
        selected = pd.concat([train, validation], ignore_index=True)
    else:
        baseline = normalized[normalized["manipulation"] == "controlled_baseline"].copy()
        position = _balanced_confirmation_rows(
            normalized,
            manipulation="position_only",
            variants=(1, 2, 3),
            seed=seed,
        )
        label = _balanced_confirmation_rows(
            normalized,
            manipulation="label_only",
            variants=(4, 5, 6),
            seed=seed,
        )
        selected = pd.concat([baseline, position, label], ignore_index=True)
        selected["readout_role"] = "confirmation"
    if selected.empty:
        raise ValueError(f"{stage} selection produced no readout rows")
    selected["readout_work_key"] = stage + "|" + selected["work_key"].astype(str)
    if selected["readout_work_key"].duplicated().any():
        raise AssertionError("Readout ledger produced duplicate work keys")
    return selected.sort_values("readout_work_key", kind="mergesort").reset_index(drop=True)


@dataclass(frozen=True)
class CapturedReadouts:
    activations: torch.Tensor
    raw_log_probs: torch.Tensor
    batches: int
    actual_tokens: int
    padded_tokens: int
    input_preparation_seconds: float = 0.0
    forward_seconds: float = 0.0


def capture_layer_readouts(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    batch_size: int,
    max_batch_tokens: int | None = None,
    device=None,
) -> CapturedReadouts:
    preparation_started = time.monotonic()
    if not prompts or batch_size <= 0:
        raise ValueError("prompts must be non-empty and batch_size must be positive")
    label_ids = single_token_label_ids(tokenizer)
    if label_ids is None or len(set(label_ids.values())) != 4:
        raise ValueError("Decision-binding readout requires four distinct single-token A-D labels")
    encoded = [tokenize_text(tokenizer, str(prompt)) for prompt in prompts]
    checkpoints = [resolve_readout_checkpoints(tokenizer, str(prompt)) for prompt in prompts]
    if any(len(ids) != point.sequence_length for ids, point in zip(encoded, checkpoints, strict=True)):
        raise ValueError("Checkpoint sequence length does not match prompt tokenization")
    order = sorted(range(len(prompts)), key=lambda index: (len(encoded[index]), index))
    groups: list[list[int]] = []
    current: list[int] = []
    current_max = 0
    for index in order:
        length = len(encoded[index])
        candidate_max = max(current_max, length)
        if current and (
            len(current) >= batch_size
            or (max_batch_tokens is not None and candidate_max * (len(current) + 1) > max_batch_tokens)
        ):
            groups.append(current)
            current, current_max = [], 0
        current.append(index)
        current_max = max(current_max, length)
    if current:
        groups.append(current)
    input_preparation_seconds = time.monotonic() - preparation_started

    blocks = find_transformer_blocks(model)
    output_activations: torch.Tensor | None = None
    output_scores: torch.Tensor | None = None
    device_obj = device or next(model.parameters()).device
    pad_id = int(getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0) or 0)
    padded_tokens = 0
    forward_seconds = 0.0
    for indices in groups:
        preparation_started = time.monotonic()
        max_len = max(len(encoded[index]) for index in indices)
        input_ids = torch.full((len(indices), max_len), pad_id, dtype=torch.long, device=device_obj)
        attention_mask = torch.zeros_like(input_ids)
        positions = torch.empty((len(indices), 2), dtype=torch.long, device=device_obj)
        for row, index in enumerate(indices):
            ids = encoded[index]
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device_obj)
            attention_mask[row, : len(ids)] = 1
            positions[row] = torch.tensor(
                [checkpoints[index].format_end, checkpoints[index].answer_prefix_end],
                device=device_obj,
            )
        captured: dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(layer: int):
            def hook(_module, _inputs, block_output):
                hidden = block_output[0] if isinstance(block_output, tuple) else block_output
                row_indices = torch.arange(hidden.shape[0], device=hidden.device).unsqueeze(1)
                captured[layer] = hidden[row_indices, positions].detach()
                return block_output

            return hook

        try:
            for layer, block in enumerate(blocks):
                handles.append(block.register_forward_hook(make_hook(layer)))
            input_preparation_seconds += time.monotonic() - preparation_started
            if device_obj.type == "cuda":
                torch.cuda.synchronize(device_obj)
            forward_started = time.monotonic()
            with torch.inference_mode():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            if device_obj.type == "cuda":
                torch.cuda.synchronize(device_obj)
            forward_seconds += time.monotonic() - forward_started
        finally:
            for handle in handles:
                handle.remove()
        if set(captured) != set(range(len(blocks))):
            raise RuntimeError("Not every transformer-block capture hook ran")
        layer_tensor = torch.stack(
            [captured[layer] for layer in range(len(blocks))], dim=1
        ).float().cpu()
        rows = torch.arange(len(indices), device=logits.device)
        last_positions = positions[:, 1]
        log_probs = torch.log_softmax(logits[rows, last_positions], dim=-1)
        label_tensor = torch.tensor([label_ids[label] for label in LABELS], device=logits.device)
        scores = log_probs.index_select(-1, label_tensor).detach().float().cpu()
        if output_activations is None:
            output_activations = torch.empty(
                (len(prompts), *layer_tensor.shape[1:]), dtype=layer_tensor.dtype
            )
            output_scores = torch.empty((len(prompts), 4), dtype=scores.dtype)
        for row, index in enumerate(indices):
            output_activations[index] = layer_tensor[row]
            assert output_scores is not None
            output_scores[index] = scores[row]
        padded_tokens += len(indices) * max_len
    if output_activations is None or output_scores is None:
        raise RuntimeError("Batched readout did not produce every requested row")
    return CapturedReadouts(
        activations=output_activations,
        raw_log_probs=output_scores,
        batches=len(groups),
        actual_tokens=sum(map(len, encoded)),
        padded_tokens=padded_tokens,
        input_preparation_seconds=input_preparation_seconds,
        forward_seconds=forward_seconds,
    )


@dataclass(frozen=True)
class ProbeBank:
    weights: np.ndarray
    intercepts: np.ndarray
    means: np.ndarray
    classes: np.ndarray
    c: float
    target_name: str = "content"

    def log_probabilities(self, activations: np.ndarray | torch.Tensor) -> np.ndarray:
        values = np.asarray(activations, dtype=np.float64)
        if values.shape[1:3] != self.weights.shape[:2] or values.shape[-1] != self.weights.shape[-1]:
            raise ValueError("Activation shape does not match the probe bank")
        logits = np.einsum("nlcd,lckd->nlck", values - self.means[None], self.weights)
        logits += self.intercepts[None]
        logits -= logits.max(axis=-1, keepdims=True)
        return logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))


@dataclass(frozen=True)
class PatchVectors:
    probe: torch.Tensor
    full: torch.Tensor
    random: torch.Tensor
    identity: torch.Tensor


@dataclass(frozen=True)
class PatchTargets:
    donor_content_id: int
    receiver_content_label: str
    donor_symbol_label: str


_CHECKPOINT_INDEX = {"format_end": 0, "answer_prefix_end": 1}


def probe_subspace_basis(
    bank: ProbeBank,
    *,
    layer: int,
    checkpoint: int | Literal["format_end", "answer_prefix_end"],
) -> np.ndarray:
    checkpoint_index = _CHECKPOINT_INDEX.get(checkpoint, checkpoint)
    if not isinstance(checkpoint_index, (int, np.integer)):
        raise ValueError(f"Unknown checkpoint: {checkpoint!r}")
    if not 0 <= int(layer) < bank.weights.shape[0]:
        raise ValueError(f"Layer is outside the probe bank: {layer}")
    if not 0 <= int(checkpoint_index) < bank.weights.shape[1]:
        raise ValueError(f"Checkpoint is outside the probe bank: {checkpoint!r}")
    weights = np.asarray(bank.weights[int(layer), int(checkpoint_index)], dtype=np.float64)
    if weights.ndim != 2 or weights.shape[0] != 4 or not np.isfinite(weights).all():
        raise ValueError("Probe weights must contain four finite class vectors")
    centered = weights - weights.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if not singular_values.size or singular_values[0] == 0:
        raise ValueError("Probe class contrasts have rank zero")
    tolerance = np.finfo(np.float64).eps * max(centered.shape) * singular_values[0]
    rank = min(3, int(np.count_nonzero(singular_values > tolerance)))
    if rank == 0:
        raise ValueError("Probe class contrasts have rank zero")
    basis = vh[:rank].T.copy()
    # SVD vector signs are arbitrary. Fix them so persisted controls are reproducible.
    for column in range(rank):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0:
            basis[:, column] *= -1
    return basis


def _validated_basis(basis: np.ndarray, hidden: int) -> np.ndarray:
    value = np.asarray(basis, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != hidden or not 1 <= value.shape[1] <= 3:
        raise ValueError("Probe basis must have shape [hidden, rank] with rank 1-3")
    if not np.isfinite(value).all() or not np.allclose(
        value.T @ value, np.eye(value.shape[1]), atol=1e-10, rtol=1e-10
    ):
        raise ValueError("Probe basis must be finite and orthonormal")
    return value


def projected_replacement(
    receiver: torch.Tensor,
    donor: torch.Tensor,
    basis: np.ndarray,
) -> torch.Tensor:
    if receiver.ndim != 1 or donor.shape != receiver.shape:
        raise ValueError("Receiver and donor must be same-shaped one-dimensional residuals")
    value = _validated_basis(basis, receiver.numel())
    receiver64 = receiver.detach().to(dtype=torch.float64, device="cpu")
    delta = donor.detach().to(dtype=torch.float64, device="cpu") - receiver64
    basis_tensor = torch.from_numpy(value)
    projected = basis_tensor @ (basis_tensor.T @ delta)
    return (receiver64 + projected).to(device=receiver.device, dtype=receiver.dtype)


def build_patch_vectors(
    receiver: torch.Tensor,
    donor: torch.Tensor,
    probe_basis: np.ndarray,
    *,
    seed: str,
) -> PatchVectors:
    if receiver.ndim != 1 or donor.shape != receiver.shape:
        raise ValueError("Receiver and donor must be same-shaped one-dimensional residuals")
    basis = _validated_basis(probe_basis, receiver.numel())
    hidden, rank = basis.shape
    if hidden - rank < rank:
        raise ValueError("Hidden size is too small for a rank-matched orthogonal control")
    receiver64 = receiver.detach().to(dtype=torch.float64, device="cpu")
    donor64 = donor.detach().to(dtype=torch.float64, device="cpu")
    basis_tensor = torch.from_numpy(basis)
    probe_delta = basis_tensor @ (basis_tensor.T @ (donor64 - receiver64))
    probe_norm = float(torch.linalg.vector_norm(probe_delta))

    digest = hashlib.sha256(str(seed).encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big", signed=False))
    random_basis = None
    for _ in range(4):
        candidate = rng.standard_normal((hidden, rank))
        candidate -= basis @ (basis.T @ candidate)
        q, r = np.linalg.qr(candidate, mode="reduced")
        if np.min(np.abs(np.diag(r))) > 1e-10:
            random_basis = q
            break
    if random_basis is None:
        raise RuntimeError("Could not construct the deterministic orthogonal control basis")
    random_basis_tensor = torch.from_numpy(random_basis)
    random_delta = random_basis_tensor @ (random_basis_tensor.T @ (donor64 - receiver64))
    random_norm = float(torch.linalg.vector_norm(random_delta))
    if probe_norm == 0:
        random_delta.zero_()
    elif random_norm > 0:
        random_delta *= probe_norm / random_norm
    else:
        random_delta = random_basis_tensor[:, 0] * probe_norm

    def restore(value: torch.Tensor) -> torch.Tensor:
        return value.to(device=receiver.device, dtype=receiver.dtype)

    return PatchVectors(
        probe=restore(receiver64 + probe_delta),
        full=donor.detach().to(device=receiver.device, dtype=receiver.dtype).clone(),
        random=restore(receiver64 + random_delta),
        identity=receiver.detach().clone(),
    )


def patch_target_labels(donor: object, receiver: object) -> PatchTargets:
    donor_label = str(donor["raw_predicted_label"])
    donor_coordinates = winner_coordinates(
        label=donor_label,
        labels_by_position=donor["labels_by_position"],
        content_ids_by_position=donor["content_ids_by_position"],
    )
    receiver_labels = [
        str(value)
        for value in _sequence(receiver["labels_by_position"], name="labels_by_position")
    ]
    receiver_contents = [
        int(value)
        for value in _sequence(
            receiver["content_ids_by_position"], name="content_ids_by_position"
        )
    ]
    # Reuse the coordinate validator instead of trusting separately stored mappings.
    winner_coordinates(
        label="A",
        labels_by_position=receiver_labels,
        content_ids_by_position=receiver_contents,
    )
    receiver_position = receiver_contents.index(donor_coordinates.content_id)
    return PatchTargets(
        donor_content_id=donor_coordinates.content_id,
        receiver_content_label=receiver_labels[receiver_position],
        donor_symbol_label=donor_label,
    )


def target_margin(scores: dict[str, float], target_label: str) -> float:
    if set(scores) != set(LABELS):
        raise ValueError("Expected scores for exactly A, B, C, D")
    if target_label not in LABELS:
        raise ValueError(f"Unknown target label: {target_label!r}")
    values = {label: float(scores[label]) for label in LABELS}
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("Target-margin scores must be finite")
    return values[target_label] - max(
        value for label, value in values.items() if label != target_label
    )


def prepare_patch_pair_ledger(
    readout_ledger: pd.DataFrame,
    *,
    split: Literal["validation", "test"],
    stable_control_count: int = 96,
    label_binding_control_count: int = 96,
    seed: str = "decision-binding-v1",
) -> pd.DataFrame:
    if split not in {"validation", "test"}:
        raise ValueError("Patch-pair split must be validation or test")
    if stable_control_count < 0 or label_binding_control_count < 0:
        raise ValueError("Control counts must be non-negative")
    required = {
        "readout_work_key",
        "work_key",
        "item_id",
        "subject",
        "split",
        "wrapper_name",
        "manipulation",
        "variant",
        "prompt",
        "prompt_sha256",
        "labels_by_position",
        "content_ids_by_position",
        "correct_content_id",
        "raw_predicted_label",
        "winner_content_id",
        "text_identity_ambiguous",
        "raw_tie",
        *_SCORE_COLUMNS,
    }
    missing = required - set(readout_ledger.columns)
    if missing:
        raise ValueError(f"Readout ledger is missing patch columns: {sorted(missing)}")
    if readout_ledger["readout_work_key"].duplicated().any():
        raise ValueError("Readout ledger contains duplicate readout work keys")
    source = readout_ledger[readout_ledger["split"].astype(str) == split].copy()
    if source.empty:
        raise ValueError(f"Readout ledger has no {split} rows")

    records: list[dict[str, object]] = []
    block_columns = ["item_id", "wrapper_name", "split"]
    for block_key, block in source.groupby(block_columns, sort=False):
        baseline = block[block["manipulation"] == "controlled_baseline"]
        if len(baseline) != 1:
            raise ValueError(f"Patch block {block_key} requires exactly one baseline")
        donor = baseline.iloc[0]
        variants = block[block["manipulation"].isin(["position_only", "label_only"])]
        for _, receiver in variants.iterrows():
            ambiguous_content = bool(donor["text_identity_ambiguous"]) or bool(
                receiver["text_identity_ambiguous"]
            )
            targets = None if ambiguous_content else patch_target_labels(donor, receiver)
            raw_tie = bool(donor["raw_tie"]) or bool(receiver["raw_tie"])
            content_stable = int(donor["winner_content_id"]) == int(
                receiver["winner_content_id"]
            )
            label_remapped = str(donor["raw_predicted_label"]) != str(
                receiver["raw_predicted_label"]
            )
            if ambiguous_content:
                candidate_kind = "not_applicable_ambiguous_content"
            elif raw_tie:
                candidate_kind = "not_applicable_raw_tie"
            elif not content_stable:
                candidate_kind = "answer_conflict"
            elif str(receiver["manipulation"]) == "label_only" and label_remapped:
                candidate_kind = "label_binding_candidate"
            else:
                candidate_kind = "stable_candidate"
            record: dict[str, object] = {
                "pair_work_key": f"pair|{donor['work_key']}|{receiver['work_key']}",
                "item_id": donor["item_id"],
                "subject": donor["subject"],
                "split": split,
                "wrapper_name": donor["wrapper_name"],
                "manipulation": receiver["manipulation"],
                "variant": int(receiver["variant"]),
                "pair_kind": candidate_kind,
                "selected_for_patching": candidate_kind == "answer_conflict",
                "donor_work_key": donor["work_key"],
                "receiver_work_key": receiver["work_key"],
                "donor_prompt": donor["prompt"],
                "receiver_prompt": receiver["prompt"],
                "donor_prompt_sha256": donor["prompt_sha256"],
                "receiver_prompt_sha256": receiver["prompt_sha256"],
                "donor_labels_by_position": donor["labels_by_position"],
                "receiver_labels_by_position": receiver["labels_by_position"],
                "donor_content_ids_by_position": donor["content_ids_by_position"],
                "receiver_content_ids_by_position": receiver["content_ids_by_position"],
                "donor_raw_predicted_label": donor["raw_predicted_label"],
                "receiver_raw_predicted_label": receiver["raw_predicted_label"],
                "donor_raw_predicted_content_id": int(donor["winner_content_id"]),
                "receiver_raw_predicted_content_id": int(receiver["winner_content_id"]),
                "correct_content_id": int(donor["correct_content_id"]),
                "receiver_content_target_label": None
                if targets is None
                else targets.receiver_content_label,
                "donor_symbol_target_label": None if targets is None else targets.donor_symbol_label,
                "content_identity_evaluable": not ambiguous_content,
                "raw_tie": raw_tie,
            }
            for score_column in _SCORE_COLUMNS:
                record[f"donor_{score_column}"] = float(donor[score_column])
                record[f"receiver_{score_column}"] = float(receiver[score_column])
            records.append(record)
    pairs = pd.DataFrame(records)
    if pairs.empty:
        raise ValueError(f"Readout ledger has no patchable variants for {split}")
    if pairs["pair_work_key"].duplicated().any():
        raise ValueError("Patch-pair ledger produced duplicate work keys")

    def choose(candidate_kind: str, selected_kind: str, count: int) -> None:
        candidates = pairs.index[pairs["pair_kind"] == candidate_kind].tolist()
        ranked = sorted(
            candidates,
            key=lambda index: _stable_digest(seed, pairs.at[index, "pair_work_key"]),
        )
        chosen = set(ranked[:count])
        for index in candidates:
            if index in chosen:
                pairs.at[index, "pair_kind"] = selected_kind
                pairs.at[index, "selected_for_patching"] = True
            else:
                pairs.at[index, "pair_kind"] = f"not_selected_{selected_kind.removesuffix('_control')}"

    choose("label_binding_candidate", "label_binding_control", label_binding_control_count)
    choose("stable_candidate", "stable_control", stable_control_count)
    return pairs.sort_values("pair_work_key", kind="mergesort").reset_index(drop=True)


def run_patch_pair(
    model,
    tokenizer,
    pair: Mapping[str, object],
    banks: Mapping[str, ProbeBank],
    frozen_selection: Mapping[str, Mapping[str, object]],
    *,
    device=None,
) -> pd.DataFrame:
    if not bool(pair.get("selected_for_patching")):
        raise ValueError("Patch pair is not selected for patching")
    for mechanism in ("content", "label"):
        if mechanism not in banks:
            raise ValueError(f"Probe banks are missing {mechanism}")
        if banks[mechanism].target_name != mechanism:
            raise ValueError(
                f"{mechanism} probe bank carries target {banks[mechanism].target_name!r}"
            )
    donor_prompt = str(pair["donor_prompt"])
    receiver_prompt = str(pair["receiver_prompt"])
    for prefix, prompt in (("donor", donor_prompt), ("receiver", receiver_prompt)):
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if str(pair[f"{prefix}_prompt_sha256"]) != digest:
            raise ValueError(f"{prefix} prompt checksum mismatch")
    donor = {
        "raw_predicted_label": pair["donor_raw_predicted_label"],
        "labels_by_position": pair["donor_labels_by_position"],
        "content_ids_by_position": pair["donor_content_ids_by_position"],
    }
    receiver = {
        "labels_by_position": pair["receiver_labels_by_position"],
        "content_ids_by_position": pair["receiver_content_ids_by_position"],
    }
    targets = patch_target_labels(donor, receiver)
    if (
        targets.receiver_content_label != str(pair["receiver_content_target_label"])
        or targets.donor_symbol_label != str(pair["donor_symbol_target_label"])
    ):
        raise ValueError("Stored patch targets disagree with the structured coordinate mapping")
    bias_scores = {
        label: float(pair[f"receiver_bias_score_{label}"])
        for label in LABELS
    }
    captured = capture_layer_readouts(
        model,
        tokenizer,
        [donor_prompt, receiver_prompt],
        batch_size=2,
        device=device,
    )
    if any(
        captured.activations.shape[1] != banks[mechanism].weights.shape[0]
        for mechanism in ("content", "label")
    ):
        raise ValueError("Captured model layers do not match the frozen probe banks")
    unpatched = _scored_label_result(
        {
            label: float(captured.raw_log_probs[1, index])
            for index, label in enumerate(LABELS)
        },
        bias_scores,
        targets.receiver_content_label,
    )
    receiver_checkpoints = resolve_readout_checkpoints(tokenizer, receiver_prompt)
    checkpoint_positions = {
        "format_end": receiver_checkpoints.format_end,
        "answer_prefix_end": receiver_checkpoints.answer_prefix_end,
    }
    rows: list[dict[str, object]] = []

    def result_row(
        *,
        mechanism: str,
        checkpoint: str,
        layer: int,
        condition: str,
        scored: Mapping[str, object],
        intervention_norm: float,
        probe_rank: int,
    ) -> dict[str, object]:
        raw_scores = {label: float(scored[f"raw_score_{label}"]) for label in LABELS}
        cal_scores = {label: float(scored[f"cal_score_{label}"]) for label in LABELS}
        record: dict[str, object] = {
            "pair_work_key": pair["pair_work_key"],
            "item_id": pair["item_id"],
            "subject": pair["subject"],
            "split": pair["split"],
            "wrapper_name": pair["wrapper_name"],
            "manipulation": pair["manipulation"],
            "variant": int(pair["variant"]),
            "pair_kind": pair["pair_kind"],
            "donor_work_key": pair["donor_work_key"],
            "receiver_work_key": pair["receiver_work_key"],
            "mechanism": mechanism,
            "checkpoint": checkpoint,
            "layer": int(layer),
            "condition": condition,
            "probe_rank": int(probe_rank),
            "intervention_norm": float(intervention_norm),
            "content_target_label": targets.receiver_content_label,
            "symbol_target_label": targets.donor_symbol_label,
            "content_target_margin": target_margin(raw_scores, targets.receiver_content_label),
            "symbol_target_margin": target_margin(raw_scores, targets.donor_symbol_label),
            "cal_content_target_margin": target_margin(cal_scores, targets.receiver_content_label),
            "cal_symbol_target_margin": target_margin(cal_scores, targets.donor_symbol_label),
            "raw_predicted_label": scored["raw_pred_label"],
            "cal_predicted_label": scored["cal_pred_label"],
        }
        for label in LABELS:
            record[f"raw_score_{label}"] = raw_scores[label]
            record[f"cal_score_{label}"] = cal_scores[label]
        return record

    for mechanism in ("content", "label"):
        bank = banks[mechanism]
        if mechanism not in frozen_selection:
            raise ValueError(f"Frozen selection is missing {mechanism}")
        specification = frozen_selection[mechanism]
        if not bool(specification.get("usable")):
            raise ValueError(f"Frozen {mechanism} readout is not usable for causal patching")
        checkpoint = str(specification["checkpoint"])
        if checkpoint not in checkpoint_positions:
            raise ValueError(f"Unknown frozen checkpoint: {checkpoint!r}")
        layers = sorted({int(layer) for layer in specification["patch_layers"]})
        if not layers or min(layers) < 0 or max(layers) >= captured.activations.shape[1]:
            raise ValueError(f"Frozen {mechanism} patch layers are outside the model")
        replacements: dict[str, dict[int, dict[int, torch.Tensor]]] = {
            condition: {} for condition in ("probe", "full", "random", "identity")
        }
        norms: dict[tuple[int, str], float] = {}
        ranks: dict[int, int] = {}
        checkpoint_index = _CHECKPOINT_INDEX[checkpoint]
        for layer in layers:
            basis = probe_subspace_basis(bank, layer=layer, checkpoint=checkpoint)
            ranks[layer] = basis.shape[1]
            receiver_vector = captured.activations[1, layer, checkpoint_index]
            donor_vector = captured.activations[0, layer, checkpoint_index]
            vectors = build_patch_vectors(
                receiver_vector,
                donor_vector,
                basis,
                seed=f"{pair['pair_work_key']}|{mechanism}|{layer}|{checkpoint}",
            )
            for condition in replacements:
                vector = getattr(vectors, condition)
                replacements[condition][layer] = {checkpoint_positions[checkpoint]: vector}
                norms[(layer, condition)] = float(torch.linalg.vector_norm(vector - receiver_vector))
            rows.append(
                result_row(
                    mechanism=mechanism,
                    checkpoint=checkpoint,
                    layer=layer,
                    condition="unpatched",
                    scored=unpatched,
                    intervention_norm=0.0,
                    probe_rank=ranks[layer],
                )
            )
        for condition, replacements_by_layer in replacements.items():
            scored_layers = score_layers_with_position_replacements(
                model,
                tokenizer,
                receiver_prompt,
                correct_label=targets.receiver_content_label,
                bias_scores=bias_scores,
                replacements_by_layer=replacements_by_layer,
                device=device,
            )
            for layer in layers:
                rows.append(
                    result_row(
                        mechanism=mechanism,
                        checkpoint=checkpoint,
                        layer=layer,
                        condition=condition,
                        scored=scored_layers[layer],
                        intervention_norm=norms[(layer, condition)],
                        probe_rank=ranks[layer],
                    )
                )
    return pd.DataFrame(rows).sort_values(
        ["mechanism", "layer", "condition"], kind="mergesort"
    ).reset_index(drop=True)


def fit_probe_bank(
    activations: np.ndarray | torch.Tensor,
    targets: Sequence[int],
    *,
    c: float = 1e-2,
    max_iter: int = 5000,
    target_name: str = "content",
    row_mask: Sequence[bool] | None = None,
    class_weight: Literal["balanced"] | None = None,
) -> ProbeBank:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    values = np.asarray(activations)
    labels = np.asarray(targets, dtype=np.int64)
    if values.ndim != 4 or len(values) != len(labels):
        raise ValueError("Probe activations must have shape [rows, layers, checkpoints, hidden]")
    selected = (
        np.ones(len(values), dtype=bool)
        if row_mask is None
        else np.asarray(row_mask, dtype=bool)
    )
    if selected.shape != (len(values),) or not selected.any():
        raise ValueError("Probe row mask must select at least one activation row")
    selected_labels = labels[selected]
    if sorted(np.unique(selected_labels).tolist()) != list(range(4)):
        raise ValueError("Probe fitting requires all four raw winner classes")
    layers, checkpoints, hidden = values.shape[1:]
    weights = np.empty((layers, checkpoints, 4, hidden), dtype=np.float64)
    intercepts = np.empty((layers, checkpoints, 4), dtype=np.float64)
    means = np.empty((layers, checkpoints, hidden), dtype=np.float64)
    for layer in range(layers):
        for checkpoint in range(checkpoints):
            slice_values = np.asarray(values[selected, layer, checkpoint], dtype=np.float64)
            means[layer, checkpoint] = slice_values.mean(axis=0)
            estimator = LogisticRegression(
                C=float(c), solver="lbfgs", max_iter=int(max_iter), class_weight=class_weight
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                estimator.fit(slice_values - means[layer, checkpoint], selected_labels)
            if any(issubclass(warning.category, ConvergenceWarning) for warning in caught):
                raise RuntimeError(f"Probe did not converge at layer {layer}, checkpoint {checkpoint}")
            weights[layer, checkpoint] = estimator.coef_
            intercepts[layer, checkpoint] = estimator.intercept_
    return ProbeBank(
        weights,
        intercepts,
        means,
        np.arange(4, dtype=np.int64),
        float(c),
        str(target_name),
    )


def fit_coordinate_probe_banks(
    activations: np.ndarray | torch.Tensor,
    ledger: pd.DataFrame,
    *,
    c: float = 1e-2,
    max_iter: int = 5000,
) -> dict[str, ProbeBank]:
    """Fit independent content, position, label, and legacy content readers."""
    values = np.asarray(activations)
    if len(values) != len(ledger):
        raise ValueError("Ledger and activation row counts differ")
    required = {
        "readout_role",
        "manipulation",
        "winner_content_id",
        "winner_position",
        "winner_label_index",
        "winner_unique",
    }
    missing = required - set(ledger.columns)
    if missing:
        raise ValueError(f"Probe ledger is missing columns: {sorted(missing)}")
    training = ledger["readout_role"].astype(str).eq("probe_train").to_numpy()
    if not training.any():
        raise ValueError("Probe ledger contains no probe_train rows")
    banks: dict[str, ProbeBank] = {}
    for name, target_column in (
        ("content", "winner_content_id"),
        ("position", "winner_position"),
        ("label", "winner_label_index"),
    ):
        evaluable_column = f"{name}_evaluable"
        evaluable = (
            ledger[evaluable_column].astype(bool).to_numpy()
            if evaluable_column in ledger
            else ledger["winner_unique"].astype(bool).to_numpy()
        )
        mask = training & evaluable
        banks[name] = fit_probe_bank(
            values,
            ledger[target_column].to_numpy(dtype=np.int64),
            c=c,
            max_iter=max_iter,
            target_name=name,
            row_mask=mask,
            class_weight="balanced",
        )
    legacy_mask = (
        training
        & ledger["winner_unique"].astype(bool).to_numpy()
        & ledger["manipulation"].astype(str).eq("controlled_baseline").to_numpy()
    )
    if "content_evaluable" in ledger:
        legacy_mask &= ledger["content_evaluable"].astype(bool).to_numpy()
    banks["legacy_content"] = fit_probe_bank(
        values,
        ledger["winner_content_id"].to_numpy(dtype=np.int64),
        c=c,
        max_iter=max_iter,
        target_name="legacy_content",
        row_mask=legacy_mask,
        class_weight=None,
    )
    return banks


def save_probe_bank(path: str | Path, bank: ProbeBank) -> None:
    np.savez_compressed(
        Path(path),
        weights=bank.weights,
        intercepts=bank.intercepts,
        means=bank.means,
        classes=bank.classes,
        c=np.asarray(bank.c),
        target_name=np.asarray(bank.target_name),
    )


def load_probe_bank(path: str | Path) -> ProbeBank:
    with np.load(Path(path), allow_pickle=False) as payload:
        return ProbeBank(
            payload["weights"],
            payload["intercepts"],
            payload["means"],
            payload["classes"],
            float(payload["c"]),
            str(payload["target_name"].item()) if "target_name" in payload else "content",
        )


def evaluate_probe_banks(
    banks: Mapping[str, ProbeBank],
    activations: np.ndarray | torch.Tensor,
    ledger: pd.DataFrame,
) -> pd.DataFrame:
    """Score all coordinate readers without duplicating structural output rows."""
    expected = {"content", "position", "label", "legacy_content"}
    if set(banks) != expected:
        raise ValueError(f"Probe banks must be exactly {sorted(expected)}")
    for name, bank in banks.items():
        if bank.target_name != name:
            raise ValueError(f"Probe bank {name!r} carries target {bank.target_name!r}")
    if len(ledger) != len(activations):
        raise ValueError("Ledger and activation row counts differ")
    required = {
        "readout_work_key", "item_id", "readout_role", "manipulation",
        "winner_unique", "winner_content_id", "winner_position", "winner_label_index",
    }
    missing = required - set(ledger.columns)
    if missing:
        raise ValueError(f"Readout ledger is missing columns: {sorted(missing)}")
    compact_columns = [
        column for column in (
            "readout_work_key", "work_key", "item_id", "subject", "split",
            "wrapper_name", "readout_role", "manipulation", "variant",
            "winner_unique", "winner_content_id", "winner_position",
            "winner_label_index", "text_identity_ambiguous", "content_evaluable",
            "position_evaluable", "label_evaluable",
        ) if column in ledger
    ]
    compact = ledger.reset_index(drop=True)[compact_columns].copy()
    unique = compact["winner_unique"].astype(bool).to_numpy()
    ambiguous = (
        compact.get("text_identity_ambiguous", pd.Series(False, index=compact.index))
        .astype(bool).to_numpy()
    )
    default_masks = {
        "content": unique & ~ambiguous,
        "position": unique,
        "label": unique,
    }
    masks = {
        name: (
            compact[f"{name}_evaluable"].astype(bool).to_numpy()
            if f"{name}_evaluable" in compact
            else default_masks[name]
        )
        for name in ("content", "position", "label")
    }
    targets = {
        "content": compact["winner_content_id"].to_numpy(dtype=int),
        "position": compact["winner_position"].to_numpy(dtype=int),
        "label": compact["winner_label_index"].to_numpy(dtype=int),
    }
    values = np.asarray(activations)
    reference = banks["content"]
    shape = (len(values), *reference.weights.shape[:2], 4)
    if any(
        bank.weights.shape[:2] != reference.weights.shape[:2]
        or bank.weights.shape[-1] != values.shape[-1]
        for bank in banks.values()
    ):
        raise ValueError("Probe bank shapes differ from the activations")
    frames: list[pd.DataFrame] = []
    for layer in range(shape[1]):
        for checkpoint_index, checkpoint in enumerate(("format_end", "answer_prefix_end")):
            frame = compact.copy()
            frame["layer"] = layer
            frame["checkpoint"] = checkpoint
            row_indices = np.arange(len(frame))
            site_values = np.asarray(values[:, layer, checkpoint_index], dtype=np.float64)
            for name, bank in banks.items():
                scores = (site_values - bank.means[layer, checkpoint_index]) @ bank.weights[
                    layer, checkpoint_index
                ].T
                scores += bank.intercepts[layer, checkpoint_index]
                scores -= scores.max(axis=1, keepdims=True)
                scores -= np.log(np.exp(scores).sum(axis=1, keepdims=True))
                prediction = scores.argmax(axis=1)
                for class_id in range(4):
                    frame[f"{name}_log_prob_{class_id}"] = scores[:, class_id]
                if name == "legacy_content":
                    frame[f"{name}_pred_class"] = prediction
                    continue
                mask = masks[name]
                target = targets[name]
                frame[f"{name}_evaluable"] = mask
                frame[f"{name}_pred_class"] = pd.array(
                    np.where(mask, prediction, np.nan), dtype="Int64"
                )
                frame[f"{name}_log_prob"] = np.where(
                    mask, scores[row_indices, target], np.nan
                )
                frame[f"{name}_correct"] = pd.array(
                    np.where(mask, prediction == target, np.nan), dtype="boolean"
                )
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def evaluate_probe_bank(
    bank: ProbeBank,
    activations: np.ndarray | torch.Tensor,
    ledger: pd.DataFrame,
) -> pd.DataFrame:
    if len(ledger) != len(activations):
        raise ValueError("Ledger and activation row counts differ")
    log_probs = bank.log_probabilities(activations)
    compact_columns = [
        column
        for column in (
            "readout_work_key",
            "work_key",
            "item_id",
            "subject",
            "split",
            "wrapper_name",
            "manipulation",
            "variant",
            "winner_unique",
            "winner_content_id",
            "winner_position",
            "winner_label_index",
            "text_identity_ambiguous",
        )
        if column in ledger.columns
    ]
    compact_ledger = ledger.reset_index(drop=True)[compact_columns].copy()
    if "text_identity_ambiguous" not in compact_ledger:
        compact_ledger["text_identity_ambiguous"] = False
    frames: list[pd.DataFrame] = []
    checkpoint_names = ("format_end", "answer_prefix_end")
    for layer in range(log_probs.shape[1]):
        for checkpoint, checkpoint_name in enumerate(checkpoint_names):
            frame = compact_ledger.copy()
            scores = log_probs[:, layer, checkpoint]
            frame["layer"] = layer
            frame["checkpoint"] = checkpoint_name
            raw_probe_prediction = scores.argmax(axis=1)
            evaluable = (
                frame["winner_unique"].astype(bool)
                & ~frame["text_identity_ambiguous"].astype(bool)
            )
            frame["probe_pred_class"] = pd.array(
                np.where(evaluable, raw_probe_prediction, np.nan), dtype="Int64"
            )
            for class_id in range(4):
                frame[f"probe_log_prob_{class_id}"] = scores[:, class_id]
            targets = {
                "content": frame["winner_content_id"].to_numpy(dtype=int),
                "position": frame["winner_position"].to_numpy(dtype=int),
                "label": frame["winner_label_index"].to_numpy(dtype=int),
            }
            for coordinate, target in targets.items():
                values = scores[np.arange(len(frame)), target]
                correct = raw_probe_prediction == target
                frame[f"{coordinate}_evaluable"] = evaluable
                frame[f"{coordinate}_log_prob"] = np.where(evaluable, values, np.nan)
                frame[f"{coordinate}_correct"] = np.where(evaluable, correct, np.nan)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _item_confusion(frame: pd.DataFrame, target: str, prediction: str) -> np.ndarray:
    columns = ["item_id", target, prediction]
    if "wrapper_name" in frame:
        columns.append("wrapper_name")
    scored = frame[columns].dropna().copy()
    if scored.empty:
        raise ValueError("No evaluable rows for macro accuracy")
    if "wrapper_name" not in scored:
        scored["wrapper_name"] = "__single_wrapper__"
    scored[target] = scored[target].astype(int)
    scored[prediction] = scored[prediction].astype(int)
    counts = scored.groupby(
        ["item_id", "wrapper_name", target, prediction], sort=False
    ).size()
    distributions = counts / counts.groupby(
        level=["item_id", "wrapper_name", target], sort=False
    ).transform("sum")
    per_item = distributions.groupby(level=["item_id", target, prediction], sort=False).mean()
    item_ids = sorted(scored["item_id"].astype(str).unique())
    item_index = {item_id: index for index, item_id in enumerate(item_ids)}
    confusion = np.full((len(item_ids), 4, 4), np.nan, dtype=float)
    for (item_id, target_class, predicted_class), value in per_item.items():
        row = item_index[str(item_id)]
        target_index = int(target_class)
        if np.isnan(confusion[row, target_index]).all():
            confusion[row, target_index] = 0.0
        confusion[row, target_index, int(predicted_class)] = float(value)
    return confusion


def _macro_accuracy(frame: pd.DataFrame, target: str, prediction: str) -> float:
    per_item_class = np.diagonal(
        _item_confusion(frame, target, prediction), axis1=1, axis2=2
    )
    return float(np.nanmean(np.nanmean(per_item_class, axis=0)))


def _selection_gate(
    frame: pd.DataFrame,
    *,
    target: str,
    prediction: str,
    bootstrap_samples: int,
    permutation_samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, bool]:
    frame = frame.dropna(subset=[prediction]).copy()
    observed = _macro_accuracy(frame, target, prediction)
    confusion = _item_confusion(frame, target, prediction)
    n_items = len(confusion)
    per_item_class = np.diagonal(confusion, axis1=1, axis2=2)
    draws = rng.integers(0, n_items, size=(bootstrap_samples, n_items))
    sampled = per_item_class[draws]
    class_counts = np.sum(~np.isnan(sampled), axis=1)
    class_means = np.divide(
        np.nansum(sampled, axis=1),
        class_counts,
        out=np.full(class_counts.shape, np.nan, dtype=float),
        where=class_counts > 0,
    )
    boot = np.nanmean(class_means, axis=1)
    null: list[float] = []
    for start in range(0, permutation_samples, 100):
        size = min(100, permutation_samples - start)
        offsets = rng.integers(0, 4, size=(size, n_items))
        item_rows = np.arange(n_items)
        for offset in offsets:
            class_scores = []
            for new_class in range(4):
                original_class = (new_class - offset) % 4
                numerator = confusion[item_rows, original_class, new_class]
                valid = ~np.isnan(numerator)
                if valid.any():
                    class_scores.append(float(np.mean(numerator[valid])))
            null.append(float(np.mean(class_scores)))
    lower = float(np.quantile(boot, 0.025))
    null_99 = float(np.quantile(null, 0.99))
    return observed, lower, null_99, bool(observed >= 0.50 and lower > 0.25 and observed > null_99)


_TARGET_COLUMNS = {
    "content": "winner_content_id",
    "position": "winner_position",
    "label": "winner_label_index",
}
_SELECTIVITY_CONTRASTS = {
    "content": (("position_only", "position"), ("label_only", "label")),
    "position": (("position_only", "content"),),
    "label": (("label_only", "content"),),
}


def _coordinate_log_probability(
    frame: pd.DataFrame,
    *,
    reader: str,
    coordinate: str,
) -> np.ndarray:
    targets = frame[_TARGET_COLUMNS[coordinate]].to_numpy(dtype=int)
    scores = frame[[f"{reader}_log_prob_{class_id}" for class_id in range(4)]].to_numpy(float)
    return scores[np.arange(len(frame)), targets]


def _selectivity_metrics(
    frame: pd.DataFrame,
    *,
    reader: str,
    bootstrap_samples: int,
    seed: int,
) -> tuple[float, float, dict[str, dict[str, float]]]:
    contrasts: dict[str, dict[str, float]] = {}
    for manipulation, nuisance in _SELECTIVITY_CONTRASTS[reader]:
        arm = frame[frame["manipulation"].astype(str) == manipulation].copy()
        arm = arm[arm[f"{reader}_evaluable"].astype(bool)]
        if arm.empty:
            raise ValueError(f"No evaluable {manipulation} rows for {reader} selectivity")
        differences = _coordinate_log_probability(
            arm, reader=reader, coordinate=reader
        ) - _coordinate_log_probability(arm, reader=reader, coordinate=nuisance)
        difference_frame = pd.DataFrame(
            {
                "item_id": arm["item_id"].astype(str).to_numpy(),
                "wrapper_name": (
                    arm["wrapper_name"].astype(str).to_numpy()
                    if "wrapper_name" in arm
                    else "__single_wrapper__"
                ),
                "difference": differences,
            }
        )
        per_item_wrapper = difference_frame.groupby(
            ["item_id", "wrapper_name"], sort=False
        )["difference"].mean()
        per_item = per_item_wrapper.groupby(level="item_id", sort=False).mean().to_numpy(float)
        rng = np.random.default_rng(
            int(_stable_digest(str(seed), reader, manipulation, nuisance)[:16], 16)
        )
        draws = per_item[
            rng.integers(0, len(per_item), size=(bootstrap_samples, len(per_item)))
        ].mean(axis=1)
        contrasts[f"{manipulation}:{nuisance}"] = {
            "mean": float(per_item.mean()),
            "bootstrap_lower_95": float(np.quantile(draws, 0.025)),
        }
    return (
        min(value["mean"] for value in contrasts.values()),
        min(value["bootstrap_lower_95"] for value in contrasts.values()),
        contrasts,
    )


def _accuracy_rows(frame: pd.DataFrame, reader: str) -> pd.DataFrame:
    allowed = {
        "content": {"position_only", "label_only"},
        "position": {"position_only"},
        "label": {"label_only"},
    }[reader]
    return frame[
        frame["manipulation"].astype(str).isin(allowed)
        & frame[f"{reader}_evaluable"].astype(bool)
    ]


def select_readout_layers(
    evaluated: pd.DataFrame,
    *,
    bootstrap_samples: int = 5000,
    permutation_samples: int = 1000,
    seed: int = 0,
) -> dict[str, dict[str, object]]:
    required = {
        "item_id", "readout_role", "manipulation", "layer", "checkpoint",
        "winner_content_id", "winner_position", "winner_label_index",
        *{
            f"{reader}_{suffix}"
            for reader in ("content", "position", "label")
            for suffix in ("pred_class", "evaluable")
        },
        *{
            f"{reader}_log_prob_{class_id}"
            for reader in ("content", "position", "label")
            for class_id in range(4)
        },
    }
    missing = required - set(evaluated.columns)
    if missing or bootstrap_samples <= 0 or permutation_samples <= 0:
        raise ValueError(f"Invalid layer-selection input; missing={sorted(missing)}")
    selection = evaluated[evaluated["readout_role"].astype(str) == "layer_select"].copy()
    gate = evaluated[evaluated["readout_role"].astype(str) == "reader_gate"].copy()
    if selection.empty or gate.empty:
        raise ValueError("Layer selection and reader gate must both contain rows")
    result: dict[str, dict[str, object]] = {}
    for name in ("content", "position", "label"):
        target = _TARGET_COLUMNS[name]
        candidates: list[dict[str, object]] = []
        for (layer, checkpoint), group in selection.groupby(
            ["layer", "checkpoint"], sort=True
        ):
            group = group[group[f"{name}_evaluable"].astype(bool)]
            selectivity, selectivity_lower, contrasts = _selectivity_metrics(
                group,
                reader=name,
                bootstrap_samples=bootstrap_samples,
                seed=int(_stable_digest(str(seed), name, layer, checkpoint)[:16], 16),
            )
            accuracy_group = _accuracy_rows(group, name)
            accuracy = _macro_accuracy(accuracy_group, target, f"{name}_pred_class")
            candidates.append(
                {
                    "layer": int(layer),
                    "checkpoint": str(checkpoint),
                    "selectivity": selectivity,
                    "selectivity_bootstrap_lower_95": selectivity_lower,
                    "contrasts": contrasts,
                    "macro_accuracy": accuracy,
                }
            )
        if not candidates:
            raise ValueError(f"No usable rows for {name} layer selection")
        chosen = max(
            candidates,
            key=lambda value: (
                value["selectivity"],
                value["macro_accuracy"],
                -value["layer"],
                value["checkpoint"] == "format_end",
            ),
        )
        selected_layer = int(chosen["layer"])
        checkpoint = str(chosen["checkpoint"])
        gated = gate[
            (gate["layer"].astype(int) == selected_layer)
            & (gate["checkpoint"].astype(str) == checkpoint)
            & gate[f"{name}_evaluable"].astype(bool)
        ]
        accuracy_gate = _accuracy_rows(gated, name)
        rng = np.random.default_rng(
            int(_stable_digest(str(seed), "gate", name, selected_layer, checkpoint)[:16], 16)
        )
        observed, lower, null_99, accuracy_usable = _selection_gate(
            accuracy_gate,
            target=target,
            prediction=f"{name}_pred_class",
            bootstrap_samples=bootstrap_samples,
            permutation_samples=permutation_samples,
            rng=rng,
        )
        gate_selectivity, gate_selectivity_lower, gate_contrasts = _selectivity_metrics(
            gated,
            reader=name,
            bootstrap_samples=bootstrap_samples,
            seed=int(_stable_digest(str(seed), "gate-selectivity", name)[:16], 16),
        )
        max_layer = int(evaluated["layer"].max())
        patch_layers = sorted({0, *range(max(0, selected_layer - 1), min(max_layer, selected_layer + 1) + 1)})
        result[name] = {
            "selected_layer": selected_layer,
            "checkpoint": checkpoint,
            "patch_layers": patch_layers,
            "selectivity": gate_selectivity,
            "selectivity_bootstrap_lower_95": gate_selectivity_lower,
            "macro_accuracy": observed,
            "bootstrap_lower_95": lower,
            "permutation_99": null_99,
            "selection_metrics": {
                **chosen,
                "items": int(selection["item_id"].nunique()),
            },
            "gate_metrics": {
                "items": int(gated["item_id"].nunique()),
                "macro_accuracy": observed,
                "bootstrap_lower_95": lower,
                "permutation_99": null_99,
                "selectivity": gate_selectivity,
                "selectivity_bootstrap_lower_95": gate_selectivity_lower,
                "contrasts": gate_contrasts,
            },
            "usable": bool(accuracy_usable and gate_selectivity_lower > 0.0),
        }
    return result
