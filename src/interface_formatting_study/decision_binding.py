from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .hooks import find_transformer_blocks
from .interventions import score_prompt_with_optional_edit
from .patching import score_layers_with_position_replacements
from .scoring import single_token_label_ids, tokenize_text


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
            & (normalized["manipulation"] == "controlled_baseline")
        ].copy()
        validation = normalized[normalized["split"] == "validation"].copy()
        train["readout_role"] = "probe_train"
        validation["readout_role"] = "layer_select"
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


def capture_layer_readouts(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    batch_size: int,
    max_batch_tokens: int | None = None,
    device=None,
) -> CapturedReadouts:
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

    blocks = find_transformer_blocks(model)
    output_activations: list[torch.Tensor | None] = [None] * len(prompts)
    output_scores: list[torch.Tensor | None] = [None] * len(prompts)
    device_obj = device or next(model.parameters()).device
    pad_id = int(getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0) or 0)
    padded_tokens = 0
    for indices in groups:
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
            with torch.inference_mode():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
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
        for row, index in enumerate(indices):
            output_activations[index] = layer_tensor[row]
            output_scores[index] = scores[row]
        padded_tokens += len(indices) * max_len
    if any(value is None for value in (*output_activations, *output_scores)):
        raise RuntimeError("Batched readout did not produce every requested row")
    return CapturedReadouts(
        activations=torch.stack([value for value in output_activations if value is not None]),
        raw_log_probs=torch.stack([value for value in output_scores if value is not None]),
        batches=len(groups),
        actual_tokens=sum(map(len, encoded)),
        padded_tokens=padded_tokens,
    )


@dataclass(frozen=True)
class ProbeBank:
    weights: np.ndarray
    intercepts: np.ndarray
    means: np.ndarray
    classes: np.ndarray
    c: float

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
            targets = patch_target_labels(donor, receiver)
            raw_tie = bool(donor["raw_tie"]) or bool(receiver["raw_tie"])
            content_stable = int(donor["winner_content_id"]) == int(
                receiver["winner_content_id"]
            )
            label_remapped = str(donor["raw_predicted_label"]) != str(
                receiver["raw_predicted_label"]
            )
            if raw_tie:
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
                "receiver_content_target_label": targets.receiver_content_label,
                "donor_symbol_target_label": targets.donor_symbol_label,
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
    bank: ProbeBank,
    frozen_selection: Mapping[str, Mapping[str, object]],
    *,
    device=None,
) -> pd.DataFrame:
    if not bool(pair.get("selected_for_patching")):
        raise ValueError("Patch pair is not selected for patching")
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
    if captured.activations.shape[1] != bank.weights.shape[0]:
        raise ValueError("Captured model layers do not match the frozen probe bank")
    unpatched = score_prompt_with_optional_edit(
        model,
        tokenizer,
        receiver_prompt,
        correct_label=targets.receiver_content_label,
        bias_scores=bias_scores,
        device=device,
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
) -> ProbeBank:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    values = np.asarray(activations, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    if values.ndim != 4 or len(values) != len(labels):
        raise ValueError("Probe activations must have shape [rows, layers, checkpoints, hidden]")
    if sorted(np.unique(labels).tolist()) != list(range(4)):
        raise ValueError("Probe fitting requires all four raw winner classes")
    layers, checkpoints, hidden = values.shape[1:]
    weights = np.empty((layers, checkpoints, 4, hidden), dtype=np.float64)
    intercepts = np.empty((layers, checkpoints, 4), dtype=np.float64)
    means = values.mean(axis=0)
    for layer in range(layers):
        for checkpoint in range(checkpoints):
            estimator = LogisticRegression(C=float(c), solver="lbfgs", max_iter=int(max_iter))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                estimator.fit(values[:, layer, checkpoint] - means[layer, checkpoint], labels)
            if any(issubclass(warning.category, ConvergenceWarning) for warning in caught):
                raise RuntimeError(f"Probe did not converge at layer {layer}, checkpoint {checkpoint}")
            weights[layer, checkpoint] = estimator.coef_
            intercepts[layer, checkpoint] = estimator.intercept_
    return ProbeBank(weights, intercepts, means, np.arange(4, dtype=np.int64), float(c))


def save_probe_bank(path: str | Path, bank: ProbeBank) -> None:
    np.savez_compressed(
        Path(path),
        weights=bank.weights,
        intercepts=bank.intercepts,
        means=bank.means,
        classes=bank.classes,
        c=np.asarray(bank.c),
    )


def load_probe_bank(path: str | Path) -> ProbeBank:
    with np.load(Path(path), allow_pickle=False) as payload:
        return ProbeBank(
            payload["weights"],
            payload["intercepts"],
            payload["means"],
            payload["classes"],
            float(payload["c"]),
        )


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
    frames: list[pd.DataFrame] = []
    checkpoint_names = ("format_end", "answer_prefix_end")
    for layer in range(log_probs.shape[1]):
        for checkpoint, checkpoint_name in enumerate(checkpoint_names):
            frame = compact_ledger.copy()
            scores = log_probs[:, layer, checkpoint]
            frame["layer"] = layer
            frame["checkpoint"] = checkpoint_name
            frame["probe_pred_class"] = scores.argmax(axis=1)
            for class_id in range(4):
                frame[f"probe_log_prob_{class_id}"] = scores[:, class_id]
            targets = {
                "content": frame["winner_content_id"].to_numpy(dtype=int),
                "position": frame["winner_position"].to_numpy(dtype=int),
                "label": frame["winner_label_index"].to_numpy(dtype=int),
            }
            for coordinate, target in targets.items():
                frame[f"{coordinate}_log_prob"] = scores[np.arange(len(frame)), target]
                frame[f"{coordinate}_correct"] = frame["probe_pred_class"].to_numpy() == target
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _macro_accuracy(frame: pd.DataFrame, target: str) -> float:
    scored = frame[["item_id", target, "probe_pred_class"]].copy()
    scored["correct"] = scored[target].astype(int) == scored["probe_pred_class"].astype(int)
    per_item_class = scored.groupby(["item_id", target], sort=False)["correct"].mean()
    return float(per_item_class.groupby(level=1).mean().mean())


def _selection_gate(
    frame: pd.DataFrame,
    *,
    target: str,
    bootstrap_samples: int,
    permutation_samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, bool]:
    observed = _macro_accuracy(frame, target)
    _, item_index = np.unique(frame["item_id"].astype(str), return_inverse=True)
    targets = frame[target].to_numpy(dtype=int)
    predictions = frame["probe_pred_class"].to_numpy(dtype=int)
    n_items = int(item_index.max()) + 1
    counts = np.zeros((n_items, 4), dtype=float)
    prediction_counts = np.zeros((n_items, 4, 4), dtype=float)
    np.add.at(counts, (item_index, targets), 1)
    np.add.at(prediction_counts, (item_index, targets, predictions), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_item_class = np.where(counts > 0, np.diagonal(prediction_counts, axis1=1, axis2=2) / counts, np.nan)
    draws = rng.integers(0, n_items, size=(bootstrap_samples, n_items))
    boot = np.nanmean(np.nanmean(per_item_class[draws], axis=1), axis=1)
    null: list[float] = []
    for start in range(0, permutation_samples, 100):
        size = min(100, permutation_samples - start)
        offsets = rng.integers(0, 4, size=(size, n_items))
        item_rows = np.arange(n_items)
        for offset in offsets:
            class_scores = []
            for new_class in range(4):
                original_class = (new_class - offset) % 4
                denominator = counts[item_rows, original_class]
                numerator = prediction_counts[item_rows, original_class, new_class]
                valid = denominator > 0
                if valid.any():
                    class_scores.append(float(np.mean(numerator[valid] / denominator[valid])))
            null.append(float(np.mean(class_scores)))
    lower = float(np.quantile(boot, 0.025))
    null_99 = float(np.quantile(null, 0.99))
    return observed, lower, null_99, bool(observed >= 0.50 and lower > 0.25 and observed > null_99)


def select_readout_layers(
    evaluated: pd.DataFrame,
    *,
    bootstrap_samples: int = 5000,
    permutation_samples: int = 1000,
    seed: int = 0,
) -> dict[str, dict[str, object]]:
    required = {
        "item_id", "layer", "checkpoint", "probe_pred_class", "winner_content_id",
        "winner_position", "winner_label_index", "content_log_prob", "position_log_prob",
        "label_log_prob", "winner_unique",
    }
    missing = required - set(evaluated.columns)
    if missing or bootstrap_samples <= 0 or permutation_samples <= 0:
        raise ValueError(f"Invalid layer-selection input; missing={sorted(missing)}")
    candidates = evaluated[evaluated["winner_unique"].astype(bool)].copy()
    if "manipulation" in candidates:
        candidates = candidates[candidates["manipulation"] != "controlled_baseline"]
    rng = np.random.default_rng(seed)
    result: dict[str, dict[str, object]] = {}
    for name, checkpoint, target in (
        ("content", "format_end", "winner_content_id"),
        ("label", "answer_prefix_end", "winner_label_index"),
    ):
        subset = candidates[candidates["checkpoint"] == checkpoint]
        rows = []
        for layer, group in subset.groupby("layer", sort=True):
            scores = {
                coordinate: float(group.groupby("item_id", sort=False)[f"{coordinate}_log_prob"].mean().mean())
                for coordinate in ("content", "position", "label")
            }
            selectivity = scores[name] - max(value for key, value in scores.items() if key != name)
            rows.append((selectivity, int(layer), scores))
        if not rows:
            raise ValueError(f"No usable {checkpoint} rows for layer selection")
        _, selected_layer, scores = max(rows, key=lambda row: (row[0], -row[1]))
        selected = subset[subset["layer"].astype(int) == selected_layer]
        observed, lower, null_99, usable = _selection_gate(
            selected,
            target=target,
            bootstrap_samples=bootstrap_samples,
            permutation_samples=permutation_samples,
            rng=rng,
        )
        max_layer = int(evaluated["layer"].max())
        patch_layers = sorted({0, *range(max(0, selected_layer - 1), min(max_layer, selected_layer + 1) + 1)})
        result[name] = {
            "selected_layer": selected_layer,
            "checkpoint": checkpoint,
            "patch_layers": patch_layers,
            "selectivity": float(max(rows, key=lambda row: (row[0], -row[1]))[0]),
            "coordinate_scores": scores,
            "macro_accuracy": observed,
            "bootstrap_lower_95": lower,
            "permutation_99": null_99,
            "usable": usable,
        }
    return result
