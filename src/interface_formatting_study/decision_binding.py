from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import torch

from .hooks import find_transformer_blocks
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
    frames: list[pd.DataFrame] = []
    checkpoint_names = ("format_end", "answer_prefix_end")
    for layer in range(log_probs.shape[1]):
        for checkpoint, checkpoint_name in enumerate(checkpoint_names):
            frame = ledger.reset_index(drop=True).copy()
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
