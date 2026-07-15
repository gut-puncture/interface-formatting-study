from __future__ import annotations

import copy
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Mapping

import torch

from .utils import EPSILON, LABELS, validate_label

VARIANT_TEMPLATES: tuple[str, ...] = ("{L}",)


@dataclass(frozen=True)
class MarginResult:
    pred_label: str
    correct: bool
    margin: float
    is_tie: bool
    correct_score: float
    best_wrong_score: float


@dataclass
class ScoringTelemetry:
    batches: int = 0
    actual_tokens: int = 0
    padded_tokens: int = 0
    input_preparation_seconds: float = 0.0

    @property
    def padding_ratio(self) -> float:
        if self.padded_tokens == 0:
            return 0.0
        return 1.0 - (self.actual_tokens / self.padded_tokens)


def label_variants(label: str) -> list[str]:
    label = validate_label(label)
    return [template.format(L=label) for template in VARIANT_TEMPLATES]


def tokenize_text(tokenizer, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text, add_special_tokens=False))
    encoded = tokenizer(text, add_special_tokens=False)
    return list(encoded["input_ids"])


def _pad_id(tokenizer) -> int:
    value = getattr(tokenizer, "pad_token_id", None)
    if value is not None:
        return int(value)
    value = getattr(tokenizer, "eos_token_id", None)
    if value is not None:
        return int(value)
    return 0


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
    except AttributeError:
        return torch.device("cpu")


def single_token_label_ids(
    tokenizer,
    *,
    labels: tuple[str, ...] = LABELS,
) -> dict[str, int] | None:
    token_ids: dict[str, int] = {}
    for label in labels:
        validate_label(label)
        ids = tokenize_text(tokenizer, label)
        if len(ids) != 1:
            return None
        token_ids[label] = int(ids[0])
    return token_ids


def _score_single_token_labels_many(
    model,
    tokenizer,
    prompts: list[str],
    label_ids: Mapping[str, int],
    *,
    batch_size: int,
    max_batch_tokens: int | None,
    device: torch.device | str | None,
    telemetry: ScoringTelemetry | None,
) -> list[dict[str, float]]:
    prompt_ids = [tokenize_text(tokenizer, prompt) for prompt in prompts]
    if any(not ids for ids in prompt_ids):
        raise ValueError("Prompts must contain at least one token to score completions")
    device_obj = torch.device(device) if device is not None else _model_device(model)
    pad_token_id = _pad_id(tokenizer)
    ordered = sorted(range(len(prompts)), key=lambda index: (len(prompt_ids[index]), index))
    output: list[dict[str, float] | None] = [None for _ in prompts]

    def flush(indices: list[int]) -> None:
        if not indices:
            return
        preparation_started = time.monotonic()
        max_len = max(len(prompt_ids[index]) for index in indices)
        input_ids = torch.full((len(indices), max_len), pad_token_id, dtype=torch.long, device=device_obj)
        attention_mask = torch.zeros_like(input_ids)
        for row, index in enumerate(indices):
            ids = prompt_ids[index]
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device_obj)
            attention_mask[row, : len(ids)] = 1
        if telemetry is not None:
            telemetry.batches += 1
            telemetry.actual_tokens += sum(len(prompt_ids[index]) for index in indices)
            telemetry.padded_tokens += len(indices) * max_len
            telemetry.input_preparation_seconds += time.monotonic() - preparation_started
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            for row, index in enumerate(indices):
                log_probs = torch.log_softmax(logits[row, len(prompt_ids[index]) - 1], dim=-1)
                output[index] = {
                    label: float(log_probs[token_id].detach().cpu())
                    for label, token_id in label_ids.items()
                }

    current: list[int] = []
    current_max_len = 0
    for index in ordered:
        length = len(prompt_ids[index])
        candidate_max_len = max(current_max_len, length)
        count_exceeded = len(current) >= batch_size
        tokens_exceeded = (
            max_batch_tokens is not None
            and current
            and candidate_max_len * (len(current) + 1) > max_batch_tokens
        )
        if count_exceeded or tokens_exceeded:
            flush(current)
            current = []
            current_max_len = 0
        current.append(index)
        current_max_len = max(current_max_len, length)
    flush(current)
    if any(scores is None for scores in output):
        raise RuntimeError("single-token scorer failed to produce every prompt result")
    return [scores for scores in output if scores is not None]


def completion_logps(
    model,
    tokenizer,
    prompt: str,
    completions: list[str],
    *,
    batch_size: int = 40,
    device: torch.device | str | None = None,
    forward_context_factory: Callable[[], object] | None = None,
) -> list[float]:
    """Score completions after a prompt using manual token concatenation.

    The prompt and completion are tokenized separately, then concatenated at the
    token-id level. This avoids BPE merges across the prompt-completion boundary.
    """

    if not completions:
        return []
    prompt_ids = tokenize_text(tokenizer, prompt)
    if not prompt_ids:
        raise ValueError("Prompt must contain at least one token to score completions")
    completion_ids = [tokenize_text(tokenizer, completion) for completion in completions]
    if any(len(ids) == 0 for ids in completion_ids):
        raise ValueError("Completion variants must contain at least one token")

    device_obj = torch.device(device) if device is not None else _model_device(model)
    pad_token_id = _pad_id(tokenizer)
    scores: list[float] = []

    for start in range(0, len(completions), batch_size):
        batch_ids = completion_ids[start : start + batch_size]
        sequences = [prompt_ids + ids for ids in batch_ids]
        max_len = max(len(seq) for seq in sequences)
        input_ids = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long, device=device_obj)
        attention_mask = torch.zeros_like(input_ids)
        for row, seq in enumerate(sequences):
            seq_tensor = torch.tensor(seq, dtype=torch.long, device=device_obj)
            input_ids[row, : len(seq)] = seq_tensor
            attention_mask[row, : len(seq)] = 1

        context = forward_context_factory() if forward_context_factory else nullcontext()
        with context:
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                log_probs = torch.log_softmax(logits, dim=-1)

        for row, ids in enumerate(batch_ids):
            total = torch.zeros((), dtype=log_probs.dtype, device=log_probs.device)
            for offset, token_id in enumerate(ids):
                absolute_pos = len(prompt_ids) + offset
                total = total + log_probs[row, absolute_pos - 1, token_id]
            scores.append(float(total.detach().cpu()))
    return scores


def _repeat_past_key_values(past_key_values, repeats: int):
    if hasattr(past_key_values, "batch_repeat_interleave"):
        repeated = copy.deepcopy(past_key_values)
        result = repeated.batch_repeat_interleave(repeats)
        return repeated if result is None else result
    if isinstance(past_key_values, tuple):
        return tuple(
            tuple(tensor.repeat_interleave(repeats, dim=0) for tensor in layer)
            if isinstance(layer, tuple)
            else layer.repeat_interleave(repeats, dim=0)
            for layer in past_key_values
        )
    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values).__name__}")


def completion_logps_cached(
    model,
    tokenizer,
    prompt: str,
    completions: list[str],
    *,
    batch_size: int = 40,
    device: torch.device | str | None = None,
) -> list[float]:
    if not completions:
        return []
    prompt_ids = tokenize_text(tokenizer, prompt)
    if not prompt_ids:
        raise ValueError("Prompt must contain at least one token to score completions")
    completion_ids = [tokenize_text(tokenizer, completion) for completion in completions]
    if any(len(ids) == 0 for ids in completion_ids):
        raise ValueError("Completion variants must contain at least one token")

    device_obj = torch.device(device) if device is not None else _model_device(model)
    prompt_prefix_ids = prompt_ids[:-1]
    seed_ids_by_completion = [[prompt_ids[-1]] + ids[:-1] for ids in completion_ids]
    pad_token_id = _pad_id(tokenizer)

    past_key_values = None
    if prompt_prefix_ids:
        prefix_tensor = torch.tensor([prompt_prefix_ids], dtype=torch.long, device=device_obj)
        prefix_attention = torch.ones_like(prefix_tensor)
        try:
            with torch.no_grad():
                prefix_outputs = model(input_ids=prefix_tensor, attention_mask=prefix_attention, use_cache=True)
        except TypeError:
            return completion_logps(model, tokenizer, prompt, completions, batch_size=batch_size, device=device_obj)
        past_key_values = getattr(prefix_outputs, "past_key_values", None)
        if past_key_values is None:
            return completion_logps(model, tokenizer, prompt, completions, batch_size=batch_size, device=device_obj)

    scores = [0.0 for _ in completion_ids]
    indexed_seeds = list(enumerate(seed_ids_by_completion))
    for start in range(0, len(indexed_seeds), batch_size):
        batch = indexed_seeds[start : start + batch_size]
        max_len = max(len(seed_ids) for _, seed_ids in batch)
        input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long, device=device_obj)
        if prompt_prefix_ids:
            attention_mask = torch.ones(
                (len(batch), len(prompt_prefix_ids) + max_len),
                dtype=torch.long,
                device=device_obj,
            )
        else:
            attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long, device=device_obj)
        for row_index, (_, seed_ids) in enumerate(batch):
            seed_tensor = torch.tensor(seed_ids, dtype=torch.long, device=device_obj)
            input_ids[row_index, : len(seed_ids)] = seed_tensor
            if prompt_prefix_ids:
                if len(seed_ids) < max_len:
                    attention_mask[row_index, len(prompt_prefix_ids) + len(seed_ids) :] = 0
            else:
                attention_mask[row_index, : len(seed_ids)] = 1
        kwargs = {}
        if past_key_values is not None:
            kwargs["past_key_values"] = _repeat_past_key_values(past_key_values, len(batch))
            kwargs["use_cache"] = True
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            log_probs = torch.log_softmax(outputs.logits, dim=-1)
        for row_index, (completion_index, _) in enumerate(batch):
            total = torch.zeros((), dtype=log_probs.dtype, device=log_probs.device)
            for offset, token_id in enumerate(completion_ids[completion_index]):
                total = total + log_probs[row_index, offset, token_id]
            scores[completion_index] = float(total.detach().cpu())
    return scores


def completion_logps_many(
    model,
    tokenizer,
    prompts: list[str],
    completions: list[str],
    *,
    batch_size: int = 128,
    max_batch_tokens: int | None = None,
    device: torch.device | str | None = None,
    forward_context_factory: Callable[[], object] | None = None,
    use_cache: bool = False,
    telemetry: ScoringTelemetry | None = None,
) -> list[list[float]]:
    """Score the same completion set for many prompts in length-aware batches."""

    if not prompts:
        return []
    if not completions:
        return [[] for _ in prompts]
    if use_cache and forward_context_factory is None:
        return [
            completion_logps_cached(
                model,
                tokenizer,
                prompt,
                completions,
                batch_size=batch_size,
                device=device,
            )
            for prompt in prompts
        ]
    prompt_ids_by_index = [tokenize_text(tokenizer, prompt) for prompt in prompts]
    if any(len(ids) == 0 for ids in prompt_ids_by_index):
        raise ValueError("Prompts must contain at least one token to score completions")
    completion_ids = [tokenize_text(tokenizer, completion) for completion in completions]
    if any(len(ids) == 0 for ids in completion_ids):
        raise ValueError("Completion variants must contain at least one token")

    device_obj = torch.device(device) if device is not None else _model_device(model)
    pad_token_id = _pad_id(tokenizer)
    out = [[0.0 for _ in completions] for _ in prompts]
    entries: list[tuple[int, int, int]] = []
    for prompt_index, prompt_ids in enumerate(prompt_ids_by_index):
        for completion_index, ids in enumerate(completion_ids):
            entries.append((len(prompt_ids) + len(ids), prompt_index, completion_index))
    entries.sort(key=lambda entry: entry[0])

    def flush(batch: list[tuple[int, int, int]]) -> None:
        if not batch:
            return
        preparation_started = time.monotonic()
        sequences: list[list[int]] = []
        for _, prompt_index, completion_index in batch:
            sequences.append(prompt_ids_by_index[prompt_index] + completion_ids[completion_index])
        max_len = max(len(seq) for seq in sequences)
        input_ids = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long, device=device_obj)
        attention_mask = torch.zeros_like(input_ids)
        for row_index, seq in enumerate(sequences):
            seq_tensor = torch.tensor(seq, dtype=torch.long, device=device_obj)
            input_ids[row_index, : len(seq)] = seq_tensor
            attention_mask[row_index, : len(seq)] = 1
        if telemetry is not None:
            telemetry.batches += 1
            telemetry.actual_tokens += sum(len(sequence) for sequence in sequences)
            telemetry.padded_tokens += len(sequences) * max_len
            telemetry.input_preparation_seconds += time.monotonic() - preparation_started

        context = forward_context_factory() if forward_context_factory else nullcontext()
        with context:
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                log_probs = torch.log_softmax(outputs.logits, dim=-1)

        for row_index, (_, prompt_index, completion_index) in enumerate(batch):
            prompt_len = len(prompt_ids_by_index[prompt_index])
            ids = completion_ids[completion_index]
            total = torch.zeros((), dtype=log_probs.dtype, device=log_probs.device)
            for offset, token_id in enumerate(ids):
                absolute_pos = prompt_len + offset
                total = total + log_probs[row_index, absolute_pos - 1, token_id]
            out[prompt_index][completion_index] = float(total.detach().cpu())

    current: list[tuple[int, int, int]] = []
    current_max_len = 0
    for entry in entries:
        total_len = entry[0]
        candidate_max_len = max(current_max_len, total_len)
        would_exceed_count = len(current) >= batch_size
        would_exceed_tokens = (
            max_batch_tokens is not None
            and current
            and candidate_max_len * (len(current) + 1) > max_batch_tokens
        )
        if would_exceed_count or would_exceed_tokens:
            flush(current)
            current = []
            current_max_len = 0
        current.append(entry)
        current_max_len = max(current_max_len, total_len)
    flush(current)
    return out


def score_labels(
    model,
    tokenizer,
    prompt: str,
    *,
    labels: tuple[str, ...] = LABELS,
    batch_size: int = 40,
    device: torch.device | str | None = None,
    forward_context_factory: Callable[[], object] | None = None,
) -> dict[str, float]:
    completions: list[str] = []
    label_for_completion: list[str] = []
    for label in labels:
        validate_label(label)
        variants = label_variants(label)
        completions.extend(variants)
        label_for_completion.extend([label] * len(variants))

    logps = completion_logps(
        model,
        tokenizer,
        prompt,
        completions,
        batch_size=batch_size,
        device=device,
        forward_context_factory=forward_context_factory,
    )
    out: dict[str, float] = {}
    for label in labels:
        values = torch.tensor(
            [score for score, score_label in zip(logps, label_for_completion, strict=True) if score_label == label],
            dtype=torch.float64,
        )
        out[label] = float(torch.logsumexp(values, dim=0) - torch.log(torch.tensor(float(len(values)))))
    return out


def score_labels_many(
    model,
    tokenizer,
    prompts: list[str],
    *,
    labels: tuple[str, ...] = LABELS,
    batch_size: int = 128,
    max_batch_tokens: int | None = None,
    device: torch.device | str | None = None,
    forward_context_factory: Callable[[], object] | None = None,
    use_cache: bool = False,
    telemetry: ScoringTelemetry | None = None,
) -> list[dict[str, float]]:
    label_ids = single_token_label_ids(tokenizer, labels=labels)
    if label_ids is not None and forward_context_factory is None and not use_cache:
        return _score_single_token_labels_many(
            model,
            tokenizer,
            prompts,
            label_ids,
            batch_size=batch_size,
            max_batch_tokens=max_batch_tokens,
            device=device,
            telemetry=telemetry,
        )
    completions: list[str] = []
    label_for_completion: list[str] = []
    for label in labels:
        validate_label(label)
        variants = label_variants(label)
        completions.extend(variants)
        label_for_completion.extend([label] * len(variants))

    logps_by_prompt = completion_logps_many(
        model,
        tokenizer,
        prompts,
        completions,
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        device=device,
        forward_context_factory=forward_context_factory,
        use_cache=use_cache,
        telemetry=telemetry,
    )
    all_scores: list[dict[str, float]] = []
    for logps in logps_by_prompt:
        prompt_scores: dict[str, float] = {}
        for label in labels:
            values = torch.tensor(
                [score for score, score_label in zip(logps, label_for_completion, strict=True) if score_label == label],
                dtype=torch.float64,
            )
            prompt_scores[label] = float(torch.logsumexp(values, dim=0) - torch.log(torch.tensor(float(len(values)))))
        all_scores.append(prompt_scores)
    return all_scores


def predict_from_scores(scores: Mapping[str, float], *, epsilon: float = EPSILON) -> tuple[str, bool]:
    ordered = sorted(((label, float(scores[label])) for label in LABELS), key=lambda kv: (-kv[1], kv[0]))
    if len(ordered) > 1 and abs(ordered[0][1] - ordered[1][1]) <= epsilon:
        return ordered[0][0], True
    return ordered[0][0], False


def correct_answer_margin(
    scores: Mapping[str, float],
    correct_label: str,
    *,
    epsilon: float = EPSILON,
) -> MarginResult:
    correct_label = validate_label(correct_label, name="correct_label")
    pred_label, is_tie = predict_from_scores(scores, epsilon=epsilon)
    correct_score = float(scores[correct_label])
    wrong_scores = [float(score) for label, score in scores.items() if label != correct_label]
    best_wrong_score = max(wrong_scores)
    margin = correct_score - best_wrong_score
    return MarginResult(
        pred_label=pred_label,
        correct=(pred_label == correct_label and not is_tie and margin > epsilon),
        margin=float(margin),
        is_tie=bool(is_tie or abs(margin) <= epsilon),
        correct_score=correct_score,
        best_wrong_score=float(best_wrong_score),
    )
