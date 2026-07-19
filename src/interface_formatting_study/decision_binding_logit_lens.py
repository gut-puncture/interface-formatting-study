"""Exact, training-free layerwise scoring for the decision-binding logit lens.

This module deliberately owns only the numerical seam.  Artifact preparation,
resume, analysis, and operator behavior belong to the experiment CLI.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from .hooks import find_transformer_blocks


PARITY_ATOL = 0.02
LETTERS = ("A", "B", "C", "D")
_FIXED_ROOT_TOKENIZER_CACHE: dict[int, tuple[object, object]] = {}


@dataclass(frozen=True)
class CandidateContinuation:
    surface: str
    token_ids: tuple[int, ...]
    surface_sha256: str
    utf8_bytes: int
    eligible: bool
    eligibility_reason: str


@dataclass(frozen=True)
class ContinuationAudit:
    prompt: str
    prompt_ids: tuple[int, ...]
    prompt_sha256: str
    prompt_token_sha256: str
    label_token_ids: tuple[int, int, int, int]
    candidates: tuple[CandidateContinuation, ...]
    duplicate_surface_indices: tuple[tuple[int, ...], ...]
    identical_token_indices: tuple[tuple[int, ...], ...]
    prefix_collision_indices: tuple[tuple[int, int], ...]
    shared_first_token_indices: tuple[tuple[int, ...], ...]
    label_like_indices: tuple[int, ...]
    identity_comparison_evaluable: bool
    identity_ineligibility_reasons: tuple[str, ...]


@dataclass(frozen=True)
class LayerwisePathScores:
    """Scores are shaped ``[transformer block, candidate-or-letter]``."""

    first_token_logp: torch.Tensor
    total_logp: torch.Tensor
    mean_token_logp: torch.Tensor
    letter_logp: torch.Tensor
    max_final_native_difference: float


def _encode(tokenizer, text: str) -> tuple[int, ...]:
    if hasattr(tokenizer, "encode"):
        values = tokenizer.encode(text, add_special_tokens=False)
    else:
        values = tokenizer(text, add_special_tokens=False)["input_ids"]
    return tuple(int(value) for value in values)


def _decode(tokenizer, token_ids: Sequence[int]) -> str:
    kwargs = {
        "skip_special_tokens": False,
        "clean_up_tokenization_spaces": False,
    }
    try:
        return str(tokenizer.decode(list(token_ids), **kwargs))
    except TypeError:
        return str(tokenizer.decode(list(token_ids)))


def continuation_tokenization_policy(tokenizer) -> str:
    """Describe the exact fixed-root continuation backend behavior."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    pre_tokenizer = getattr(backend, "pre_tokenizer", None)
    description = repr(pre_tokenizer)
    if description.startswith("Metaspace("):
        if (
            'replacement="▁"' not in description
            or "prepend_scheme=first" not in description
            or "split=False" not in description
        ):
            raise ValueError("unexpected Metaspace tokenizer policy")
        return "fixed_root_metaspace_without_implicit_prefix"

    normalizer = repr(getattr(backend, "normalizer", None))
    if (
        normalizer.startswith("Sequence(normalizers=[Prepend(")
        and 'prepend="▁"' in normalizer
        and 'Replace(pattern=String(" "), content="▁")' in normalizer
    ):
        return "fixed_root_sentencepiece_without_implicit_prefix"
    return "unchanged_canonical_backend"


def _fixed_root_continuation_tokenizer(tokenizer):
    """Disable an implicit standalone word prefix at the fixed boundary."""

    policy = continuation_tokenization_policy(tokenizer)
    if policy == "unchanged_canonical_backend":
        return tokenizer
    cached = _FIXED_ROOT_TOKENIZER_CACHE.get(id(tokenizer))
    if cached is not None and cached[0] is tokenizer:
        return cached[1]
    continuation_tokenizer = copy.deepcopy(tokenizer)
    if policy == "fixed_root_metaspace_without_implicit_prefix":
        from tokenizers.pre_tokenizers import Metaspace

        continuation_tokenizer.backend_tokenizer.pre_tokenizer = Metaspace(
            replacement="▁", prepend_scheme="never", split=False
        )
    elif policy == "fixed_root_sentencepiece_without_implicit_prefix":
        from tokenizers.normalizers import Replace, Sequence as NormalizerSequence

        continuation_tokenizer.backend_tokenizer.normalizer = NormalizerSequence(
            [Replace(" ", "▁")]
        )
    else:  # pragma: no cover - policy is exhaustive and bound above
        raise AssertionError(f"unsupported continuation tokenizer policy: {policy}")
    _FIXED_ROOT_TOKENIZER_CACHE[id(tokenizer)] = (tokenizer, continuation_tokenizer)
    return continuation_tokenizer


def _token_hash(token_ids: Sequence[int]) -> str:
    payload = ",".join(str(int(value)) for value in token_ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _groups(values: Sequence[object]) -> tuple[tuple[int, ...], ...]:
    grouped: dict[object, list[int]] = {}
    for index, value in enumerate(values):
        grouped.setdefault(value, []).append(index)
    return tuple(tuple(indices) for indices in grouped.values() if len(indices) > 1)


def audit_fixed_root_continuations(
    tokenizer,
    prompt: str,
    candidate_surfaces: Sequence[str],
    *,
    max_context_tokens: int | None = None,
) -> ContinuationAudit:
    """Freeze exact candidate token paths after an already-authenticated root.

    Prompt and candidate tokens are encoded separately and concatenated at the
    ID boundary. An implicit standalone Metaspace or SentencePiece prefix is
    disabled for the continuation because the fixed root already contains every
    real boundary byte. The exact source prompt string and its exact tokenizer
    IDs are both bound; the combined decode must reproduce the decoded fixed
    root plus the exact surface. No spelling, case, whitespace, punctuation, or
    tokenization variants are introduced.
    """

    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a non-empty string")
    if isinstance(candidate_surfaces, (str, bytes)) or len(candidate_surfaces) != 4:
        raise ValueError("candidate_surfaces must contain exactly four strings")
    if any(not isinstance(surface, str) for surface in candidate_surfaces):
        raise ValueError("candidate_surfaces must contain exactly four strings")

    prompt_ids = _encode(tokenizer, prompt)
    if not prompt_ids:
        raise ValueError("prompt must contain at least one token")
    decoded_prompt = _decode(tokenizer, prompt_ids)

    continuation_tokenizer = _fixed_root_continuation_tokenizer(tokenizer)
    label_ids = tuple(_encode(continuation_tokenizer, letter) for letter in LETTERS)
    if any(len(values) != 1 for values in label_ids):
        raise ValueError("A/B/C/D must be distinct single-token continuations")
    flattened_labels = tuple(values[0] for values in label_ids)
    if len(set(flattened_labels)) != 4:
        raise ValueError("A/B/C/D must be distinct single-token continuations")
    if any(
        _decode(tokenizer, (*prompt_ids, token_id)) != decoded_prompt + letter
        for letter, token_id in zip(LETTERS, flattened_labels, strict=True)
    ):
        raise ValueError("A/B/C/D token decoding changes at the exact prompt boundary")

    special_ids = {int(value) for value in getattr(tokenizer, "all_special_ids", ())}
    candidates: list[CandidateContinuation] = []
    for surface in candidate_surfaces:
        token_ids = _encode(continuation_tokenizer, surface)
        reason = "eligible"
        if not surface or not token_ids:
            reason = "empty_candidate"
        elif any(token_id in special_ids for token_id in token_ids):
            reason = "special_token"
        elif (
            max_context_tokens is not None
            and len(prompt_ids) + len(token_ids) > int(max_context_tokens)
        ):
            reason = "context_overflow"
        elif _decode(tokenizer, token_ids) != surface:
            reason = "candidate_decode_mismatch"
        elif _decode(tokenizer, (*prompt_ids, *token_ids)) != decoded_prompt + surface:
            reason = "combined_decode_mismatch"
        candidates.append(
            CandidateContinuation(
                surface=surface,
                token_ids=token_ids,
                surface_sha256=hashlib.sha256(surface.encode("utf-8")).hexdigest(),
                utf8_bytes=len(surface.encode("utf-8")),
                eligible=reason == "eligible",
                eligibility_reason=reason,
            )
        )

    surfaces = [candidate.surface for candidate in candidates]
    token_sequences = [candidate.token_ids for candidate in candidates]
    duplicate_surfaces = _groups(surfaces)
    identical_tokens = _groups(token_sequences)
    prefix_collisions = tuple(
        (left, right)
        for left in range(4)
        for right in range(left + 1, 4)
        if token_sequences[left]
        and token_sequences[right]
        and token_sequences[left] != token_sequences[right]
        and (
            token_sequences[left] == token_sequences[right][: len(token_sequences[left])]
            or token_sequences[right] == token_sequences[left][: len(token_sequences[right])]
        )
    )
    first_tokens = [values[0] if values else None for values in token_sequences]
    shared_first = tuple(group for group in _groups(first_tokens) if first_tokens[group[0]] is not None)
    label_like = tuple(index for index, surface in enumerate(surfaces) if surface in LETTERS)

    reasons: list[str] = []
    if any(not candidate.eligible for candidate in candidates):
        reasons.extend(
            dict.fromkeys(
                candidate.eligibility_reason
                for candidate in candidates
                if not candidate.eligible
            )
        )
    for present, reason in (
        (bool(duplicate_surfaces), "duplicate_surface"),
        (bool(identical_tokens), "identical_token_sequence"),
        (bool(prefix_collisions), "prefix_collision"),
        (bool(label_like), "label_like_candidate"),
    ):
        if present:
            reasons.append(reason)

    return ContinuationAudit(
        prompt=prompt,
        prompt_ids=prompt_ids,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        prompt_token_sha256=_token_hash(prompt_ids),
        label_token_ids=flattened_labels,  # type: ignore[arg-type]
        candidates=tuple(candidates),
        duplicate_surface_indices=duplicate_surfaces,
        identical_token_indices=identical_tokens,
        prefix_collision_indices=prefix_collisions,
        shared_first_token_indices=shared_first,
        label_like_indices=label_like,
        identity_comparison_evaluable=not reasons,
        identity_ineligibility_reasons=tuple(reasons),
    )


def tied_argmax_indices(scores: Sequence[float] | torch.Tensor) -> tuple[int, ...]:
    values = scores if isinstance(scores, torch.Tensor) else torch.as_tensor(scores, dtype=torch.float64)
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("scores must be a non-empty vector")
    if not torch.isfinite(values).all():
        raise ValueError("scores contain non-finite values")
    maximum = values.max()
    return tuple(int(index) for index in torch.nonzero(values == maximum, as_tuple=False).flatten())


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cpu")


def _final_norm_and_head(model):
    inner = getattr(model, "model", None)
    norm = getattr(inner, "norm", None)
    head = getattr(model, "lm_head", None)
    if norm is None or head is None:
        raise ValueError("logit lens requires model.model.norm and model.lm_head")
    return norm, head


def _forward_last_position(
    model,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor,
    target_token_ids: Sequence[int],
    past_key_values=None,
    use_cache: bool,
):
    blocks = find_transformer_blocks(model)
    norm, head = _final_norm_and_head(model)
    captured: dict[int, torch.Tensor] = {}

    def hook_for(layer: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Clone the one position we need so the view does not keep the
            # complete layer sequence storage alive through projection.
            captured[layer] = hidden[:, -1, :].clone()
            return output

        return hook

    handles = [block.register_forward_hook(hook_for(layer)) for layer, block in enumerate(blocks)]
    try:
        with torch.inference_mode():
            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "use_cache": use_cache,
                "logits_to_keep": 1,
            }
            if past_key_values is not None:
                kwargs["past_key_values"] = past_key_values
            outputs = model(**kwargs)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(range(len(blocks))):
        raise RuntimeError("not every transformer-block logit-lens hook ran")

    target_tensor = torch.tensor(target_token_ids, dtype=torch.long, device=input_ids.device)
    with torch.inference_mode():
        layer_rows: list[torch.Tensor] = []
        if len(blocks) > 1:
            intermediate_hidden = torch.stack(
                [captured[layer] for layer in range(len(blocks) - 1)], dim=1
            )
            intermediate_projected = head(norm(intermediate_hidden)).float()
            layer_rows.append(
                torch.log_softmax(intermediate_projected, dim=-1).index_select(
                    -1, target_tensor
                )
            )
        # Preserve the native [batch, sequence, hidden] projection shape. BF16
        # GEMM kernels can differ when the singleton sequence axis is removed.
        final_projected = head(
            norm(captured[len(blocks) - 1].unsqueeze(1))
        ).float()
        final_scores = torch.log_softmax(final_projected, dim=-1).index_select(
            -1, target_tensor
        )[:, -1, :]
        layer_rows.append(final_scores.unsqueeze(1))
        native = torch.log_softmax(outputs.logits[:, -1, :].float(), dim=-1).index_select(
            -1, target_tensor
        )
    layer_scores = torch.cat(layer_rows, dim=1)
    if not torch.isfinite(layer_scores).all() or not torch.isfinite(native).all():
        raise ValueError("logit-lens projection contains non-finite values")
    final_difference = float((layer_scores[:, -1] - native).abs().max().detach().cpu())
    if final_difference > PARITY_ATOL:
        raise ValueError(
            f"final logit-lens/native parity difference {final_difference:.6g} exceeds {PARITY_ATOL}"
        )
    return (
        layer_scores.detach().float().cpu(),
        getattr(outputs, "past_key_values", None),
        final_difference,
    )


def _validate_scoring_inputs(model, audit: ContinuationAudit, expected_layers: int) -> int:
    blocks = find_transformer_blocks(model)
    if len(blocks) != int(expected_layers):
        raise ValueError(
            f"model has {len(blocks)} transformer blocks; expected {int(expected_layers)}"
        )
    if not audit.prompt_ids:
        raise ValueError("continuation audit has no prompt tokens")
    return len(blocks)


def _empty_outputs(layers: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    first = torch.full((layers, 4), float("nan"), dtype=torch.float32)
    total = torch.full_like(first, float("nan"))
    mean = torch.full_like(first, float("nan"))
    return first, total, mean


def _root_forward(model, audit: ContinuationAudit, *, use_cache: bool):
    device = _model_device(model)
    input_ids = torch.tensor([audit.prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    first_ids = [candidate.token_ids[0] for candidate in audit.candidates if candidate.token_ids]
    targets = tuple(dict.fromkeys((*audit.label_token_ids, *first_ids)))
    scores, cache, difference = _forward_last_position(
        model,
        input_ids,
        attention_mask=attention_mask,
        target_token_ids=targets,
        use_cache=use_cache,
    )
    index_by_token = {token_id: index for index, token_id in enumerate(targets)}
    return scores[0], cache, difference, index_by_token


def score_candidate_paths_scalar(
    model,
    audit: ContinuationAudit,
    *,
    expected_layers: int = 32,
) -> LayerwisePathScores:
    """Uncached correctness oracle using complete prompt-prefix forwards."""

    layers = _validate_scoring_inputs(model, audit, expected_layers)
    first, total, mean = _empty_outputs(layers)
    root, _cache, max_difference, root_index = _root_forward(model, audit, use_cache=False)
    letter = torch.stack([root[:, root_index[token_id]] for token_id in audit.label_token_ids], dim=1)

    device = _model_device(model)
    for candidate_index, candidate in enumerate(audit.candidates):
        if not candidate.eligible:
            continue
        path = candidate.token_ids
        first[:, candidate_index] = root[:, root_index[path[0]]]
        total[:, candidate_index] = first[:, candidate_index]
        for offset in range(1, len(path)):
            prefix = (*audit.prompt_ids, *path[:offset])
            input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
            target = path[offset]
            projected, _unused_cache, difference = _forward_last_position(
                model,
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                target_token_ids=(target,),
                use_cache=False,
            )
            total[:, candidate_index] += projected[0, :, 0]
            max_difference = max(max_difference, difference)
        mean[:, candidate_index] = total[:, candidate_index] / len(path)
    return LayerwisePathScores(first, total, mean, letter, max_difference)


def _select_cache_rows(cache, indices: Sequence[int], *, device: torch.device):
    selection = torch.tensor(indices, dtype=torch.long, device=device)
    if hasattr(cache, "batch_select_indices"):
        selected = copy.deepcopy(cache)
        result = selected.batch_select_indices(selection)
        return selected if result is None else result
    if isinstance(cache, tuple):
        return tuple(
            tuple(tensor.index_select(0, selection.to(tensor.device)) for tensor in layer)
            if isinstance(layer, tuple)
            else layer.index_select(0, selection.to(layer.device))
            for layer in cache
        )
    raise TypeError(f"unsupported past_key_values type: {type(cache).__name__}")


def score_candidate_paths_cached(
    model,
    audit: ContinuationAudit,
    *,
    expected_layers: int = 32,
    phase_callback: Callable[[str], None] | None = None,
) -> LayerwisePathScores:
    """Production scorer: one root prefill plus batched shared-prefix frontiers."""

    return score_candidate_paths_cached_many(
        model,
        [audit],
        expected_layers=expected_layers,
        phase_callback=phase_callback,
    )[0]


def score_candidate_paths_cached_many(
    model,
    audits: Sequence[ContinuationAudit],
    *,
    expected_layers: int = 32,
    phase_callback: Callable[[str], None] | None = None,
) -> list[LayerwisePathScores]:
    """Batch equal-length roots and all divergent trie frontiers.

    The caller owns length grouping and runtime batch/token caps.  Requiring one
    exact root length here keeps cache positions unambiguous and avoids padding
    becoming part of the scientific computation.
    """

    if not audits:
        return []
    root_lengths = {len(audit.prompt_ids) for audit in audits}
    if len(root_lengths) != 1:
        raise ValueError("cached root batches require prompts of equal token length")
    for audit in audits:
        _validate_scoring_inputs(model, audit, expected_layers)

    layers = int(expected_layers)
    device = _model_device(model)
    input_ids = torch.tensor([audit.prompt_ids for audit in audits], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    targets = tuple(
        dict.fromkeys(
            token_id
            for audit in audits
            for token_id in (
                *audit.label_token_ids,
                *(candidate.token_ids[0] for candidate in audit.candidates if candidate.token_ids),
            )
        )
    )
    if phase_callback is not None:
        phase_callback("root")
    try:
        root_scores, parent_cache, max_difference = _forward_last_position(
            model,
            input_ids,
            attention_mask=attention_mask,
            target_token_ids=targets,
            use_cache=True,
        )
    except BaseException:
        if phase_callback is not None:
            phase_callback("idle")
        raise
    if parent_cache is None:
        raise ValueError("model did not return a KV cache")
    target_index = {token_id: index for index, token_id in enumerate(targets)}

    first_by_audit: list[torch.Tensor] = []
    total_by_audit: list[torch.Tensor] = []
    mean_by_audit: list[torch.Tensor] = []
    letter_by_audit: list[torch.Tensor] = []
    eligible_paths: dict[tuple[int, int], tuple[int, ...]] = {}
    for audit_index, audit in enumerate(audits):
        first, total, mean = _empty_outputs(layers)
        root = root_scores[audit_index]
        letter = torch.stack(
            [root[:, target_index[token_id]] for token_id in audit.label_token_ids],
            dim=1,
        )
        for candidate_index, candidate in enumerate(audit.candidates):
            if not candidate.eligible:
                continue
            path = candidate.token_ids
            eligible_paths[(audit_index, candidate_index)] = path
            first[:, candidate_index] = root[:, target_index[path[0]]]
            total[:, candidate_index] = first[:, candidate_index]
        first_by_audit.append(first)
        total_by_audit.append(total)
        mean_by_audit.append(mean)
        letter_by_audit.append(letter)

    # Frontier identities include the audit row so unrelated prompts never
    # share a cache, while equal prefixes within one prompt are evaluated once.
    frontier = sorted(
        {
            (audit_index, path[:1])
            for (audit_index, _candidate_index), path in eligible_paths.items()
            if len(path) > 1
        }
    )
    parent_indices = [audit_index for audit_index, _prefix in frontier]
    depth = 1
    while frontier:
        if phase_callback is not None:
            phase_callback("branch")
        selected_cache = _select_cache_rows(parent_cache, parent_indices, device=device)
        branch_ids = torch.tensor(
            [[prefix[-1]] for _audit_index, prefix in frontier],
            dtype=torch.long,
            device=device,
        )
        branch_mask = torch.ones(
            (len(frontier), len(audits[0].prompt_ids) + depth),
            dtype=torch.long,
            device=device,
        )
        frontier_set = set(frontier)
        next_tokens = sorted(
            {
                path[depth]
                for (audit_index, _candidate_index), path in eligible_paths.items()
                if len(path) > depth and (audit_index, path[:depth]) in frontier_set
            }
        )
        projected, returned_cache, difference = _forward_last_position(
            model,
            branch_ids,
            attention_mask=branch_mask,
            target_token_ids=next_tokens,
            past_key_values=selected_cache,
            use_cache=True,
        )
        if returned_cache is None:
            raise ValueError("model stopped returning a KV cache during trie traversal")
        max_difference = max(max_difference, difference)
        next_token_index = {token_id: index for index, token_id in enumerate(next_tokens)}
        prefix_index = {entry: index for index, entry in enumerate(frontier)}
        for (audit_index, candidate_index), path in eligible_paths.items():
            if len(path) > depth:
                row = prefix_index[(audit_index, path[:depth])]
                total_by_audit[audit_index][:, candidate_index] += projected[
                    row, :, next_token_index[path[depth]]
                ]

        next_frontier = sorted(
            {
                (audit_index, path[: depth + 1])
                for (audit_index, _candidate_index), path in eligible_paths.items()
                if len(path) > depth + 1
            }
        )
        parent_indices = [
            prefix_index[(audit_index, prefix[:-1])]
            for audit_index, prefix in next_frontier
        ]
        frontier = next_frontier
        parent_cache = returned_cache
        depth += 1

    if phase_callback is not None:
        phase_callback("idle")

    results: list[LayerwisePathScores] = []
    for audit_index, _audit in enumerate(audits):
        for (path_audit_index, candidate_index), path in eligible_paths.items():
            if path_audit_index == audit_index:
                mean_by_audit[audit_index][:, candidate_index] = (
                    total_by_audit[audit_index][:, candidate_index] / len(path)
                )
        results.append(
            LayerwisePathScores(
                first_by_audit[audit_index],
                total_by_audit[audit_index],
                mean_by_audit[audit_index],
                letter_by_audit[audit_index],
                max_difference,
            )
        )
    return results
