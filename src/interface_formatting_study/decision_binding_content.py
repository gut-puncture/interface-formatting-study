from __future__ import annotations

import hashlib
import itertools
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

from .causal_design import _raw_label_spans, _text_spans
from .causal_option_maps import (
    LETTERS,
    ParsedPrompt,
    SourceSpan,
    parse_prompt_options,
    validate_option_map,
)
from .scoring import tokenize_text


_REQUIRED_LEDGER_COLUMNS = {
    "work_key",
    "item_id",
    "wrapper_name",
    "split",
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
    "winner_position",
    "winner_unique",
    "text_identity_ambiguous",
    "readout_role",
}
_REQUIRED_APPLICABILITY_COLUMNS = {
    "item_id",
    "wrapper_name",
    "split",
    "source_prompt_sha256",
    "parse_provenance",
}


@dataclass(frozen=True)
class _Replacement:
    span: SourceSpan
    text: str
    content_id: int | None = None
    selected_position: int | None = None


def _sequence(value: object, *, name: str) -> list[object]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    if not isinstance(value, Sequence):
        try:
            values = list(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError(f"{name} must be a sequence") from exc
    else:
        values = list(value)
    if len(values) != 4:
        raise ValueError(f"{name} must contain four values")
    return values


def _select_content_representation(
    parsed: ParsedPrompt,
    source: str,
    candidate_texts: Sequence[str],
) -> int:
    eligible: list[tuple[int, int]] = []
    for index, representation in enumerate(parsed.representations):
        matched_chars = 0
        valid = True
        for slot in representation.slots:
            payloads = [span.text(source) for span in slot.payload_spans]
            candidate = str(candidate_texts[slot.content_id])
            if not payloads or any(not payload or payload not in candidate for payload in payloads):
                valid = False
                break
            matched_chars += sum(len(payload) for payload in payloads)
        if valid:
            eligible.append((matched_chars, index))
    if not eligible:
        raise ValueError("no complete content-bearing option representation")
    best_chars = max(chars for chars, _ in eligible)
    return max(index for chars, index in eligible if chars == best_chars)


def _transform_with_selected_sites(
    parsed: ParsedPrompt,
    source: str,
    *,
    candidate_texts: Sequence[str],
    position_shift: int,
    label_shift: int,
) -> tuple[str, list[int], list[tuple[int, int]]]:
    validate_option_map(parsed, source)
    if position_shift not in range(4) or label_shift not in range(4):
        raise ValueError("position_shift and label_shift must be in [0, 3]")
    selected_representation = _select_content_representation(parsed, source, candidate_texts)
    replacements: list[_Replacement] = []
    actual_contents: list[int] = [-1] * 4
    for representation_index, representation in enumerate(parsed.representations):
        slots = representation.slots
        for position, target in enumerate(slots):
            moving = slots[(position - position_shift) % 4]
            new_label = LETTERS[(moving.content_id + label_shift) % 4]
            replacements.extend(_Replacement(span, new_label) for span in target.label_spans)
            if len(target.payload_spans) != len(moving.payload_spans):
                raise ValueError("option payload representations are inconsistent")
            for payload_index, (target_span, moving_span) in enumerate(
                zip(target.payload_spans, moving.payload_spans, strict=True)
            ):
                selected = (
                    representation_index == selected_representation
                    and payload_index == len(target.payload_spans) - 1
                )
                replacements.append(
                    _Replacement(
                        target_span,
                        moving_span.text(source),
                        moving.content_id if selected else None,
                        position if selected else None,
                    )
                )
                if selected:
                    actual_contents[position] = moving.content_id

    pieces: list[str] = []
    cursor = 0
    output_length = 0
    output_spans: list[tuple[int, int] | None] = [None] * 4
    for replacement in sorted(replacements, key=lambda item: item.span.start):
        if replacement.span.start < cursor:
            raise ValueError("option-map replacements overlap")
        unchanged = source[cursor : replacement.span.start]
        pieces.extend((unchanged, replacement.text))
        output_length += len(unchanged)
        start = output_length
        output_length += len(replacement.text)
        if replacement.selected_position is not None:
            output_spans[replacement.selected_position] = (start, output_length)
        cursor = replacement.span.end
    pieces.append(source[cursor:])
    if sorted(actual_contents) != [0, 1, 2, 3] or any(span is None for span in output_spans):
        raise ValueError("selected representation did not produce four candidate sites")
    return "".join(pieces), actual_contents, [span for span in output_spans if span is not None]


def _source_selected_sites(
    parsed: ParsedPrompt,
    source: str,
    candidate_texts: Sequence[str],
) -> tuple[list[int], list[tuple[int, int]]]:
    validate_option_map(parsed, source)
    representation = parsed.representations[
        _select_content_representation(parsed, source, candidate_texts)
    ]
    contents = [slot.content_id for slot in representation.slots]
    spans = [
        (slot.payload_spans[-1].start, slot.payload_spans[-1].end)
        for slot in representation.slots
    ]
    return contents, spans


def _last_complete_candidate_sequence(
    prompt: str,
    candidate_texts: Sequence[str],
) -> tuple[list[int], list[tuple[int, int]]]:
    if len(candidate_texts) != 4 or len(set(map(str, candidate_texts))) != 4:
        raise ValueError("candidate content identity is ambiguous")
    marker = prompt.rfind("Return only the letter")
    option_region = prompt[: marker if marker >= 0 else len(prompt)]
    occurrences = [
        [(match.start(), match.end()) for match in re.finditer(re.escape(str(text)), option_region)]
        for text in candidate_texts
    ]
    if any(not spans for spans in occurrences):
        raise ValueError("candidate text is absent from the exact prompt")
    best: tuple[tuple[object, ...], list[tuple[int, int, int]]] | None = None
    for content_order in itertools.permutations(range(4)):
        states: list[tuple[list[tuple[int, int, int]], int]] = [([], -1)]
        for content_id in content_order:
            next_states = [
                (sequence + [(content_id, start, end)], end)
                for sequence, previous_end in states
                for start, end in occurrences[content_id]
                if start >= previous_end
            ]
            if not next_states:
                break
            states = sorted(
                next_states,
                key=lambda state: (state[0][0][1], tuple(value for site in state[0] for value in site[1:])),
                reverse=True,
            )[:128]
        else:
            sequence = max(
                states,
                key=lambda state: (state[0][0][1], tuple(value for site in state[0] for value in site[1:])),
            )[0]
            key: tuple[object, ...] = (
                sequence[0][1],
                tuple(value for site in sequence for value in site[1:]),
            )
            if best is None or key > best[0]:
                best = (key, sequence)
    if best is None:
        raise ValueError("four candidate texts do not form one ordered option block")
    return [site[0] for site in best[1]], [(site[1], site[2]) for site in best[1]]


def _legacy_candidate_sites(
    prompt: str,
    wrapper_name: str,
    candidate_texts: Sequence[str],
    labels_by_position: Sequence[str],
    content_ids_by_position: Sequence[int],
) -> tuple[list[int], list[tuple[int, int]]]:
    raw_labels = _raw_label_spans(prompt, wrapper_name)
    expected_labels = [str(value) for value in labels_by_position]
    if len(raw_labels) == 4 and [span.value for span in raw_labels] == expected_labels:
        marker = prompt.rfind("Return only the letter")
        option_end = marker if marker >= 0 else len(prompt)
        bounded_spans: list[tuple[int, int]] = []
        for position, label_span in enumerate(raw_labels):
            region_end = raw_labels[position + 1].start if position + 1 < 4 else option_end
            content_id = int(content_ids_by_position[position])
            candidate = str(candidate_texts[content_id])
            occurrences = list(
                re.finditer(re.escape(candidate), prompt[label_span.end : region_end])
            )
            if len(occurrences) != 1:
                bounded_spans = []
                break
            match = occurrences[0]
            bounded_spans.append(
                (label_span.end + match.start(), label_span.end + match.end())
            )
        if len(bounded_spans) == 4:
            return [int(value) for value in content_ids_by_position], bounded_spans

    label_to_content = {
        str(label): int(content_id)
        for label, content_id in zip(labels_by_position, content_ids_by_position, strict=True)
    }
    choices_by_label = [str(candidate_texts[label_to_content[label]]) for label in LETTERS]
    try:
        label_spans = _raw_label_spans(prompt, wrapper_name)
        text_spans = _text_spans(prompt, choices_by_label, label_spans)
        content_by_text = {str(text): index for index, text in enumerate(candidate_texts)}
        ordered = sorted(
            ((content_by_text[span.value], span.start, span.end) for span in text_spans),
            key=lambda site: site[1],
        )
        if len(ordered) != 4:
            raise ValueError("legacy locator did not produce four candidate sites")
        return [site[0] for site in ordered], [(site[1], site[2]) for site in ordered]
    except (KeyError, ValueError):
        return _last_complete_candidate_sequence(prompt, candidate_texts)


def prepare_candidate_sites(
    ledger: pd.DataFrame,
    applicability: pd.DataFrame,
    *,
    option_maps: Mapping[tuple[str, str], ParsedPrompt],
) -> pd.DataFrame:
    """Bind exact existing prompts to four literal candidate-content endpoints."""

    if missing := _REQUIRED_LEDGER_COLUMNS - set(ledger.columns):
        raise ValueError(f"content-reader ledger is missing columns: {sorted(missing)}")
    if missing := _REQUIRED_APPLICABILITY_COLUMNS - set(applicability.columns):
        raise ValueError(f"content-reader applicability is missing columns: {sorted(missing)}")
    if ledger.empty or ledger["work_key"].duplicated().any():
        raise ValueError("content-reader ledger must be non-empty with unique work keys")
    keys = ["item_id", "wrapper_name", "split"]
    if applicability.duplicated(keys).any():
        raise ValueError("content-reader applicability contains duplicate blocks")
    status_by_key = {
        tuple(map(str, key)): row
        for key, row in applicability.set_index(keys).iterrows()
    }

    output: list[dict[str, object]] = []
    for block_key, block in ledger.groupby(keys, sort=False):
        key = tuple(map(str, block_key))
        if key not in status_by_key:
            raise ValueError(f"content-reader block lacks applicability: {key}")
        status = status_by_key[key]
        baseline = block[block["manipulation"].astype(str) == "controlled_baseline"]
        if len(baseline) != 1:
            raise ValueError(f"content-reader block requires one controlled baseline: {key}")
        baseline_row = baseline.iloc[0]
        source = str(baseline_row["prompt"])
        source_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if (
            source_digest != str(status["source_prompt_sha256"])
            or source_digest != str(baseline_row["source_prompt_sha256"])
        ):
            raise ValueError(f"source prompt checksum mismatch: {key}")
        candidates = [str(value) for value in _sequence(baseline_row["candidate_texts"], name="candidate_texts")]
        provenance = str(status["parse_provenance"])
        parsed: ParsedPrompt | None = None
        use_direct = provenance == "legacy_deterministic"
        if not use_direct:
            parsed = option_maps.get((key[0], key[1]))
            if parsed is None:
                parsed = parse_prompt_options(source, key[1], candidates)
            if not parsed.separable or not parsed.representations:
                parsed = None
                use_direct = True

        for row in block.sort_values("variant", kind="mergesort").to_dict("records"):
            prompt = str(row["prompt"])
            if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != str(row["prompt_sha256"]):
                raise ValueError(f"{row['work_key']} has a prompt checksum mismatch")
            row_candidates = [str(value) for value in _sequence(row["candidate_texts"], name="candidate_texts")]
            if row_candidates != candidates:
                raise ValueError(f"{row['work_key']} changes candidate content within a block")
            if use_direct:
                actual_contents, char_spans = _legacy_candidate_sites(
                    prompt,
                    key[1],
                    candidates,
                    [str(value) for value in _sequence(row["labels_by_position"], name="labels_by_position")],
                    [int(value) for value in _sequence(row["content_ids_by_position"], name="content_ids_by_position")],
                )
            else:
                assert parsed is not None
                if int(row["position_shift"]) == 0 and int(row["label_shift"]) == 0:
                    actual_contents, char_spans = _source_selected_sites(
                        parsed, source, candidates
                    )
                else:
                    replayed, actual_contents, char_spans = _transform_with_selected_sites(
                        parsed,
                        source,
                        candidate_texts=candidates,
                        position_shift=int(row["position_shift"]),
                        label_shift=int(row["label_shift"]),
                    )
                    if replayed != prompt:
                        raise ValueError(
                            f"{row['work_key']} does not match the audited deterministic transform"
                        )
            for content_id, (start, end) in zip(actual_contents, char_spans, strict=True):
                if not prompt[start:end] or prompt[start:end] not in candidates[content_id]:
                    raise ValueError(f"{row['work_key']} candidate span does not match displayed content")
            stored_contents = [
                int(value)
                for value in _sequence(row["content_ids_by_position"], name="content_ids_by_position")
            ]
            winner_position = int(row["winner_position"])
            if winner_position not in range(4):
                raise ValueError(f"{row['work_key']} has an invalid winner position")
            output.append(
                {
                    **row,
                    "actual_content_ids_by_position": actual_contents,
                    "content_char_spans": char_spans,
                    "mapping_matches_ledger": actual_contents == stored_contents,
                    "actual_winner_content_id": actual_contents[winner_position],
                    "content_target_evaluable": bool(row["winner_unique"])
                    and not bool(row["text_identity_ambiguous"]),
                }
            )
    return pd.DataFrame(output).sort_values("work_key", kind="mergesort").reset_index(drop=True)


def locate_content_token_indices(
    tokenizer,
    prompt: str,
    char_spans: Sequence[tuple[int, int]],
) -> list[int]:
    """Return the final full token contained in each audited candidate payload."""

    try:
        encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    except (TypeError, NotImplementedError) as exc:
        raise ValueError("tokenizer does not provide offset mappings") from exc
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded or "offset_mapping" not in encoded:
        raise ValueError("tokenizer does not provide input IDs and offset mappings")
    input_ids = list(encoded["input_ids"])
    offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"]]
    if len(input_ids) != len(offsets) or input_ids != tokenize_text(tokenizer, prompt):
        raise ValueError("tokenizer input IDs and offset mapping disagree")
    indices: list[int] = []
    for start, end in char_spans:
        if start < 0 or end <= start or end > len(prompt):
            raise ValueError("invalid candidate payload span")
        overlapping = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_end > token_start and token_start < end and token_end > start
        ]
        if not overlapping:
            raise ValueError("candidate payload has no tokenizer token")
        index = overlapping[-1]
        token_start, token_end = offsets[index]
        if token_start < start or token_end > end:
            raise ValueError("token crosses candidate payload boundary")
        indices.append(index)
    if len(indices) != len(char_spans):
        raise AssertionError("candidate token endpoint count drift")
    return indices
