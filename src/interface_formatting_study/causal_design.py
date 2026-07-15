from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from .calibration import SAME_WRAPPER_REDACTION, make_content_free_prompt_with_metadata
from .causal_option_maps import (
    ParsedPrompt,
    parse_prompt_options,
    remap_option_map_by_redaction_diff,
    remap_option_map_through_canonical_redaction,
    transform_with_option_map,
)
from .vanilla import build_vanilla_prompt


ACTIVE_WRAPPERS = (
    "csv_inline",
    "graphql_query",
    "html_form",
    "ini_file",
    "key_equals",
    "protobuf_msg",
    "shell_heredoc",
    "toml_config",
)
CONTROLLED_FORMATS = ("plain", *ACTIVE_WRAPPERS)
LETTERS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class Assignment:
    variant: int
    manipulation: str
    content_ids_by_position: tuple[int, int, int, int]
    labels_by_position: tuple[str, str, str, str]
    position_shift: int
    label_shift: int


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    value: str


_LETTER_INSTRUCTION = "Return only the letter (A, B, C, or D)."
_TEXT_INSTRUCTION = "Return only the exact answer text, not its letter."


_LABEL_PATTERNS: dict[str, tuple[str, ...]] = {
    "csv_inline": (
        r"(?m)^(?:[^,\n]*,)?[ \t]*(?P<label>[A-D])(?=[ \t]*,)",
        r"(?<=,)(?P<label>[A-D])(?=,)",
        r"(?m)\bOption[ \t]+(?P<label>[A-D])(?=[ \t]*(?:,|$))",
        r"(?<![A-Za-z0-9_])(?P<label>[A-D])(?=\))",
    ),
    "graphql_query": (
        r'\bletter\s*:\s*"(?P<label>[A-D])"',
        r'\blabel\s*:\s*"(?P<label>[A-D])"',
        r"\bchoice\s*:\s*(?P<label>[A-D])(?![A-Za-z0-9_])",
        r'\boption\s*:\s*"(?P<label>[A-D])"',
        r"\boption(?P<label>[A-D])(?=\s*:)",
        r'(?<![A-Za-z0-9_])(?P<label>[A-D])(?=\s*:\s*")',
    ),
    "html_form": (
        r'\bvalue=["\'](?P<label>[A-D])["\']',
        r"(?<=>|[ \t])(?P<label>[A-D])(?=\)|:)",
    ),
    "ini_file": (
        r'(?mi)^[ \t]*(?:option_|choice_)?(?P<label>[A-D])(?=[ \t]*=)',
    ),
    "key_equals": (
        r'(?mi)(?<![A-Za-z0-9_])(?:OPTION_)?(?P<label>[A-D])(?=[ \t]*=)',
        r"(?mi)\bOPTIONS_(?P<label>[A-D])(?==)",
        r"(?m)(?<![A-Za-z0-9_])(?P<label>[A-D])(?=\))",
    ),
    "protobuf_msg": (
        r"(?i)\bOPTION_(?P<label>[A-D])\b",
        r"(?i)\boption_(?P<label>[A-D])\b",
        r"(?i)\boption(?P<label>[A-D])\b",
        r'\blabel\s*:\s*"(?P<label>[A-D])"',
        r'(?<=["\'])(?P<label>[A-D])(?=\))',
        r"\b(?:Option|option)[ \t]+(?P<label>[A-D])(?=\s*:)",
        r"(?m)^[ \t]*(?P<label>[A-D])(?=[ \t]*=)",
        r"(?m)//[ \t]*(?P<label>[A-D])(?=\))",
    ),
    "shell_heredoc": (
        r"(?m)^[ \t]*(?P<label>[A-D])(?=\))",
        r"(?<=[\"'])(?P<label>[A-D])(?=\))",
        r"(?m)^[ \t]*(?P<label>[A-D])(?==)",
        r'\bANSWER=["\'](?P<label>[A-D])["\'](?=[ \t]*#)',
    ),
    "toml_config": (
        r'(?mi)^[ \t]*(?:option_|choice_|option\.|["\'])(?P<label>[A-D])(?:["\'])?(?=[ \t]*=)',
        r"(?mi)^[ \t]*(?P<label>[A-D])(?=[ \t]*=)",
    ),
    "plain": (r"(?m)^[ \t]*(?P<label>[A-D])(?=\))",),
}


def _raw_label_spans(prompt: str, wrapper_name: str) -> tuple[_Span, ...]:
    patterns = _LABEL_PATTERNS.get(wrapper_name)
    if patterns is None:
        raise ValueError(f"unknown_source_wrapper:{wrapper_name}")
    option_region = prompt.split(f"\n\n{_LETTER_INSTRUCTION}", 1)[0]
    instruction_lists = [
        match.span()
        for match in re.finditer(
            r"\(\s*A\s*[,/]\s*B\s*[,/]\s*C\s*(?:,\s*or|[,/])\s*D\s*\)",
            option_region,
            flags=re.IGNORECASE,
        )
    ]
    by_position: dict[tuple[int, int], _Span] = {}
    for pattern in patterns:
        for match in re.finditer(pattern, option_region):
            start, end = match.span("label")
            if any(list_start <= start < list_end for list_start, list_end in instruction_lists):
                continue
            by_position[(start, end)] = _Span(start, end, match.group("label").upper())
    return tuple(sorted(by_position.values(), key=lambda span: span.start))


def _html_label_spans(prompt: str, choices: Sequence[str]) -> tuple[_Span, ...]:
    option_region = prompt.split(f"\n\n{_LETTER_INSTRUCTION}", 1)[0]
    candidates = _raw_label_spans(prompt, "html_form")
    containers = list(
        re.finditer(
            r"(?is)<(?:option|label|div)\b[^>]*>.*?</(?:option|label|div)>",
            option_region,
        )
    )
    selected: list[_Span] = []
    for label, choice in zip(LETTERS, map(str, choices), strict=True):
        matches: dict[tuple[int, int], tuple[tuple[int, int], list[_Span]]] = {}
        for occurrence in re.finditer(re.escape(choice), option_region):
            for container in containers:
                if container.start() <= occurrence.start() and occurrence.end() <= container.end():
                    labels = [
                        span
                        for span in candidates
                        if container.start() <= span.start and span.end <= container.end()
                    ]
                    if labels and {span.value for span in labels} == {label}:
                        score = (len(labels), -(container.end() - container.start()))
                        matches[container.span()] = (score, labels)
        if not matches:
            raise ValueError("option_labels_not_unambiguous")
        best_score = max(score for score, _ in matches.values())
        best = [labels for score, labels in matches.values() if score == best_score]
        if len(best) != 1:
            raise ValueError("option_labels_not_unambiguous")
        selected.extend(best[0])
    unique = {(span.start, span.end): span for span in selected}
    return tuple(sorted(unique.values(), key=lambda span: span.start))


def _label_spans(
    prompt: str, wrapper_name: str, choices: Sequence[str]
) -> tuple[_Span, ...]:
    if wrapper_name == "html_form":
        return _html_label_spans(prompt, choices)
    spans = _raw_label_spans(prompt, wrapper_name)
    counts = {label: sum(span.value == label for span in spans) for label in LETTERS}
    per_label = next(iter(counts.values())) if len(set(counts.values())) == 1 else 0
    if not spans or per_label != 1:
        raise ValueError("option_labels_not_unambiguous")
    return spans


def _text_spans(
    prompt: str, choices: Sequence[str], label_spans: Sequence[_Span]
) -> tuple[_Span, ...]:
    if len(choices) != 4 or len(set(map(str, choices))) != 4:
        raise ValueError("option_texts_not_unambiguous")
    first_label_position = {
        label: min(span.start for span in label_spans if span.value == label) for label in LETTERS
    }
    option_region_end = prompt.find(f"\n\n{_LETTER_INSTRUCTION}")
    if option_region_end < 0:
        option_region_end = len(prompt)
    spans: list[_Span] = []
    for index, choice in enumerate(map(str, choices)):
        occurrences = list(re.finditer(re.escape(choice), prompt))
        if len(occurrences) > 1:
            lower = first_label_position[LETTERS[index]]
            upper = (
                first_label_position[LETTERS[index + 1]]
                if index + 1 < len(LETTERS)
                else option_region_end
            )
            occurrences = [match for match in occurrences if lower <= match.start() < upper]
        if len(occurrences) != 1:
            raise ValueError("option_texts_not_unambiguous")
        match = occurrences[0]
        spans.append(_Span(match.start(), match.end(), choice))
    if len({(span.start, span.end) for span in spans}) != 4:
        raise ValueError("option_texts_not_unambiguous")
    return tuple(spans)


def _replace_spans(prompt: str, replacements: Sequence[tuple[_Span, str]]) -> str:
    ordered = sorted(replacements, key=lambda pair: pair[0].start)
    parts: list[str] = []
    cursor = 0
    for span, replacement in ordered:
        if span.start < cursor:
            raise ValueError("option_spans_overlap")
        parts.extend((prompt[cursor : span.start], replacement))
        cursor = span.end
    parts.append(prompt[cursor:])
    return "".join(parts)


def _source_counterfactual(
    prompt: str,
    wrapper_name: str,
    assignment: Assignment,
    choices: Sequence[str],
) -> str:
    if assignment.manipulation == "controlled_baseline":
        return prompt
    if assignment.manipulation not in {"position_only", "label_only"}:
        raise ValueError(f"Unknown manipulation {assignment.manipulation!r}")
    label_spans = _label_spans(prompt, wrapper_name, choices)
    replacements = [
        (span, assignment.labels_by_position[LETTERS.index(span.value)]) for span in label_spans
    ]
    if assignment.manipulation == "position_only":
        replacements.extend(
            (span, str(choices[assignment.content_ids_by_position[index]]))
            for index, span in enumerate(_text_spans(prompt, choices, label_spans))
        )
    return _replace_spans(prompt, replacements)


def _text_readout_prompt(prompt: str) -> str:
    if prompt.count(_LETTER_INSTRUCTION) != 1:
        raise ValueError("Source prompt must contain exactly one terminal letter instruction")
    return prompt.replace(_LETTER_INSTRUCTION, _TEXT_INSTRUCTION)


def _assignment(variant: int, manipulation: str, position_shift: int, label_shift: int) -> Assignment:
    content_by_position = [0, 0, 0, 0]
    for content_id in range(4):
        content_by_position[(content_id + position_shift) % 4] = content_id
    labels_by_position = [
        LETTERS[(content_id + label_shift) % 4] for content_id in content_by_position
    ]
    return Assignment(
        variant=variant,
        manipulation=manipulation,
        content_ids_by_position=tuple(content_by_position),
        labels_by_position=tuple(labels_by_position),
        position_shift=position_shift,
        label_shift=label_shift,
    )


def balanced_assignments() -> tuple[Assignment, ...]:
    """Minimal balanced block that changes physical position and letter separately."""

    rows = [_assignment(0, "controlled_baseline", 0, 0)]
    rows.extend(_assignment(shift, "position_only", shift, 0) for shift in range(1, 4))
    rows.extend(_assignment(shift + 3, "label_only", 0, shift) for shift in range(1, 4))
    return tuple(rows)


def _canonical_items(frame: pd.DataFrame) -> list[dict[str, object]]:
    required = {
        "item_id",
        "subject",
        "split",
        "question",
        "choices",
        "correct_index",
        "wrapper_name",
        "wrapped_prompt",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Causal design input is missing columns: {sorted(missing)}")
    items: list[dict[str, object]] = []
    for item_id, group in frame.groupby("item_id", sort=True):
        wrapper_rows = group[group["wrapper_name"].isin(ACTIVE_WRAPPERS)].copy()
        if set(wrapper_rows["wrapper_name"]) != set(ACTIVE_WRAPPERS) or len(wrapper_rows) != len(ACTIVE_WRAPPERS):
            raise ValueError(f"Item {item_id} does not have exactly one row for every active wrapper")
        records = wrapper_rows.to_dict("records")
        canonical = records[0]
        for row in records[1:]:
            if any(
                row[key] != canonical[key]
                for key in ("subject", "split", "question", "choices", "correct_index")
            ):
                raise ValueError(f"Canonical fields disagree across wrappers for item {item_id}")
        canonical = dict(canonical)
        canonical["source_prompts"] = {
            str(row["wrapper_name"]): str(row["wrapped_prompt"]) for row in records
        }
        items.append(canonical)
    return items


def _format_sources(
    item: dict[str, object], format_name: str
) -> tuple[str, dict[str, object], str]:
    choices = [str(choice) for choice in item["choices"]]
    source_prompt = (
        build_vanilla_prompt(item["question"], choices)
        if format_name == "plain"
        else str(item["source_prompts"][format_name])
    )
    calibration = make_content_free_prompt_with_metadata(
        {
            "wrapped_prompt": source_prompt,
            "question": item["question"],
            "choices": choices,
        }
    )
    calibration_wrapper = (
        format_name
        if calibration["content_free_calibration_kind"] == "same_wrapper_redaction"
        else "key_equals"
    )
    return source_prompt, calibration, calibration_wrapper


def _exclusion_reasons(item: dict[str, object]) -> list[dict[str, str]]:
    choices = [str(choice) for choice in item["choices"]]
    placeholders = [f"OPTION_{label}_PLACEHOLDER" for label in LETTERS]
    reasons: list[dict[str, str]] = []
    assignments = balanced_assignments()
    for format_name in CONTROLLED_FORMATS:
        try:
            source_prompt, calibration, calibration_wrapper = _format_sources(item, format_name)
            _text_readout_prompt(source_prompt)
            source_calibration = str(calibration["content_free_prompt"])
            for assignment in assignments[1:]:
                _source_counterfactual(source_prompt, format_name, assignment, choices)
                _source_counterfactual(
                    source_calibration,
                    calibration_wrapper,
                    assignment,
                    placeholders,
                )
        except ValueError as exc:
            reason = str(exc)
            if reason.startswith("unknown_source_wrapper:"):
                reason = "unknown_source_wrapper"
            reasons.append(
                {
                    "item_id": str(item["item_id"]),
                    "split": str(item["split"]),
                    "subject": str(item["subject"]),
                    "correct_label": LETTERS[int(item["correct_index"])],
                    "wrapper_name": format_name,
                    "reason": reason,
                }
            )
    return reasons


def _build_design_from_items(items: Sequence[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    assignments = balanced_assignments()
    placeholders = [f"OPTION_{label}_PLACEHOLDER" for label in LETTERS]
    for item in items:
        item_id = str(item["item_id"])
        question = str(item["question"])
        choices = [str(choice) for choice in item["choices"]]
        correct_content_id = int(item["correct_index"])
        for format_name in CONTROLLED_FORMATS:
            source_prompt, calibration, calibration_wrapper = _format_sources(item, format_name)
            source_prompt_sha = hashlib.sha256(source_prompt.encode("utf-8")).hexdigest()
            source_calibration = str(calibration["content_free_prompt"])
            for assignment in assignments:
                prompt = _source_counterfactual(source_prompt, format_name, assignment, choices)
                calibration_prompt = _source_counterfactual(
                    source_calibration,
                    calibration_wrapper,
                    assignment,
                    placeholders,
                )
                correct_position = assignment.content_ids_by_position.index(correct_content_id)
                correct_label = assignment.labels_by_position[correct_position]
                rows.append(
                    {
                        "work_key": f"letter|{item_id}|{format_name}|{assignment.variant}",
                        "item_id": item_id,
                        "subject": item["subject"],
                        "split": item["split"],
                        "wrapper_name": format_name,
                        "arm": "letter_intervention",
                        "variant": assignment.variant,
                        "manipulation": assignment.manipulation,
                        "position_shift": assignment.position_shift,
                        "label_shift": assignment.label_shift,
                        "template_scope": "source_prompt_counterfactual",
                        "source_prompt_sha256": source_prompt_sha,
                        "prompt": prompt,
                        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                        "calibration_prompt": calibration_prompt,
                        "calibration_prompt_sha256": hashlib.sha256(
                            calibration_prompt.encode("utf-8")
                        ).hexdigest(),
                        "content_free_calibration_kind": calibration[
                            "content_free_calibration_kind"
                        ],
                        "content_free_fallback_reason": calibration[
                            "content_free_fallback_reason"
                        ],
                        "content_ids_by_position": list(assignment.content_ids_by_position),
                        "labels_by_position": list(assignment.labels_by_position),
                        "candidate_texts": choices,
                        "correct_content_id": correct_content_id,
                        "correct_position": correct_position,
                        "correct_label": correct_label,
                        "correct_text": choices[correct_content_id],
                    }
                )
            identity = assignments[0]
            text_prompt = _text_readout_prompt(source_prompt)
            rows.append(
                {
                    "work_key": f"text|{item_id}|{format_name}|0",
                    "item_id": item_id,
                    "subject": item["subject"],
                    "split": item["split"],
                    "wrapper_name": format_name,
                    "arm": "answer_text",
                    "variant": 0,
                    "manipulation": "controlled_baseline",
                    "position_shift": 0,
                    "label_shift": 0,
                    "template_scope": "source_prompt_counterfactual",
                    "source_prompt_sha256": source_prompt_sha,
                    "prompt": text_prompt,
                    "prompt_sha256": hashlib.sha256(text_prompt.encode("utf-8")).hexdigest(),
                    "calibration_prompt": None,
                    "calibration_prompt_sha256": None,
                    "content_free_calibration_kind": None,
                    "content_free_fallback_reason": None,
                    "content_ids_by_position": list(identity.content_ids_by_position),
                    "labels_by_position": list(identity.labels_by_position),
                    "candidate_texts": choices,
                    "correct_content_id": correct_content_id,
                    "correct_position": correct_content_id,
                    "correct_label": LETTERS[correct_content_id],
                    "correct_text": choices[correct_content_id],
                }
            )
    design = pd.DataFrame(rows)
    if design.empty:
        return design
    if not design["work_key"].is_unique:
        raise AssertionError("Causal design produced duplicate work keys")
    return design.sort_values("work_key", kind="mergesort").reset_index(drop=True)


def build_causal_design_with_exclusions(
    frame: pd.DataFrame, *, splits: Sequence[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    allowed_splits = None if splits is None else {str(split) for split in splits}
    eligible: list[dict[str, object]] = []
    exclusions: list[dict[str, str]] = []
    for item in _canonical_items(frame):
        if allowed_splits is not None and str(item["split"]) not in allowed_splits:
            continue
        item_reasons = _exclusion_reasons(item)
        if item_reasons:
            exclusions.extend(item_reasons)
        else:
            eligible.append(item)
    exclusion_frame = pd.DataFrame(
        exclusions,
        columns=["item_id", "split", "subject", "correct_label", "wrapper_name", "reason"],
    ).sort_values(["item_id", "wrapper_name"], kind="mergesort", ignore_index=True)
    return _build_design_from_items(eligible), exclusion_frame


def build_causal_design(frame: pd.DataFrame, *, splits: Sequence[str] | None = None) -> pd.DataFrame:
    design, exclusions = build_causal_design_with_exclusions(frame, splits=splits)
    if not exclusions.empty:
        raise ValueError(
            "Some items are not safely transformable; use "
            "build_causal_design_with_exclusions to retain the exclusion record"
        )
    return design


def _v3_rows_for_item_format(
    item: dict[str, object],
    format_name: str,
    parsed_source: ParsedPrompt | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    item_id = str(item["item_id"])
    choices = [str(choice) for choice in item["choices"]]
    correct_content_id = int(item["correct_index"])
    source_prompt, calibration, calibration_wrapper = _format_sources(item, format_name)
    source_prompt_sha = hashlib.sha256(source_prompt.encode("utf-8")).hexdigest()
    source_calibration = str(calibration["content_free_prompt"])
    placeholders = [f"OPTION_{label}_PLACEHOLDER" for label in LETTERS]
    parse_reason = ""
    if parsed_source is None:
        try:
            parsed_source = parse_prompt_options(source_prompt, format_name, choices)
        except ValueError as exc:
            parse_reason = str(exc)
    parsed_calibration = None
    if parsed_source is not None and parsed_source.separable:
        try:
            parsed_calibration = parse_prompt_options(
                source_calibration,
                calibration_wrapper,
                placeholders,
            )
        except ValueError as exc:
            parse_reason = f"calibration:{exc}"
            if calibration["content_free_calibration_kind"] == SAME_WRAPPER_REDACTION:
                try:
                    parsed_calibration = remap_option_map_through_canonical_redaction(
                        parsed_source,
                        source_prompt,
                        question=str(item["question"]),
                        choices=choices,
                        redacted=source_calibration,
                    )
                except ValueError:
                    try:
                        parsed_calibration = remap_option_map_by_redaction_diff(
                            parsed_source,
                            source_prompt,
                            source_calibration,
                        )
                    except ValueError:
                        pass
    separable = bool(
        parsed_source is not None
        and parsed_source.separable
        and parsed_calibration is not None
        and parsed_calibration.separable
    )
    transform_provenance = parsed_source.provenance if parsed_source is not None else "unresolved"
    legacy_prompts: dict[int, tuple[str, str]] = {}
    if parsed_source is not None and not parsed_source.separable:
        parse_reason = parsed_source.not_applicable_reason
    elif not separable:
        try:
            legacy_prompts = {
                assignment.variant: (
                    _source_counterfactual(source_prompt, format_name, assignment, choices),
                    _source_counterfactual(
                        source_calibration,
                        calibration_wrapper,
                        assignment,
                        placeholders,
                    ),
                )
                for assignment in balanced_assignments()[1:]
            }
        except ValueError:
            legacy_prompts = {}
        else:
            separable = True
            transform_provenance = "legacy_deterministic"
    if not separable and not parse_reason:
        parse_reason = "option_structure_not_resolved"

    rows: list[dict[str, object]] = []

    def append_letter(assignment: Assignment, prompt: str, calibration_prompt: str) -> None:
        correct_position = assignment.content_ids_by_position.index(correct_content_id)
        correct_label = assignment.labels_by_position[correct_position]
        rows.append(
            {
                "work_key": f"letter|{item_id}|{format_name}|{assignment.variant}",
                "item_id": item_id,
                "subject": item["subject"],
                "split": item["split"],
                "wrapper_name": format_name,
                "arm": "letter_intervention",
                "variant": assignment.variant,
                "manipulation": assignment.manipulation,
                "position_shift": assignment.position_shift,
                "label_shift": assignment.label_shift,
                "template_scope": "source_prompt_counterfactual",
                "source_prompt_sha256": source_prompt_sha,
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "calibration_prompt": calibration_prompt,
                "calibration_prompt_sha256": hashlib.sha256(calibration_prompt.encode("utf-8")).hexdigest(),
                "content_free_calibration_kind": calibration["content_free_calibration_kind"],
                "content_free_fallback_reason": calibration["content_free_fallback_reason"],
                "content_ids_by_position": list(assignment.content_ids_by_position),
                "labels_by_position": list(assignment.labels_by_position),
                "candidate_texts": choices,
                "correct_content_id": correct_content_id,
                "correct_position": correct_position,
                "correct_label": correct_label,
                "correct_text": choices[correct_content_id],
            }
        )

    assignments = balanced_assignments()
    append_letter(assignments[0], source_prompt, source_calibration)
    if separable:
        for assignment in assignments[1:]:
            if legacy_prompts:
                prompt, calibration_prompt = legacy_prompts[assignment.variant]
            else:
                assert parsed_source is not None and parsed_calibration is not None
                prompt = transform_with_option_map(
                    parsed_source,
                    source_prompt,
                    position_shift=assignment.position_shift,
                    label_shift=assignment.label_shift,
                )
                calibration_prompt = transform_with_option_map(
                    parsed_calibration,
                    source_calibration,
                    position_shift=assignment.position_shift,
                    label_shift=assignment.label_shift,
                )
            append_letter(assignment, prompt, calibration_prompt)

    text_prompt = _text_readout_prompt(source_prompt)
    rows.append(
        {
            "work_key": f"text|{item_id}|{format_name}|0",
            "item_id": item_id,
            "subject": item["subject"],
            "split": item["split"],
            "wrapper_name": format_name,
            "arm": "answer_text",
            "variant": 0,
            "manipulation": "controlled_baseline",
            "position_shift": 0,
            "label_shift": 0,
            "template_scope": "source_prompt_counterfactual",
            "source_prompt_sha256": source_prompt_sha,
            "prompt": text_prompt,
            "prompt_sha256": hashlib.sha256(text_prompt.encode("utf-8")).hexdigest(),
            "calibration_prompt": None,
            "calibration_prompt_sha256": None,
            "content_free_calibration_kind": None,
            "content_free_fallback_reason": None,
            "content_ids_by_position": [0, 1, 2, 3],
            "labels_by_position": list(LETTERS),
            "candidate_texts": choices,
            "correct_content_id": correct_content_id,
            "correct_position": correct_content_id,
            "correct_label": LETTERS[correct_content_id],
            "correct_text": choices[correct_content_id],
        }
    )
    applicability = {
        "item_id": item_id,
        "subject": item["subject"],
        "split": item["split"],
        "correct_label": LETTERS[correct_content_id],
        "wrapper_name": format_name,
        "source_prompt_sha256": source_prompt_sha,
        "baseline_applicable": True,
        "text_applicable": True,
        "position_applicable": separable,
        "label_applicable": separable,
        "parse_provenance": transform_provenance,
        "not_applicable_reason": "" if separable else parse_reason,
    }
    return rows, applicability


def build_causal_design_v3(
    frame: pd.DataFrame,
    *,
    splits: Sequence[str] | None = None,
    option_maps: dict[tuple[str, str], ParsedPrompt] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the full-population design without whole-item parser exclusions."""

    allowed_splits = None if splits is None else {str(split) for split in splits}
    rows: list[dict[str, object]] = []
    applicability: list[dict[str, object]] = []
    supplied = option_maps or {}
    for item in _canonical_items(frame):
        if allowed_splits is not None and str(item["split"]) not in allowed_splits:
            continue
        for format_name in CONTROLLED_FORMATS:
            block, status = _v3_rows_for_item_format(
                item,
                format_name,
                supplied.get((str(item["item_id"]), format_name)),
            )
            rows.extend(block)
            applicability.append(status)
    design = pd.DataFrame(rows).sort_values("work_key", kind="mergesort").reset_index(drop=True)
    if design["work_key"].duplicated().any():
        raise AssertionError("Causal v3 design produced duplicate work keys")
    applicability_frame = pd.DataFrame(applicability).sort_values(
        ["item_id", "wrapper_name"], kind="mergesort", ignore_index=True
    )
    return design, applicability_frame
