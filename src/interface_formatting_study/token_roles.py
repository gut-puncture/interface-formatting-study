from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .scoring import tokenize_text
from .utils import LABELS, validate_label
from .vanilla import VANILLA_ANSWER_INSTRUCTION, VANILLA_ANSWER_PREFIX, choices_by_label

ROLE_QUESTION = "question"
ROLE_OPTIONS = tuple(f"option_{label}" for label in LABELS)
ROLE_WRAPPER_SYNTAX = "wrapper_syntax"
ROLE_INSTRUCTION = "instruction"
ROLE_ANSWER_PREFIX = "answer_prefix"
TOKEN_ROLES = (
    ROLE_QUESTION,
    *ROLE_OPTIONS,
    ROLE_WRAPPER_SYNTAX,
    ROLE_INSTRUCTION,
    ROLE_ANSWER_PREFIX,
)
CONTENT_ROLES = frozenset((ROLE_QUESTION, *ROLE_OPTIONS))
OPTION_ROLE_BY_LABEL = {label: f"option_{label}" for label in LABELS}


@dataclass(frozen=True)
class TokenRoleLabeling:
    token_ids: list[int]
    token_roles: list[str]
    role_positions: dict[str, list[int]]
    char_spans_by_role: dict[str, list[tuple[int, int]]]


def _find_all(prompt: str, needle: str, *, ignore_case: bool = False) -> list[tuple[int, int]]:
    if not needle:
        return []
    haystack = prompt.lower() if ignore_case else prompt
    target = needle.lower() if ignore_case else needle
    hits: list[tuple[int, int]] = []
    start = haystack.find(target)
    while start >= 0:
        hits.append((start, start + len(needle)))
        start = haystack.find(target, start + 1)
    return hits


def _span_positions_with_offsets(tokenizer, prompt: str, start: int, end: int) -> list[int] | None:
    try:
        encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    except TypeError:
        return None
    offsets = encoded.get("offset_mapping") if isinstance(encoded, dict) else None
    if offsets is None:
        return None
    positions: list[int] = []
    for position, offset in enumerate(offsets):
        token_start, token_end = int(offset[0]), int(offset[1])
        if token_end <= token_start:
            continue
        if token_start < end and token_end > start:
            positions.append(position)
    return positions


def char_span_to_token_positions(tokenizer, prompt: str, start: int, end: int) -> list[int]:
    if start < 0 or end < start or end > len(prompt):
        raise ValueError(f"invalid character span {(start, end)} for prompt length {len(prompt)}")
    offset_positions = _span_positions_with_offsets(tokenizer, prompt, start, end)
    if offset_positions is not None:
        return offset_positions
    start_token = len(tokenize_text(tokenizer, prompt[:start]))
    end_token = len(tokenize_text(tokenizer, prompt[:end]))
    positions = list(range(start_token, end_token))
    if not positions and end > start and end_token > 0:
        positions = [end_token - 1]
    return positions


def _question_spans(prompt: str, question: object) -> list[tuple[int, int]]:
    if question is None:
        return []
    text = str(question)
    hits = _find_all(prompt, text)
    if not hits:
        hits = _find_all(prompt, text, ignore_case=True)
    return hits


def _option_context_score(prompt: str, start: int, label: str) -> int:
    label = validate_label(label)
    before = prompt[max(0, start - 180) : start]
    near = before[-90:]
    score = 0
    if f"OPTION_{label}" in before:
        score += 45
    if re.search(rf"\bletter\s*:\s*['\"]{label}['\"]", before):
        score += 40
    if re.search(rf"\bvalue\s*=\s*['\"]{label}['\"]", before):
        score += 40
    if f",{label}," in before:
        score += 35
    if re.search(rf"(?<![A-Za-z0-9_]){label}\s*[\)\]=:\-]", near):
        score += 30
    if re.search(rf"(?<![A-Za-z0-9_]){label}\s*$", near.strip()):
        score += 10
    if re.search(rf"\boption\W+value\s*=\s*['\"]{label}['\"]", before, flags=re.IGNORECASE):
        score += 25
    return score


def _option_spans(prompt: str, label: str, choice: str) -> list[tuple[int, int]]:
    hits = _find_all(prompt, choice)
    if not hits:
        hits = _find_all(prompt, choice, ignore_case=True)
    if not hits:
        return []
    scored = [(span, _option_context_score(prompt, span[0], label)) for span in hits]
    meaningful = [span for span, score in scored if score >= 20]
    if meaningful:
        return meaningful
    if len(choice) >= 4:
        return hits
    best_score = max(score for _, score in scored)
    return [span for span, score in scored if score == best_score][:1]


def canonical_char_spans(prompt: str, question: object, choices: object) -> dict[str, list[tuple[int, int]]]:
    spans: dict[str, list[tuple[int, int]]] = {role: [] for role in TOKEN_ROLES}
    spans[ROLE_QUESTION] = _question_spans(prompt, question)
    try:
        options = choices_by_label(choices)
    except ValueError:
        options = {}
    for label, choice in options.items():
        spans[OPTION_ROLE_BY_LABEL[label]] = _option_spans(prompt, label, choice)
    spans[ROLE_INSTRUCTION] = _find_all(prompt, VANILLA_ANSWER_INSTRUCTION)
    spans[ROLE_ANSWER_PREFIX] = _find_all(prompt, VANILLA_ANSWER_PREFIX)
    return spans


def _apply_spans(
    *,
    tokenizer,
    prompt: str,
    roles: list[str],
    spans: Iterable[tuple[str, tuple[int, int]]],
) -> None:
    for role, (start, end) in spans:
        for position in char_span_to_token_positions(tokenizer, prompt, start, end):
            if 0 <= position < len(roles):
                roles[position] = role


def label_token_roles(tokenizer, prompt: str, *, question: object = None, choices: object = None) -> TokenRoleLabeling:
    token_ids = tokenize_text(tokenizer, prompt)
    token_roles = [ROLE_WRAPPER_SYNTAX for _ in token_ids]
    char_spans = canonical_char_spans(prompt, question, choices)

    content_spans: list[tuple[str, tuple[int, int]]] = []
    for role in (ROLE_QUESTION, *ROLE_OPTIONS):
        content_spans.extend((role, span) for span in char_spans[role])
    _apply_spans(tokenizer=tokenizer, prompt=prompt, roles=token_roles, spans=content_spans)

    instruction_spans = [(ROLE_INSTRUCTION, span) for span in char_spans[ROLE_INSTRUCTION]]
    _apply_spans(tokenizer=tokenizer, prompt=prompt, roles=token_roles, spans=instruction_spans)

    answer_spans = [(ROLE_ANSWER_PREFIX, span) for span in char_spans[ROLE_ANSWER_PREFIX]]
    _apply_spans(tokenizer=tokenizer, prompt=prompt, roles=token_roles, spans=answer_spans)

    role_positions = {role: [] for role in TOKEN_ROLES}
    for position, role in enumerate(token_roles):
        role_positions.setdefault(role, []).append(position)
    return TokenRoleLabeling(
        token_ids=token_ids,
        token_roles=token_roles,
        role_positions=role_positions,
        char_spans_by_role=char_spans,
    )


def combined_role_positions(labeling: TokenRoleLabeling, roles: Iterable[str]) -> list[int]:
    positions: list[int] = []
    for role in roles:
        positions.extend(labeling.role_positions.get(role, []))
    return sorted(set(positions))
