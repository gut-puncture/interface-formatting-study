from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from .scoring import tokenize_text
from .utils import LABELS, validate_label

CONTENT_SUFFIX = "Return only the letter"
LEAKY_ANCHORS = {"answer_anchor"}
SEMANTIC_ANCHORS = {
    "question_end",
    "option_A_end",
    "option_B_end",
    "option_C_end",
    "option_D_end",
    "options_end",
}
ANCHOR_GROUPS = {
    "question_end+options_end": ("question_end", "options_end"),
    "all_option_ends": ("option_A_end", "option_B_end", "option_C_end", "option_D_end"),
}


@dataclass(frozen=True)
class AnchorPosition:
    name: str
    position: int


def answer_anchor(tokenizer, prompt: str) -> int:
    prompt_ids = tokenize_text(tokenizer, prompt)
    if not prompt_ids:
        raise ValueError("Prompt has no tokens")
    return len(prompt_ids) - 1


def content_end_anchor(tokenizer, prompt: str, *, suffix: str = CONTENT_SUFFIX) -> int | None:
    lower_prompt = prompt.lower()
    idx = lower_prompt.rfind(suffix.lower())
    if idx < 0:
        return None
    prefix = prompt[:idx].rstrip()
    prefix_ids = tokenize_text(tokenizer, prefix)
    if not prefix_ids:
        return None
    return len(prefix_ids) - 1


def _choices_by_label(choices: object) -> dict[str, str] | None:
    if choices is None:
        return None
    if isinstance(choices, str):
        try:
            parsed = json.loads(choices)
        except json.JSONDecodeError:
            return None
        choices = parsed
    if hasattr(choices, "tolist"):
        choices = choices.tolist()
    if isinstance(choices, Mapping):
        out = {label: str(choices.get(label, "")) for label in LABELS}
    elif isinstance(choices, Sequence) and not isinstance(choices, (bytes, bytearray, str)):
        values = list(choices)
        if len(values) != len(LABELS):
            return None
        out = {label: str(value) for label, value in zip(LABELS, values, strict=True)}
    else:
        return None
    if any(not value for value in out.values()):
        return None
    return out


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


def _end_token_position(tokenizer, prompt: str, end_char: int) -> int | None:
    prefix = prompt[:end_char]
    ids = tokenize_text(tokenizer, prefix)
    if not ids:
        return None
    return len(ids) - 1


def _question_char_end(prompt: str, question: object) -> int | None:
    if not question:
        return None
    text = str(question)
    hits = _find_all(prompt, text)
    if not hits:
        hits = _find_all(prompt, text, ignore_case=True)
    if not hits:
        return None
    return hits[0][1]


def question_end_anchor(tokenizer, prompt: str, *, question: object = None) -> int | None:
    end_char = _question_char_end(prompt, question)
    if end_char is None:
        return None
    return _end_token_position(tokenizer, prompt, end_char)


def _option_context_score(prompt: str, start: int, label: str, *, question_end_char: int | None) -> int:
    before = prompt[max(0, start - 160) : start]
    near = before[-80:]
    score = 0
    if question_end_char is not None:
        score += 20 if start >= question_end_char else -20
    if f"OPTION_{label}" in before:
        score += 40
    if re.search(rf"\bletter\s*:\s*['\"]{label}['\"]", before):
        score += 35
    if re.search(rf"\bvalue\s*=\s*['\"]{label}['\"]", before):
        score += 35
    if f",{label}," in before:
        score += 30
    if re.search(rf"(?<![A-Za-z0-9_]){label}\s*[\)\]=:\-]", near):
        score += 25
    if re.search(rf"(?<![A-Za-z0-9_]){label}\s*$", near.strip()):
        score += 10
    return score


def option_end_anchor(tokenizer, prompt: str, *, label: str, question: object = None, choices: object = None) -> int | None:
    normalized_label = validate_label(label)
    choices_by_label = _choices_by_label(choices)
    if choices_by_label is None:
        return None
    choice = choices_by_label[normalized_label]
    hits = _find_all(prompt, choice)
    if not hits:
        hits = _find_all(prompt, choice, ignore_case=True)
    if not hits:
        return None
    question_end_char = _question_char_end(prompt, question)
    best = max(
        hits,
        key=lambda span: (
            _option_context_score(prompt, span[0], normalized_label, question_end_char=question_end_char),
            span[0],
        ),
    )
    return _end_token_position(tokenizer, prompt, best[1])


def options_end_anchor(tokenizer, prompt: str, *, question: object = None, choices: object = None) -> int | None:
    positions = [
        option_end_anchor(tokenizer, prompt, label=label, question=question, choices=choices)
        for label in LABELS
    ]
    if any(position is None for position in positions):
        return None
    return max(int(position) for position in positions if position is not None)


def anchor_components(anchor: str) -> tuple[str, ...]:
    return ANCHOR_GROUPS.get(anchor, (anchor,))


def anchor_is_leaky(anchor: str) -> bool:
    return any(component in LEAKY_ANCHORS for component in anchor_components(anchor))


def resolve_anchor(
    tokenizer,
    prompt: str,
    anchor: str,
    *,
    question: object = None,
    choices: object = None,
) -> int | None:
    if anchor == "answer_anchor":
        return answer_anchor(tokenizer, prompt)
    if anchor == "content_end":
        return content_end_anchor(tokenizer, prompt)
    if anchor == "mid_prompt":
        ids = tokenize_text(tokenizer, prompt)
        return len(ids) // 2 if ids else None
    if anchor == "question_end":
        return question_end_anchor(tokenizer, prompt, question=question)
    if anchor.startswith("option_") and anchor.endswith("_end"):
        label = anchor[len("option_") : -len("_end")]
        return option_end_anchor(tokenizer, prompt, label=label, question=question, choices=choices)
    if anchor == "options_end":
        return options_end_anchor(tokenizer, prompt, question=question, choices=choices)
    raise ValueError(f"Unknown anchor: {anchor}")


def resolve_anchor_positions(
    tokenizer,
    prompt: str,
    anchor: str,
    *,
    question: object = None,
    choices: object = None,
) -> list[AnchorPosition] | None:
    positions: list[AnchorPosition] = []
    for component in anchor_components(anchor):
        position = resolve_anchor(tokenizer, prompt, component, question=question, choices=choices)
        if position is None:
            return None
        positions.append(AnchorPosition(name=component, position=int(position)))
    if len(positions) > 1:
        values = [position.position for position in positions]
        if len(values) != len(set(values)):
            return None
    return positions
