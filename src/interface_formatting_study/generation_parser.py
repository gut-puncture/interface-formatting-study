from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

from .utils import LABELS

LABEL_PATTERN = re.compile(
    r"(?:^|\b)(?:option|answer|correct option|correct answer|the answer is|answer:)?\s*[\(\[]?\s*([ABCD])\s*[\)\]\.]?(?:\b|$)",
    re.I,
)


@dataclass(frozen=True)
class ParsedGeneration:
    label: str | None
    valid: bool
    reason: str


def parse_generation(text: str, *, choices: Mapping[str, str] | None = None) -> ParsedGeneration:
    stripped = text.strip()
    if not stripped:
        return ParsedGeneration(None, False, "empty")

    labels = {match.group(1).upper() for match in LABEL_PATTERN.finditer(stripped)}
    if len(labels) == 1:
        return ParsedGeneration(next(iter(labels)), True, "label_pattern")
    if len(labels) > 1:
        return ParsedGeneration(None, False, "multiple_conflicting_labels")

    if choices:
        lowered = stripped.lower()
        exact_matches = [label for label, answer in choices.items() if str(answer).strip().lower() in lowered]
        exact_matches = [label for label in exact_matches if label in LABELS]
        if len(set(exact_matches)) == 1:
            return ParsedGeneration(exact_matches[0], True, "exact_answer_text")
        if len(set(exact_matches)) > 1:
            return ParsedGeneration(None, False, "multiple_answer_text_matches")

    return ParsedGeneration(None, False, "no_valid_answer")

