from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from .utils import LABELS

VANILLA_ANSWER_INSTRUCTION = "Return only the letter (A, B, C, or D)."
VANILLA_ANSWER_PREFIX = "Answer:"


def choices_by_label(choices: object) -> dict[str, str]:
    if isinstance(choices, str):
        try:
            choices = json.loads(choices)
        except json.JSONDecodeError as exc:
            raise ValueError("choices string must be JSON when reconstructing vanilla prompt") from exc
    if hasattr(choices, "tolist"):
        choices = choices.tolist()
    if isinstance(choices, Mapping):
        out = {label: str(choices.get(label, "")) for label in LABELS}
    elif isinstance(choices, Sequence) and not isinstance(choices, (bytes, bytearray, str)):
        values = list(choices)
        if len(values) != len(LABELS):
            raise ValueError(f"expected {len(LABELS)} choices, got {len(values)}")
        out = {label: str(value) for label, value in zip(LABELS, values, strict=True)}
    else:
        raise ValueError(f"unsupported choices type: {type(choices).__name__}")
    missing = [label for label, value in out.items() if value == ""]
    if missing:
        raise ValueError(f"missing choices for labels: {missing}")
    return out


def build_vanilla_prompt(question: object, choices: object) -> str:
    if question is None or str(question).strip() == "":
        raise ValueError("question is required to reconstruct vanilla prompt")
    options = choices_by_label(choices)
    option_lines = [f"{label}) {options[label]}" for label in LABELS]
    return (
        f"{str(question)}\n\n"
        + "\n".join(option_lines)
        + f"\n\n{VANILLA_ANSWER_INSTRUCTION}\n{VANILLA_ANSWER_PREFIX}"
    )


def build_vanilla_prompt_from_row(row: Mapping[str, object]) -> str:
    return build_vanilla_prompt(row.get("question"), row.get("choices"))


def find_existing_vanilla_sources(root: str | Path) -> list[Path]:
    """Return likely local vanilla/unwrapped MMLU source files under a root.

    The focused pipeline does not require an archive. This helper is intentionally
    conservative and only reports filenames that look directly relevant.
    """

    base = Path(root)
    if not base.exists():
        return []
    hits: list[Path] = []
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if "mmlu" in name and any(marker in name for marker in ("vanilla", "unwrapped")):
            hits.append(path)
    return sorted(hits)
