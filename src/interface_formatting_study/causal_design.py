from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import random
from dataclasses import dataclass
from typing import Callable, Sequence

import pandas as pd


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
LETTERS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class Assignment:
    variant: int
    content_ids_by_position: tuple[int, int, int, int]
    labels_by_position: tuple[str, str, str, str]
    position_shift: int
    label_shift: int


@dataclass(frozen=True)
class RenderedPrompt:
    prompt: str
    prompt_sha256: str
    correct_output: str


def balanced_assignments(
    item_id: object,
    block_id: object = "",
) -> tuple[Assignment, Assignment, Assignment, Assignment]:
    """Four randomized-block variants balancing positions and labels independently."""

    seed = int.from_bytes(
        hashlib.sha256(f"{item_id}|{block_id}".encode("utf-8")).digest()[:8],
        "big",
    )
    rng = random.Random(seed)
    position_shifts = [1, 2, 3]
    label_shifts = [1, 2, 3]
    rng.shuffle(position_shifts)
    rng.shuffle(label_shifts)
    position_shifts.insert(0, 0)
    label_shifts.insert(0, 0)
    assignments = []
    for variant, (position_shift, label_shift) in enumerate(zip(position_shifts, label_shifts, strict=True)):
        content_by_position = [0, 0, 0, 0]
        for content_id in range(4):
            position = (content_id + position_shift) % 4
            content_by_position[position] = content_id
        labels_by_position = ["", "", "", ""]
        for position, content_id in enumerate(content_by_position):
            labels_by_position[position] = LETTERS[(content_id + label_shift) % 4]
        assignments.append(
            Assignment(
                variant=variant,
                content_ids_by_position=tuple(content_by_position),
                labels_by_position=tuple(labels_by_position),
                position_shift=position_shift,
                label_shift=label_shift,
            )
        )
    return tuple(assignments)  # type: ignore[return-value]


def _quoted(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _csv(question: str, options: list[tuple[str, str]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["field", "label", "text"])
    writer.writerow(["question", "", question])
    for label, text in options:
        writer.writerow(["option", label, text])
    return output.getvalue().rstrip("\n")


def _graphql(question: str, options: list[tuple[str, str]]) -> str:
    option_lines = "\n".join(
        f"    option(label: {_quoted(label)}, text: {_quoted(text)})" for label, text in options
    )
    return f"query MultipleChoice {{\n  question(text: {_quoted(question)})\n  options {{\n{option_lines}\n  }}\n}}"


def _html(question: str, options: list[tuple[str, str]]) -> str:
    rows = "\n".join(
        f'  <label><input type="radio" name="answer" value="{html.escape(label, quote=True)}"> '
        f"{html.escape(label)}: {html.escape(text)}</label>"
        for label, text in options
    )
    return f"<form>\n  <fieldset>\n    <legend>{html.escape(question)}</legend>\n{rows}\n  </fieldset>\n</form>"


def _ini(question: str, options: list[tuple[str, str]]) -> str:
    rows = "\n".join(f"{label} = {_quoted(text)}" for label, text in options)
    return f"[question]\ntext = {_quoted(question)}\n\n[options]\n{rows}"


def _key_equals(question: str, options: list[tuple[str, str]]) -> str:
    rows = "\n".join(f"option_{label}={_quoted(text)}" for label, text in options)
    return f"question={_quoted(question)}\n{rows}"


def _protobuf(question: str, options: list[tuple[str, str]]) -> str:
    rows = "\n".join(
        f"options {{ label: {_quoted(label)} text: {_quoted(text)} }}" for label, text in options
    )
    return f"question: {_quoted(question)}\n{rows}"


def _shell(question: str, options: list[tuple[str, str]]) -> str:
    body = "\n".join([f"Question: {question}", *(f"{label}) {text}" for label, text in options)])
    marker = "MCQ_EOF"
    while marker in body:
        marker += "_X"
    return f"cat <<'{marker}'\n{body}\n{marker}"


def _toml(question: str, options: list[tuple[str, str]]) -> str:
    rows = "\n".join(f"{label} = {_quoted(text)}" for label, text in options)
    return f"[multiple_choice]\nquestion = {_quoted(question)}\n\n[multiple_choice.options]\n{rows}"


_RENDERERS: dict[str, Callable[[str, list[tuple[str, str]]], str]] = {
    "csv_inline": _csv,
    "graphql_query": _graphql,
    "html_form": _html,
    "ini_file": _ini,
    "key_equals": _key_equals,
    "protobuf_msg": _protobuf,
    "shell_heredoc": _shell,
    "toml_config": _toml,
}


def render_causal_prompt(
    wrapper_name: str,
    question: str,
    choices: Sequence[str],
    assignment: Assignment,
    *,
    correct_content_id: int,
    readout: str,
) -> RenderedPrompt:
    if wrapper_name not in _RENDERERS:
        raise ValueError(f"Unknown causal wrapper {wrapper_name!r}")
    if len(choices) != 4 or not 0 <= int(correct_content_id) < 4:
        raise ValueError("Causal prompts require four choices and a correct content id in [0, 3]")
    if readout not in {"letter", "text"}:
        raise ValueError("readout must be 'letter' or 'text'")
    options = [
        (assignment.labels_by_position[position], str(choices[content_id]))
        for position, content_id in enumerate(assignment.content_ids_by_position)
    ]
    body = _RENDERERS[wrapper_name](str(question), options)
    instruction = (
        "Return only the letter A, B, C, or D."
        if readout == "letter"
        else "Return only the exact answer text, not its letter."
    )
    prompt = f"{body}\n\n{instruction}\nAnswer: "
    correct_position = assignment.content_ids_by_position.index(int(correct_content_id))
    correct_output = (
        assignment.labels_by_position[correct_position]
        if readout == "letter"
        else str(choices[int(correct_content_id)])
    )
    return RenderedPrompt(
        prompt=prompt,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        correct_output=correct_output,
    )


def _canonical_items(frame: pd.DataFrame) -> list[dict[str, object]]:
    required = {"item_id", "subject", "split", "question", "choices", "correct_index"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Causal design input is missing columns: {sorted(missing)}")
    items: list[dict[str, object]] = []
    for item_id, group in frame.groupby("item_id", sort=True):
        rows = group[list(required)].to_dict("records")
        canonical = rows[0]
        for row in rows[1:]:
            if any(row[key] != canonical[key] for key in ("subject", "split", "question", "choices", "correct_index")):
                raise ValueError(f"Canonical fields disagree across wrappers for item {item_id}")
        items.append(canonical)
    return items


def build_causal_design(frame: pd.DataFrame, *, splits: Sequence[str] | None = None) -> pd.DataFrame:
    allowed_splits = None if splits is None else {str(split) for split in splits}
    rows: list[dict[str, object]] = []
    for item in _canonical_items(frame):
        if allowed_splits is not None and str(item["split"]) not in allowed_splits:
            continue
        item_id = str(item["item_id"])
        question = str(item["question"])
        choices = [str(choice) for choice in item["choices"]]
        correct_content_id = int(item["correct_index"])
        for wrapper_name in ACTIVE_WRAPPERS:
            assignments = balanced_assignments(item_id, wrapper_name)
            for assignment in assignments:
                rendered = render_causal_prompt(
                    wrapper_name,
                    question,
                    choices,
                    assignment,
                    correct_content_id=correct_content_id,
                    readout="letter",
                )
                correct_position = assignment.content_ids_by_position.index(correct_content_id)
                rows.append(
                    {
                        "work_key": f"letter|{item_id}|{wrapper_name}|{assignment.variant}",
                        "item_id": item_id,
                        "subject": item["subject"],
                        "split": item["split"],
                        "wrapper_name": wrapper_name,
                        "arm": "letter_permutation",
                        "variant": assignment.variant,
                        "position_shift": assignment.position_shift,
                        "label_shift": assignment.label_shift,
                        "prompt": rendered.prompt,
                        "prompt_sha256": rendered.prompt_sha256,
                        "content_ids_by_position": list(assignment.content_ids_by_position),
                        "labels_by_position": list(assignment.labels_by_position),
                        "candidate_texts": choices,
                        "correct_content_id": correct_content_id,
                        "correct_position": correct_position,
                        "correct_label": rendered.correct_output,
                        "correct_text": choices[correct_content_id],
                    }
                )
            identity = assignments[0]
            rendered = render_causal_prompt(
                wrapper_name,
                question,
                choices,
                identity,
                correct_content_id=correct_content_id,
                readout="text",
            )
            rows.append(
                {
                    "work_key": f"text|{item_id}|{wrapper_name}|0",
                    "item_id": item_id,
                    "subject": item["subject"],
                    "split": item["split"],
                    "wrapper_name": wrapper_name,
                    "arm": "answer_text",
                    "variant": 0,
                    "position_shift": 0,
                    "label_shift": 0,
                    "prompt": rendered.prompt,
                    "prompt_sha256": rendered.prompt_sha256,
                    "content_ids_by_position": list(identity.content_ids_by_position),
                    "labels_by_position": list(identity.labels_by_position),
                    "candidate_texts": choices,
                    "correct_content_id": correct_content_id,
                    "correct_position": correct_content_id,
                    "correct_label": LETTERS[correct_content_id],
                    "correct_text": rendered.correct_output,
                }
            )
    design = pd.DataFrame(rows)
    if not design.empty and not design["work_key"].is_unique:
        raise AssertionError("Causal design produced duplicate work keys")
    return design.sort_values("work_key", kind="mergesort").reset_index(drop=True)
