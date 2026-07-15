from __future__ import annotations

import csv
import hashlib
import html
import io
import json
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
class RenderedPrompt:
    prompt: str
    prompt_sha256: str
    correct_output: str
    calibration_prompt: str | None
    calibration_prompt_sha256: str | None


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


def _quoted(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _plain(question: str, options: list[tuple[str, str]]) -> str:
    rows = "\n".join(f"{label}) {text}" for label, text in options)
    return f"{question}\n\n{rows}"


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
        f"    slot{position}: option(label: {_quoted(label)}, text: {_quoted(text)})"
        for position, (label, text) in enumerate(options)
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
    "plain": _plain,
    "csv_inline": _csv,
    "graphql_query": _graphql,
    "html_form": _html,
    "ini_file": _ini,
    "key_equals": _key_equals,
    "protobuf_msg": _protobuf,
    "shell_heredoc": _shell,
    "toml_config": _toml,
}


def _render_body(
    format_name: str,
    question: str,
    choices: Sequence[str],
    assignment: Assignment,
) -> str:
    options = [
        (assignment.labels_by_position[position], str(choices[content_id]))
        for position, content_id in enumerate(assignment.content_ids_by_position)
    ]
    return _RENDERERS[format_name](question, options)


def render_causal_prompt(
    format_name: str,
    question: str,
    choices: Sequence[str],
    assignment: Assignment,
    *,
    correct_content_id: int,
    readout: str,
) -> RenderedPrompt:
    if format_name not in _RENDERERS:
        raise ValueError(f"Unknown controlled format {format_name!r}")
    if len(choices) != 4 or not 0 <= int(correct_content_id) < 4:
        raise ValueError("Causal prompts require four choices and a correct content id in [0, 3]")
    if readout not in {"letter", "text"}:
        raise ValueError("readout must be 'letter' or 'text'")

    body = _render_body(format_name, str(question), choices, assignment)
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

    calibration_prompt = None
    calibration_sha = None
    if readout == "letter":
        placeholders = [f"OPTION_{index}_PLACEHOLDER" for index in range(4)]
        calibration_body = _render_body(
            format_name,
            "QUESTION_TEXT_PLACEHOLDER",
            placeholders,
            assignment,
        )
        calibration_prompt = f"{calibration_body}\n\n{instruction}\nAnswer: "
        calibration_sha = hashlib.sha256(calibration_prompt.encode("utf-8")).hexdigest()

    return RenderedPrompt(
        prompt=prompt,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        correct_output=correct_output,
        calibration_prompt=calibration_prompt,
        calibration_prompt_sha256=calibration_sha,
    )


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


def build_causal_design(frame: pd.DataFrame, *, splits: Sequence[str] | None = None) -> pd.DataFrame:
    allowed_splits = None if splits is None else {str(split) for split in splits}
    rows: list[dict[str, object]] = []
    assignments = balanced_assignments()
    for item in _canonical_items(frame):
        if allowed_splits is not None and str(item["split"]) not in allowed_splits:
            continue
        item_id = str(item["item_id"])
        question = str(item["question"])
        choices = [str(choice) for choice in item["choices"]]
        correct_content_id = int(item["correct_index"])
        source_prompts = item["source_prompts"]
        for format_name in CONTROLLED_FORMATS:
            source_prompt = None if format_name == "plain" else source_prompts[format_name]
            source_prompt_sha = (
                None
                if source_prompt is None
                else hashlib.sha256(str(source_prompt).encode("utf-8")).hexdigest()
            )
            for assignment in assignments:
                rendered = render_causal_prompt(
                    format_name,
                    question,
                    choices,
                    assignment,
                    correct_content_id=correct_content_id,
                    readout="letter",
                )
                correct_position = assignment.content_ids_by_position.index(correct_content_id)
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
                        "template_scope": "controlled_canonical",
                        "source_prompt_sha256": source_prompt_sha,
                        "prompt": rendered.prompt,
                        "prompt_sha256": rendered.prompt_sha256,
                        "calibration_prompt": rendered.calibration_prompt,
                        "calibration_prompt_sha256": rendered.calibration_prompt_sha256,
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
                format_name,
                question,
                choices,
                identity,
                correct_content_id=correct_content_id,
                readout="text",
            )
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
                    "template_scope": "controlled_canonical",
                    "source_prompt_sha256": source_prompt_sha,
                    "prompt": rendered.prompt,
                    "prompt_sha256": rendered.prompt_sha256,
                    "calibration_prompt": None,
                    "calibration_prompt_sha256": None,
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
