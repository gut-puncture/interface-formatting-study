from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from .utils import LABELS, validate_label, write_json


@dataclass(frozen=True)
class NormalizedExample:
    item_id: str
    subject: str | None
    question: str | None
    choices: list[str] | None
    correct_label: str
    correct_index: int | None
    wrapper_name: str
    wrapped_prompt: str
    source_split: str | None = None
    prompt_id: str | None = None
    wrapper_description: str | None = None
    canonical_fields_missing: list[str] = field(default_factory=list)
    source_record: dict[str, Any] = field(default_factory=dict)


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc


def _normalize_choices(record: dict[str, Any]) -> tuple[list[str] | None, int | None, list[str]]:
    missing: list[str] = []
    correct_label_raw = record.get("correct_label", record.get("correct_key"))
    correct_label = None if correct_label_raw is None else validate_label(str(correct_label_raw), name="correct_label")
    options = record.get("choices", record.get("options"))
    if options is None:
        missing.append("choices")
        correct_index = record.get("correct_index")
        if correct_index is None:
            return None, None, missing
        correct_index = int(correct_index)
        if correct_index < 0 or correct_index >= len(LABELS):
            raise ValueError(f"correct_index out of range: {correct_index}")
        return None, correct_index, missing
    if isinstance(options, dict):
        choices = [str(options.get(label, "")) for label in LABELS]
    elif isinstance(options, list):
        choices = [str(x) for x in options]
    else:
        raise ValueError(f"Unsupported choices/options type: {type(options).__name__}")
    if len(choices) != 4 or any(choice == "" for choice in choices):
        raise ValueError("MMLU choices must normalize to four non-empty choices")

    correct_index = record.get("correct_index")
    if correct_index is None:
        if correct_label is not None:
            correct_index = LABELS.index(correct_label)
    else:
        correct_index = int(correct_index)
        if correct_index < 0 or correct_index >= len(LABELS):
            raise ValueError(f"correct_index out of range: {correct_index}")
        if correct_label is not None and LABELS[correct_index] != correct_label:
            raise ValueError(
                f"correct_index {correct_index} is inconsistent with correct_label {correct_label!r}"
            )
    return choices, correct_index, missing


def normalize_record(record: dict[str, Any]) -> NormalizedExample:
    missing: list[str] = []
    item_id = record.get("item_id", record.get("example_id"))
    if item_id is None:
        raise ValueError("Dataset record is missing item_id/example_id")

    correct_label_raw = record.get("correct_label", record.get("correct_key"))
    if correct_label_raw is None:
        raise ValueError(f"Record {item_id!r} is missing correct_label/correct_key")
    correct_label = validate_label(str(correct_label_raw), name="correct_label")

    choices, correct_index, choice_missing = _normalize_choices(record)
    missing.extend(choice_missing)

    question = record.get("question")
    if question is None:
        missing.append("question")

    wrapped_prompt = record.get("wrapped_prompt", record.get("prompt_text"))
    if wrapped_prompt is None:
        raise ValueError(f"Record {item_id!r} is missing wrapped_prompt/prompt_text")

    wrapper_name = record.get("wrapper_name", record.get("wrapper_id"))
    if wrapper_name is None:
        raise ValueError(f"Record {item_id!r} is missing wrapper_name/wrapper_id")

    return NormalizedExample(
        item_id=str(item_id),
        subject=None if record.get("subject") is None else str(record.get("subject")),
        question=None if question is None else str(question),
        choices=choices,
        correct_label=correct_label,
        correct_index=correct_index,
        wrapper_name=str(wrapper_name),
        wrapped_prompt=str(wrapped_prompt),
        source_split=None if record.get("split") is None else str(record.get("split")),
        prompt_id=record.get("prompt_id") or record.get("prompt_uid"),
        wrapper_description=record.get("wrapper_description"),
        canonical_fields_missing=missing,
        source_record=record,
    )


def load_normalized(path: str | Path, *, limit: int | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for i, record in enumerate(iter_jsonl(path)):
        if limit is not None and i >= limit:
            break
        records.append(record)
    schema_fields = set().union(*(record.keys() for record in records)) if records else set()
    for record in records:
        normalized = normalize_record(record)
        row = asdict(normalized)
        row["source_schema_fields"] = sorted(schema_fields)
        rows.append(row)
    return pd.DataFrame(rows)


def dataset_audit(path: str | Path, *, max_examples_per_wrapper: int = 1) -> dict[str, Any]:
    item_ids: set[str] = set()
    wrappers: dict[str, int] = {}
    subjects: dict[str, int] = {}
    labels: dict[str, int] = {}
    schema_fields: set[str] = set()
    missing_counts: dict[str, int] = {}
    prompt_lengths: list[int] = []
    examples_by_wrapper: dict[str, list[str]] = {}
    rows = 0

    for record in iter_jsonl(path):
        schema_fields.update(record.keys())
        ex = normalize_record(record)
        rows += 1
        item_ids.add(ex.item_id)
        wrappers[ex.wrapper_name] = wrappers.get(ex.wrapper_name, 0) + 1
        if ex.subject is not None:
            subjects[ex.subject] = subjects.get(ex.subject, 0) + 1
        labels[ex.correct_label] = labels.get(ex.correct_label, 0) + 1
        prompt_lengths.append(len(ex.wrapped_prompt))
        for field_name in ex.canonical_fields_missing:
            missing_counts[field_name] = missing_counts.get(field_name, 0) + 1
        bucket = examples_by_wrapper.setdefault(ex.wrapper_name, [])
        if len(bucket) < max_examples_per_wrapper:
            bucket.append(ex.wrapped_prompt)

    prompt_series = pd.Series(prompt_lengths, dtype="float64")
    return {
        "dataset_path": str(path),
        "num_rows": rows,
        "num_items": len(item_ids),
        "num_wrappers": len(wrappers),
        "schema_fields": sorted(schema_fields),
        "wrapper_counts": dict(sorted(wrappers.items())),
        "subject_counts": dict(sorted(subjects.items())),
        "correct_label_distribution": {label: labels.get(label, 0) for label in LABELS},
        "canonical_fields_missing_counts": dict(sorted(missing_counts.items())),
        "prompt_length_chars": {
            "mean": float(prompt_series.mean()) if rows else 0.0,
            "median": float(prompt_series.median()) if rows else 0.0,
            "max": int(prompt_series.max()) if rows else 0,
            "min": int(prompt_series.min()) if rows else 0,
        },
        "example_prompts_by_wrapper": examples_by_wrapper,
    }


def dataset_audit_from_frame(
    df: pd.DataFrame,
    *,
    dataset_path: str | Path,
    wrapper_policy: str | None = None,
    source_num_rows: int | None = None,
    max_examples_per_wrapper: int = 1,
) -> dict[str, Any]:
    prompt_lengths = df["wrapped_prompt"].astype(str).str.len()
    missing_counts: dict[str, int] = {}
    if "canonical_fields_missing" in df:
        for fields in df["canonical_fields_missing"]:
            for field_name in fields or []:
                missing_counts[str(field_name)] = missing_counts.get(str(field_name), 0) + 1
    examples_by_wrapper = {
        wrapper: list(group["wrapped_prompt"].astype(str).head(max_examples_per_wrapper))
        for wrapper, group in df.groupby("wrapper_name", sort=True)
    }
    source_schema_fields = sorted(
        set().union(*(set(fields or []) for fields in df.get("source_schema_fields", pd.Series(dtype=object))))
    )
    return {
        "dataset_path": str(dataset_path),
        "wrapper_policy": wrapper_policy,
        "source_num_rows": None if source_num_rows is None else int(source_num_rows),
        "num_rows": int(len(df)),
        "num_items": int(df["item_id"].nunique()),
        "num_wrappers": int(df["wrapper_name"].nunique()),
        "schema_fields": source_schema_fields,
        "wrapper_counts": {k: int(v) for k, v in df["wrapper_name"].value_counts().sort_index().items()},
        "wrapper_category_counts": {k: int(v) for k, v in df["wrapper_category"].value_counts().sort_index().items()}
        if "wrapper_category" in df
        else {},
        "subject_counts": {k: int(v) for k, v in df["subject"].value_counts().sort_index().items()},
        "correct_label_distribution": {label: int((df["correct_label"] == label).sum()) for label in LABELS},
        "canonical_fields_missing_counts": dict(sorted(missing_counts.items())),
        "prompt_length_chars": {
            "mean": float(prompt_lengths.mean()) if len(df) else 0.0,
            "median": float(prompt_lengths.median()) if len(df) else 0.0,
            "max": int(prompt_lengths.max()) if len(df) else 0,
            "min": int(prompt_lengths.min()) if len(df) else 0,
        },
        "example_prompts_by_wrapper": examples_by_wrapper,
    }


def save_dataset_audit(path: str | Path, output_path: str | Path) -> dict[str, Any]:
    audit = dataset_audit(path)
    write_json(audit, output_path)
    return audit
