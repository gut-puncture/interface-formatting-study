from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .causal_option_maps import (
    OptionRepresentation,
    OptionSlot,
    ParsedPrompt,
    SourceSpan,
    validate_option_map,
)


SCHEMA_VERSION = 1
MODEL = "gpt-5.6-luna"
REASONING_EFFORT = "xhigh"

INSTRUCTIONS = """Resolve the four answer options in every source prompt.
Return exactly one JSON object per row as JSONL. Do not answer the questions.
Use model gpt-5.6-luna with xhigh reasoning only.

Preferred mode is span_map. Each representation contains four slots in physical A/B/C/D order.
Each slot has content_id 0/1/2/3 plus label_spans and payload_spans as [start,end] offsets
into the exact source_prompt (Python end-exclusive character offsets). Include every repeated
representation that must change together. Verify every span by slicing the source in Python.

Use not_applicable only when the source truly has no independently identifiable answer labels
and positions. Never invent labels. Use direct_counterfactuals only as a last resort when the
four answer options are clear but spans cannot safely express the transformation.

Every object must include annotation_id, source_prompt_sha256, mode, confidence, reason,
model, reasoning_effort, and thread_id. Do not include chain-of-thought.
"""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _annotation_id(item_id: object, wrapper_name: object) -> str:
    return f"{item_id}::{wrapper_name}"


def build_causal_option_packets(
    frame: pd.DataFrame,
    applicability: pd.DataFrame,
    output_dir: str | Path,
    *,
    max_rows: int = 15,
    max_chars: int = 75_000,
) -> dict[str, object]:
    if max_rows < 1 or max_chars < 1:
        raise ValueError("max_rows and max_chars must be positive")
    required_frame = {"item_id", "wrapper_name", "question", "choices", "wrapped_prompt"}
    required_status = {
        "item_id", "wrapper_name", "source_prompt_sha256", "position_applicable",
        "label_applicable", "not_applicable_reason",
    }
    if missing := required_frame - set(frame.columns):
        raise ValueError(f"Causal option input is missing columns: {sorted(missing)}")
    if missing := required_status - set(applicability.columns):
        raise ValueError(f"Causal applicability input is missing columns: {sorted(missing)}")
    source = frame.drop_duplicates(["item_id", "wrapper_name"])
    if len(source) != len(frame):
        raise ValueError("Causal option input contains duplicate item-wrapper rows")
    unresolved = applicability[
        ~applicability["position_applicable"].astype(bool)
        | ~applicability["label_applicable"].astype(bool)
    ]
    joined = unresolved.merge(
        source, on=["item_id", "wrapper_name"], how="left", validate="one_to_one"
    )
    if joined["wrapped_prompt"].isna().any():
        raise ValueError("Applicability contains an item-wrapper absent from source data")

    records: list[dict[str, object]] = []
    for row in joined.to_dict("records"):
        prompt = str(row["wrapped_prompt"])
        digest = _sha256(prompt.encode())
        if digest != str(row["source_prompt_sha256"]):
            raise ValueError(f"Source prompt checksum mismatch for {_annotation_id(row['item_id'], row['wrapper_name'])}")
        records.append({
            "annotation_id": _annotation_id(row["item_id"], row["wrapper_name"]),
            "item_id": str(row["item_id"]),
            "wrapper_name": str(row["wrapper_name"]),
            "question": str(row["question"]),
            "canonical_choices": [str(choice) for choice in row["choices"]],
            "source_prompt": prompt,
            "source_prompt_sha256": digest,
            "parser_reason": str(row["not_applicable_reason"]),
        })
    records.sort(key=lambda row: _sha256(str(row["annotation_id"]).encode()))

    packets: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    current_chars = 0
    for record in records:
        record_chars = len(_canonical_bytes(record).decode())
        if record_chars > max_chars:
            raise ValueError(f"Annotation row {record['annotation_id']} exceeds max_chars={max_chars}")
        if current and (len(current) >= max_rows or current_chars + record_chars > max_chars):
            packets.append(current)
            current, current_chars = [], 0
        current.append(record)
        current_chars += record_chars
    if current:
        packets.append(current)

    root = Path(output_dir)
    packets_dir = root / "packets"
    packets_dir.mkdir(parents=True, exist_ok=True)
    packet_hashes: dict[str, str] = {}
    for index, rows in enumerate(packets):
        name = f"packet-{index:04d}.json"
        encoded = _canonical_bytes({
            "schema_version": SCHEMA_VERSION,
            "instructions": INSTRUCTIONS,
            "rows": rows,
        }) + b"\n"
        packets_dir.joinpath(name).write_bytes(encoded)
        packet_hashes[name] = _sha256(encoded)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "row_count": len(records),
        "packet_count": len(packets),
        "max_rows": max_rows,
        "max_chars": max_chars,
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "instructions_sha256": _sha256(INSTRUCTIONS.encode()),
        "packet_hashes": packet_hashes,
    }
    root.mkdir(parents=True, exist_ok=True)
    root.joinpath("manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _expected(root: Path) -> dict[str, dict[str, object]]:
    manifest = json.loads(root.joinpath("manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported causal option audit schema")
    expected: dict[str, dict[str, object]] = {}
    hashes = manifest.get("packet_hashes", {})
    paths = sorted(root.joinpath("packets").glob("packet-*.json"))
    if {path.name for path in paths} != set(hashes):
        raise ValueError("Causal option packet set does not match manifest")
    for path in paths:
        if _sha256(path.read_bytes()) != hashes[path.name]:
            raise ValueError(f"Causal option packet checksum mismatch: {path.name}")
        for row in json.loads(path.read_text(encoding="utf-8"))["rows"]:
            annotation_id = str(row["annotation_id"])
            if annotation_id in expected:
                raise ValueError(f"Duplicate packet row: {annotation_id}")
            expected[annotation_id] = row
    if len(expected) != int(manifest.get("row_count", -1)):
        raise ValueError("Causal option packet row count mismatch")
    return expected


def _span(value: object) -> SourceSpan:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("Every annotation span must be [start, end]")
    return SourceSpan(int(value[0]), int(value[1]))


def _parsed_annotation(row: dict[str, object], source: dict[str, object]) -> ParsedPrompt:
    prompt = str(source["source_prompt"])
    representations = []
    raw_representations = row.get("representations")
    if not isinstance(raw_representations, list):
        raise ValueError(f"Span annotation has no representations: {row['annotation_id']}")
    for raw_representation in raw_representations:
        slots = []
        for raw_slot in raw_representation.get("slots", []):
            slots.append(OptionSlot(
                int(raw_slot["content_id"]),
                tuple(_span(value) for value in raw_slot.get("label_spans", [])),
                tuple(_span(value) for value in raw_slot.get("payload_spans", [])),
            ))
        representations.append(OptionRepresentation(tuple(slots)))
    parsed = ParsedPrompt(
        wrapper_name=str(source["wrapper_name"]),
        source_sha256=str(source["source_prompt_sha256"]),
        representations=tuple(representations),
        separable=True,
        not_applicable_reason="",
        provenance=f"{MODEL}:{REASONING_EFFORT}:{row['thread_id']}",
    )
    validate_option_map(parsed, prompt)
    return parsed


def load_causal_option_annotations(
    output_dir: str | Path,
) -> dict[tuple[str, str], ParsedPrompt]:
    root = Path(output_dir)
    expected = _expected(root)
    observed: dict[str, dict[str, object]] = {}
    for path in sorted(root.joinpath("labels").glob("*.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            annotation_id = str(row.get("annotation_id", ""))
            if annotation_id not in expected:
                raise ValueError(f"Unexpected annotation_id: {annotation_id}")
            source = expected[annotation_id]
            if row.get("source_prompt_sha256") != source["source_prompt_sha256"]:
                raise ValueError(f"Annotation source prompt checksum mismatch: {annotation_id}")
            if row.get("model") != MODEL or row.get("reasoning_effort") != REASONING_EFFORT:
                raise ValueError(f"Annotation model metadata mismatch: {annotation_id}")
            if not str(row.get("thread_id", "")).strip():
                raise ValueError(f"Annotation thread_id is missing: {annotation_id}")
            if row.get("confidence") not in {"high", "medium", "low"}:
                raise ValueError(f"Annotation confidence is invalid: {annotation_id}")
            if not str(row.get("reason", "")).strip():
                raise ValueError(f"Annotation reason is empty: {annotation_id}")
            previous = observed.get(annotation_id)
            if previous is not None and previous != row:
                raise ValueError(f"Conflicting duplicate annotation: {annotation_id}")
            observed[annotation_id] = row
    missing = sorted(set(expected) - set(observed))
    if missing:
        raise ValueError(f"Causal option annotations incomplete: {len(missing)} missing; first={missing[:5]}")

    result: dict[tuple[str, str], ParsedPrompt] = {}
    for annotation_id, source in expected.items():
        row = observed[annotation_id]
        if row.get("mode") == "span_map":
            parsed = _parsed_annotation(row, source)
        elif row.get("mode") == "not_applicable":
            parsed = ParsedPrompt(
                wrapper_name=str(source["wrapper_name"]),
                source_sha256=str(source["source_prompt_sha256"]),
                representations=(),
                separable=False,
                not_applicable_reason=str(row["reason"]),
                provenance=f"{MODEL}:{REASONING_EFFORT}:{row['thread_id']}",
            )
            validate_option_map(parsed, str(source["source_prompt"]))
        else:
            raise ValueError(f"Unsupported annotation mode for {annotation_id}: {row.get('mode')}")
        result[(str(source["item_id"]), str(source["wrapper_name"]))] = parsed
    return result
