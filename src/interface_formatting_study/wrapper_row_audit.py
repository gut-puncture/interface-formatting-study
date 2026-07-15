from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .utils import write_table_atomic
from .wrapper_audit import contains_all_choices, contains_nearly_verbatim


LABELS = {
    "formatting_only",
    "meaning_preserving_rewrite",
    "content_changed",
    "ambiguous",
}
CONFIDENCE = {"high", "medium", "low"}

AUDIT_INSTRUCTIONS = """You are auditing whether a formatted multiple-choice prompt preserves its canonical task.
For every row, return one JSONL object with audit_row_id, label, confidence, and a concise reason.
Allowed labels: formatting_only, meaning_preserving_rewrite, content_changed, ambiguous.
Use formatting_only only when question meaning, four answer texts, answer mapping, and requested task are unchanged.
Do not infer anything from wrapper name. Do not omit rows. Do not include chain-of-thought.
"""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def row_audit_id(item_id: object, wrapper_name: object) -> str:
    return f"{item_id}::{wrapper_name}"


def _audit_record(row: pd.Series) -> dict[str, object]:
    question = str(row["question"])
    choices = [str(choice) for choice in row["choices"]]
    prompt = str(row["wrapped_prompt"])
    return {
        "audit_row_id": row_audit_id(row["item_id"], row["wrapper_name"]),
        "item_id": str(row["item_id"]),
        "wrapper_name": str(row["wrapper_name"]),
        "question": question,
        "choices": choices,
        "wrapped_prompt": prompt,
        "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
        "deterministic": {
            "question_exact": question in prompt,
            "question_near": contains_nearly_verbatim(prompt, question),
            "choice_exact_count": sum(choice in prompt for choice in choices),
            "all_choices_exact": all(choice in prompt for choice in choices),
            "all_choices_near": contains_all_choices(prompt, choices),
            "letter_only_instruction": "return only the letter" in prompt.casefold(),
        },
    }


def build_audit_packets(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    max_rows: int = 500,
    max_chars: int = 600_000,
) -> dict[str, object]:
    if max_rows < 1 or max_chars < 1:
        raise ValueError("max_rows and max_chars must be positive")
    required = {"item_id", "wrapper_name", "question", "choices", "wrapped_prompt"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Wrapper audit input is missing columns: {sorted(missing)}")
    if frame.duplicated(["item_id", "wrapper_name"]).any():
        raise ValueError("Wrapper audit input contains duplicate item-wrapper rows")

    records = [_audit_record(row) for _, row in frame.iterrows()]
    records.sort(key=lambda row: _sha256_bytes(str(row["audit_row_id"]).encode("utf-8")))
    root = Path(output_dir)
    packets_dir = root / "packets"
    packets_dir.mkdir(parents=True, exist_ok=True)

    packets: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    current_chars = 0
    for record in records:
        record_chars = len(_canonical_json(record).decode("utf-8"))
        if record_chars > max_chars:
            raise ValueError(f"Audit row {record['audit_row_id']} exceeds max_chars={max_chars}")
        if current and (len(current) >= max_rows or current_chars + record_chars > max_chars):
            packets.append(current)
            current = []
            current_chars = 0
        current.append(record)
        current_chars += record_chars
    if current:
        packets.append(current)

    packet_hashes: dict[str, str] = {}
    for index, rows in enumerate(packets):
        name = f"packet-{index:04d}.json"
        payload = {
            "audit_schema_version": 1,
            "instructions": AUDIT_INSTRUCTIONS,
            "rows": rows,
        }
        encoded = _canonical_json(payload)
        (packets_dir / name).write_bytes(encoded + b"\n")
        packet_hashes[name] = _sha256_bytes(encoded + b"\n")

    manifest = {
        "audit_schema_version": 1,
        "row_count": len(records),
        "packet_count": len(packets),
        "max_rows": int(max_rows),
        "max_chars": int(max_chars),
        "instructions_sha256": _sha256_bytes(AUDIT_INSTRUCTIONS.encode("utf-8")),
        "packet_hashes": packet_hashes,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _expected_rows(root: Path) -> dict[str, dict[str, object]]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Wrapper audit manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("audit_schema_version", -1)) != 1:
        raise ValueError("Unsupported wrapper audit schema version")
    expected_hashes = manifest.get("packet_hashes")
    if not isinstance(expected_hashes, dict):
        raise ValueError("Wrapper audit manifest has no packet hashes")
    observed_paths = sorted((root / "packets").glob("packet-*.json"))
    if {path.name for path in observed_paths} != set(expected_hashes):
        raise ValueError("Wrapper audit packet set does not match its manifest")

    expected: dict[str, dict[str, object]] = {}
    for packet_path in observed_paths:
        if _sha256_bytes(packet_path.read_bytes()) != str(expected_hashes[packet_path.name]):
            raise ValueError(f"Wrapper audit packet checksum mismatch: {packet_path.name}")
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        for row in packet["rows"]:
            audit_id = str(row["audit_row_id"])
            if audit_id in expected:
                raise ValueError(f"Duplicate audit row in packets: {audit_id}")
            expected[audit_id] = row
    if len(expected) != int(manifest.get("row_count", -1)):
        raise ValueError("Wrapper audit row count does not match its manifest")
    return expected


def merge_audit_labels(output_dir: str | Path) -> pd.DataFrame:
    root = Path(output_dir)
    expected = _expected_rows(root)
    observed: dict[str, dict[str, object]] = {}
    for labels_path in sorted((root / "labels").glob("*.jsonl")):
        for line_number, line in enumerate(labels_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {labels_path}:{line_number}") from exc
            audit_id = str(row.get("audit_row_id", ""))
            if audit_id not in expected:
                raise ValueError(f"Unexpected audit_row_id {audit_id!r} in {labels_path}")
            label = str(row.get("label", ""))
            confidence = str(row.get("confidence", ""))
            reason = str(row.get("reason", "")).strip()
            if label not in LABELS:
                raise ValueError(f"Audit row {audit_id} has invalid label {label!r}")
            if confidence not in CONFIDENCE:
                raise ValueError(f"Audit row {audit_id} has invalid confidence {confidence!r}")
            if not reason:
                raise ValueError(f"Audit row {audit_id} has an empty reason")
            normalized = {
                "audit_row_id": audit_id,
                "label": label,
                "confidence": confidence,
                "reason": reason,
            }
            previous = observed.get(audit_id)
            if previous is not None and previous != normalized:
                raise ValueError(f"Conflicting duplicate audit label for {audit_id}")
            observed[audit_id] = normalized

    missing = sorted(set(expected) - set(observed))
    if missing:
        raise ValueError(f"Audit labels are incomplete: {len(missing)} rows missing; first={missing[:5]}")

    rows = []
    for audit_id in sorted(expected):
        source = expected[audit_id]
        label = observed[audit_id]
        rows.append(
            {
                **label,
                "item_id": source["item_id"],
                "wrapper_name": source["wrapper_name"],
                "prompt_sha256": source["prompt_sha256"],
                **{f"deterministic_{key}": value for key, value in source["deterministic"].items()},
            }
        )
    merged = pd.DataFrame(rows)
    summary = (
        merged.groupby(["wrapper_name", "label", "confidence"], as_index=False)
        .size()
        .rename(columns={"size": "rows"})
    )
    write_table_atomic(merged, root / "wrapper_audit_labels.parquet")
    write_table_atomic(summary, root / "wrapper_audit_summary.csv")
    return merged


def finalize_audit_labels(
    first_pass_path: str | Path,
    adjudication_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Strictly apply the second pass to every row flagged by the first pass."""

    first = pd.read_parquet(first_pass_path)
    adjudication = pd.read_parquet(adjudication_path)
    required_first = {"audit_row_id", "label", "confidence", "reason"}
    required_second = required_first | {
        "adjudicated_label",
        "adjudicated_confidence",
        "adjudicated_reason",
    }
    if missing := required_first - set(first.columns):
        raise ValueError(f"First-pass audit is missing columns: {sorted(missing)}")
    if missing := required_second - set(adjudication.columns):
        raise ValueError(f"Adjudication is missing columns: {sorted(missing)}")
    if first["audit_row_id"].duplicated().any() or adjudication["audit_row_id"].duplicated().any():
        raise ValueError("Audit finalization inputs contain duplicate audit_row_id values")
    flagged = first[first["label"].isin({"content_changed", "ambiguous"})]
    if set(adjudication["audit_row_id"].astype(str)) != set(flagged["audit_row_id"].astype(str)):
        raise ValueError("Adjudication must cover exactly every initially content-changed or ambiguous row")
    original = adjudication.set_index("audit_row_id")[["label", "confidence", "reason"]]
    expected = flagged.set_index("audit_row_id")[["label", "confidence", "reason"]]
    pd.testing.assert_frame_equal(original.sort_index(), expected.sort_index(), check_names=False)
    if not set(adjudication["adjudicated_label"]).issubset(LABELS):
        raise ValueError("Adjudication contains an invalid label")
    if not set(adjudication["adjudicated_confidence"]).issubset(CONFIDENCE):
        raise ValueError("Adjudication contains an invalid confidence")
    if adjudication["adjudicated_reason"].astype(str).str.strip().eq("").any():
        raise ValueError("Adjudication contains an empty reason")

    second = adjudication.set_index("audit_row_id")
    final = first.copy().set_index("audit_row_id")
    for column in ("adjudicated_label", "adjudicated_confidence", "adjudicated_reason"):
        final[column] = second[column]
    final["final_label"] = final["adjudicated_label"].fillna(final["label"])
    final["final_confidence"] = final["adjudicated_confidence"].fillna(final["confidence"])
    final["final_reason"] = final["adjudicated_reason"].fillna(final["reason"])
    final = final.reset_index().sort_values("audit_row_id", kind="mergesort").reset_index(drop=True)
    write_table_atomic(final, output_path)
    summary_path = Path(output_path).with_name(Path(output_path).stem + "_summary.csv")
    summary = final.groupby(["wrapper_name", "final_label"], as_index=False).size().rename(columns={"size": "rows"})
    write_table_atomic(summary, summary_path)
    return final
