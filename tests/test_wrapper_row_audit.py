from __future__ import annotations

import json

import pandas as pd
import pytest

from interface_formatting_study.wrapper_row_audit import (
    LABELS,
    build_audit_packets,
    merge_audit_labels,
    row_audit_id,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item_id": "item-1",
                "wrapper_name": "csv_inline",
                "question": "What is 2 + 2?",
                "choices": ["3", "4", "5", "6"],
                "wrapped_prompt": "question,What is 2 + 2?\nA,3\nB,4\nC,5\nD,6\nReturn only the letter.",
            },
            {
                "item_id": "item-2",
                "wrapper_name": "html_form",
                "question": "Which tag is semantic?",
                "choices": ["div", "article", "span", "b"],
                "wrapped_prompt": "<p>Choose an answer.</p>",
            },
        ]
    )


def test_audit_packets_are_deterministic_bounded_and_outcome_blind(tmp_path):
    first = build_audit_packets(_frame(), tmp_path / "first", max_rows=1, max_chars=10_000)
    second = build_audit_packets(_frame(), tmp_path / "second", max_rows=1, max_chars=10_000)

    assert first["packet_hashes"] == second["packet_hashes"]
    assert first["row_count"] == 2
    assert first["packet_count"] == 2
    packet = json.loads((tmp_path / "first" / "packets" / "packet-0000.json").read_text())
    assert packet["rows"][0]["audit_row_id"] == row_audit_id("item-1", "csv_inline")
    assert packet["rows"][0]["deterministic"]["question_exact"]
    assert packet["rows"][0]["deterministic"]["all_choices_exact"]
    assert "correct_label" not in packet["rows"][0]
    assert "prediction" not in json.dumps(packet)


def test_audit_packet_rejects_one_row_larger_than_character_budget(tmp_path):
    with pytest.raises(ValueError, match="exceeds max_chars"):
        build_audit_packets(_frame().head(1), tmp_path, max_rows=10, max_chars=10)


def test_merge_requires_exact_coverage_and_valid_labels(tmp_path):
    build_audit_packets(_frame(), tmp_path, max_rows=10, max_chars=10_000)
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    label_rows = [
        {
            "audit_row_id": row_audit_id("item-1", "csv_inline"),
            "label": "formatting_only",
            "confidence": "high",
            "reason": "Question and options are unchanged.",
        },
        {
            "audit_row_id": row_audit_id("item-2", "html_form"),
            "label": "content_changed",
            "confidence": "high",
            "reason": "The question and options are missing.",
        },
    ]
    (labels_dir / "packet-0000.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in label_rows),
        encoding="utf-8",
    )

    merged = merge_audit_labels(tmp_path)

    assert len(merged) == 2
    assert set(merged["label"]) <= LABELS
    assert (tmp_path / "wrapper_audit_labels.parquet").exists()
    assert (tmp_path / "wrapper_audit_summary.csv").exists()

    label_rows[0]["label"] = "made_up"
    (labels_dir / "packet-0000.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in label_rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid label"):
        merge_audit_labels(tmp_path)


def test_merge_rejects_changed_packet(tmp_path):
    build_audit_packets(_frame(), tmp_path, max_rows=1, max_chars=10_000)
    packet = tmp_path / "packets" / "packet-0000.json"
    packet.write_text(packet.read_text() + " ")

    with pytest.raises(ValueError, match="checksum mismatch"):
        merge_audit_labels(tmp_path)
