from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from interface_formatting_study.causal_option_audit import (
    build_causal_option_packets,
    load_causal_option_annotations,
)
from interface_formatting_study.causal_option_maps import transform_with_option_map


SUFFIX = "\n\nReturn only the letter (A, B, C, or D).\nAnswer: "


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "item_id": "item-1",
            "wrapper_name": "key_equals",
            "question": "Choose.",
            "choices": ["one", "two", "three", "four"],
            "wrapped_prompt": "Q=Choose.\nA=one\nB=two\nC=three\nD=four" + SUFFIX,
        }]
    )


def _applicability() -> pd.DataFrame:
    prompt = _frame().iloc[0]["wrapped_prompt"]
    return pd.DataFrame(
        [{
            "item_id": "item-1",
            "wrapper_name": "key_equals",
            "source_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "position_applicable": False,
            "label_applicable": False,
            "not_applicable_reason": "option_structure_not_resolved",
        }]
    )


def test_packets_are_outcome_blind_bounded_and_source_bound(tmp_path):
    manifest = build_causal_option_packets(
        _frame(), _applicability(), tmp_path, max_rows=1, max_chars=10_000
    )

    packet = json.loads((tmp_path / "packets" / "packet-0000.json").read_text())
    row = packet["rows"][0]
    assert manifest["row_count"] == 1
    assert row["source_prompt"] == _frame().iloc[0]["wrapped_prompt"]
    assert row["canonical_choices"] == ["one", "two", "three", "four"]
    assert "correct" not in json.dumps(packet).lower()
    assert "prediction" not in json.dumps(packet).lower()


def test_valid_luna_span_map_loads_and_transforms_exact_source(tmp_path):
    build_causal_option_packets(_frame(), _applicability(), tmp_path)
    prompt = _frame().iloc[0]["wrapped_prompt"]
    slots = []
    for content_id, (label, payload) in enumerate(zip("ABCD", ("one", "two", "three", "four"))):
        label_start = prompt.index(f"{label}={payload}")
        payload_start = label_start + 2
        slots.append({
            "content_id": content_id,
            "label_spans": [[label_start, label_start + 1]],
            "payload_spans": [[payload_start, payload_start + len(payload)]],
        })
    labels = tmp_path / "labels"
    labels.mkdir()
    labels.joinpath("packet-0000.jsonl").write_text(json.dumps({
        "annotation_id": "item-1::key_equals",
        "source_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "mode": "span_map",
        "confidence": "high",
        "reason": "Four explicit assignments.",
        "representations": [{"slots": slots}],
        "model": "gpt-5.6-luna",
        "reasoning_effort": "xhigh",
        "thread_id": "thread-1",
    }) + "\n")

    parsed = load_causal_option_annotations(tmp_path)[("item-1", "key_equals")]
    transformed = transform_with_option_map(parsed, prompt, position_shift=1, label_shift=0)

    assert transformed.startswith("Q=Choose.\nD=four\nA=one\nB=two\nC=three")
    assert parsed.provenance == "gpt-5.6-luna:xhigh:thread-1"


def test_annotation_rejects_changed_source_and_fabricated_model_metadata(tmp_path):
    build_causal_option_packets(_frame(), _applicability(), tmp_path)
    labels = tmp_path / "labels"
    labels.mkdir()
    labels.joinpath("bad.jsonl").write_text(json.dumps({
        "annotation_id": "item-1::key_equals",
        "source_prompt_sha256": "0" * 64,
        "mode": "not_applicable",
        "confidence": "high",
        "reason": "No labels.",
        "model": "some-other-model",
        "reasoning_effort": "max",
        "thread_id": "thread-1",
    }) + "\n")

    with pytest.raises(ValueError, match="source prompt checksum"):
        load_causal_option_annotations(tmp_path)


def test_luna_can_confirm_that_independent_arms_are_not_applicable(tmp_path):
    build_causal_option_packets(_frame(), _applicability(), tmp_path)
    prompt = _frame().iloc[0]["wrapped_prompt"]
    labels = tmp_path / "labels"
    labels.mkdir()
    labels.joinpath("packet-0000.jsonl").write_text(json.dumps({
        "annotation_id": "item-1::key_equals",
        "source_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "mode": "not_applicable",
        "confidence": "high",
        "reason": "No independent labels and positions are present.",
        "model": "gpt-5.6-luna",
        "reasoning_effort": "xhigh",
        "thread_id": "thread-2",
    }) + "\n")

    parsed = load_causal_option_annotations(tmp_path)[("item-1", "key_equals")]

    assert not parsed.separable
    assert not parsed.representations
    assert parsed.not_applicable_reason == "No independent labels and positions are present."
