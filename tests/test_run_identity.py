from __future__ import annotations

import json
import shutil

import pandas as pd
import pytest

from interface_formatting_study.model_profiles import get_model_profile
from interface_formatting_study.run_identity import build_semantic_identity
from interface_formatting_study.shards import ShardConflictError, ShardIdentityError, ShardStore, run_sharded_phase


def _identity(tmp_path, *, profile_name="phi", seed=1729, dataset_text="row\n", sequence_batch_size=4):
    dataset = tmp_path / f"dataset-{profile_name}-{seed}.jsonl"
    dataset.write_text(dataset_text)
    config = {
        "seed": seed,
        "labels": ["A", "B", "C", "D"],
        "scoring": {
            "answer_variant_policy": "canonical_letter_only",
            "sequence_batch_size": sequence_batch_size,
            "max_batch_tokens": 100,
        },
        "focused_mechanistic": {"anchors": ["options_end"], "diagnostic_cap": 300},
        "outputs": {"raw_dir": "ignored"},
    }
    source = tmp_path / "semantic.py"
    source.write_text("meaning = 1\n")
    return build_semantic_identity(
        get_model_profile(profile_name),
        config=config,
        dataset_path=dataset,
        source_paths=[source],
    )


def test_semantic_identity_binds_model_config_dataset_code_and_numeric_batch_profile(tmp_path):
    base = _identity(tmp_path)
    assert base.semantic_run_id == _identity(tmp_path).semantic_run_id
    assert base.semantic_run_id != _identity(tmp_path, profile_name="mistral").semantic_run_id
    assert base.semantic_run_id != _identity(tmp_path, seed=7).semantic_run_id
    assert base.semantic_run_id != _identity(tmp_path, dataset_text="changed\n").semantic_run_id

    payload = json.loads(json.dumps(base.payload))
    assert payload["experiment_config"]["scoring"]["sequence_batch_size"] == 4
    assert payload["experiment_config"]["scoring"]["max_batch_tokens"] == 100
    assert "outputs" not in payload["experiment_config"]

    assert _identity(tmp_path, sequence_batch_size=8).semantic_run_id != base.semantic_run_id


def test_shards_reject_wrong_identity_and_conflicting_duplicates(tmp_path):
    identity = _identity(tmp_path)
    phase_root = tmp_path / "run" / "behavioral_shards"
    store = ShardStore(phase_root, identity)
    first = store.write_shard(pd.DataFrame({"value": [1]}), work_keys=["row:1"])

    duplicate = phase_root / "duplicate-copy"
    shutil.copytree(first, duplicate)
    merged = store.merge(sort_by=["value"])
    assert merged["value"].tolist() == [1]

    store.write_shard(pd.DataFrame({"value": [2]}), work_keys=["row:1"], shard_hint="conflict")
    with pytest.raises(ShardConflictError, match="row:1"):
        store.merge()

    with pytest.raises(ShardIdentityError):
        ShardStore(phase_root, _identity(tmp_path, profile_name="mistral"))


def test_forced_stop_then_resume_matches_uninterrupted_merge(tmp_path):
    identity = _identity(tmp_path)
    items = [(f"pair:{index}", index) for index in range(7)]

    full = ShardStore(tmp_path / "full", identity)
    run_sharded_phase(
        full,
        items,
        lambda key, value: pd.DataFrame({"work_key": [key], "value": [value * 2]}),
        max_work_units=3,
    )

    resumed = ShardStore(tmp_path / "resumed", identity)
    seen = 0

    def stop_after_two():
        return seen >= 2

    def process(key, value):
        nonlocal seen
        seen += 1
        return pd.DataFrame({"work_key": [key], "value": [value * 2]})

    run_sharded_phase(resumed, items, process, max_work_units=3, should_stop=stop_after_two)
    assert resumed.completed_work_keys() == {"pair:0", "pair:1"}
    run_sharded_phase(
        resumed,
        items,
        lambda key, value: pd.DataFrame({"work_key": [key], "value": [value * 2]}),
        max_work_units=3,
    )

    pd.testing.assert_frame_equal(
        full.merge(sort_by=["work_key", "value"]),
        resumed.merge(sort_by=["work_key", "value"]),
    )
