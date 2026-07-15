from __future__ import annotations

import pandas as pd
import pytest

from interface_formatting_study.model_profiles import get_model_profile
from interface_formatting_study.model_runner import (
    EXPECTED_ACTIVE_WRAPPERS,
    _read_bound_table,
    _select_mechanistic_work,
    _write_bound_table,
    assert_full_dataset_contract,
    resolve_run_root,
)
from interface_formatting_study.run_identity import build_semantic_identity


def _identity(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("row\n")
    source = tmp_path / "code.py"
    source.write_text("x = 1\n")
    return build_semantic_identity(
        get_model_profile("phi"),
        config={"seed": 1729},
        dataset_path=dataset,
        source_paths=[source],
    )


def test_named_canaries_are_isolated_from_each_other_and_full_run(tmp_path):
    profile = get_model_profile("phi")
    identity = _identity(tmp_path)
    full = resolve_run_root(profile, identity, project_root=tmp_path, canary=False, canary_name="ignored")
    functional = resolve_run_root(profile, identity, project_root=tmp_path, canary=True, canary_name="functional")
    profiling = resolve_run_root(profile, identity, project_root=tmp_path, canary=True, canary_name="profile-bs32")

    assert len({full, functional, profiling}) == 3
    assert functional.parent.name == "canaries"
    with pytest.raises(ValueError):
        resolve_run_root(profile, identity, project_root=tmp_path, canary=True, canary_name="../escape")


def test_identity_bound_table_rejects_tampering(tmp_path):
    identity = _identity(tmp_path)
    path = tmp_path / "artifact.parquet"
    frame = pd.DataFrame(
        {
            "value": [1],
            "semantic_run_id": [identity.semantic_run_id],
            "model_id": [identity.payload["model"]["id"]],
            "model_revision": [identity.payload["model"]["revision"]],
        }
    )
    _write_bound_table(frame, path, identity)
    pd.testing.assert_frame_equal(_read_bound_table(path, identity, expected_rows=1), frame)

    pd.DataFrame({"value": [2]}).to_parquet(path, index=False)
    with pytest.raises(RuntimeError, match="checksum"):
        _read_bound_table(path, identity)


def test_full_dataset_contract_rejects_truncated_valid_shape():
    rows = []
    for wrapper in sorted(EXPECTED_ACTIVE_WRAPPERS):
        for index in range(2):
            rows.append({"item_id": f"item-{index}", "wrapper_name": wrapper})
    with pytest.raises(RuntimeError, match="24,000"):
        assert_full_dataset_contract(pd.DataFrame(rows), sorted(EXPECTED_ACTIVE_WRAPPERS))


def test_behavioral_only_run_does_not_require_conflict_pairs():
    empty = pd.DataFrame()
    diagnostic, controls = _select_mechanistic_work(
        empty,
        phases=("preflight", "behavioral", "manifest"),
        canary=True,
        split="validation",
        diagnostic_cap=1,
        patching_cap=1,
        seed=1729,
    )
    assert diagnostic.empty
    assert controls.empty

    with pytest.raises(RuntimeError, match="target-model conflict"):
        _select_mechanistic_work(
            empty,
            phases=("controls",),
            canary=True,
            split="validation",
            diagnostic_cap=1,
            patching_cap=1,
            seed=1729,
        )
