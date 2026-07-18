from __future__ import annotations

import json

import pandas as pd
import pytest

from interface_formatting_study.decision_binding_content_cli import (
    _load_activation_shard,
    _load_prepared_bundle,
    _save_activation_shard,
    _select_canary_sites,
    build_parser,
    verify_run_root,
)
from interface_formatting_study.run_identity import sha256_file


def test_content_cli_exposes_preparation_model_run_and_verification():
    parser = build_parser()

    prepared = parser.parse_args([
        "prepare", "--bundle", "bundle", "--applicability", "status",
        "--option-audit", "audit", "--output-dir", "out",
    ])
    run = parser.parse_args([
        "run-model", "--profile", "mistral", "--bundle", "prepared",
    ])
    verify = parser.parse_args([
        "verify", "--run-root", "run", "--run-id", "id", "--mode", "canary",
    ])

    assert prepared.command == "prepare"
    assert run.command == "run-model"
    assert verify.command == "verify"
    assert run.l2_grid == [1e-4, 1e-3, 1e-2, 1e-1]
    with pytest.raises(SystemExit):
        parser.parse_args(["run-model", "--profile", "mistral", "--bundle", "x", "--stage", "confirmation"])


def test_prepared_bundle_is_checksum_and_role_bound(tmp_path):
    sites = pd.DataFrame({
        "work_key": ["a", "b", "c"],
        "item_id": ["train", "select", "gate"],
        "readout_role": ["probe_train", "layer_select", "reader_gate"],
    })
    path = tmp_path / "candidate_sites.parquet"
    sites.to_parquet(path, index=False)
    manifest = {
        "schema_version": 1,
        "stage": "discovery",
        "rows": 3,
        "sha256": sha256_file(path),
        "role_items": {"probe_train": 1, "layer_select": 1, "reader_gate": 1},
        "source_model": {"id": "model", "revision": "revision", "slug": "slug"},
    }
    (tmp_path / "prepared_manifest.json").write_text(json.dumps(manifest))

    loaded_manifest, loaded = _load_prepared_bundle(tmp_path)

    assert loaded_manifest == manifest
    assert loaded["work_key"].tolist() == ["a", "b", "c"]
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(RuntimeError, match="checksum"):
        _load_prepared_bundle(tmp_path)


def test_canary_sampling_is_role_balanced_and_never_changes_source_rows():
    rows = []
    for role in ("probe_train", "layer_select", "reader_gate"):
        for item in range(8):
            for variant in range(3):
                rows.append({
                    "work_key": f"{role}|{item}|{variant}",
                    "item_id": f"{role}|{item}",
                    "readout_role": role,
                    "variant": variant,
                    "prompt": (
                        f"exact-{role}-{item}-{variant}-" + ("λ" if item == 1 else "")
                        + ("x" * 200 if item == 2 else "")
                    ),
                    "candidate_texts": ["one", "two words", "three", "four"],
                })
    sites = pd.DataFrame(rows)

    canary = _select_canary_sites(sites, items_per_role=3, seed=7)

    assert canary.groupby("readout_role")["item_id"].nunique().to_dict() == {
        "layer_select": 3,
        "probe_train": 3,
        "reader_gate": 3,
    }
    expected = sites.set_index("work_key")["prompt"]
    assert all(expected[row.work_key] == row.prompt for row in canary.itertuples())
    assert canary["prompt"].str.contains("λ").any()
    assert canary["prompt"].str.len().max() > 200


def test_activation_shards_are_identity_and_checksum_bound(tmp_path):
    path = tmp_path / "capture.pt"
    values = __import__("torch").arange(24).reshape(1, 2, 4, 3)
    _save_activation_shard(
        path, activations=values, work_keys=["work"], semantic_sha256="a" * 64
    )

    loaded = _load_activation_shard(
        path, work_keys=["work"], semantic_sha256="a" * 64
    )

    assert loaded.equal(values)
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(RuntimeError, match="checksum"):
        _load_activation_shard(path, work_keys=["work"], semantic_sha256="a" * 64)


def test_run_verifier_checks_identity_artifacts_and_claim_boundary(tmp_path):
    identity = {
        "semantic_run_id": "run-id",
        "semantic_sha256": "a" * 64,
        "model": {"id": "model", "revision": "revision", "slug": "slug"},
    }
    (tmp_path / "semantic_identity.json").write_text(json.dumps(identity))
    gate = {
        "claim": "candidate_local_linear_decodability",
        "tier_1_pass": False,
        "tier_2_pass": False,
        "patch_eligible": False,
        "opens_final_confirmation": False,
    }
    (tmp_path / "gate_report.json").write_text(json.dumps(gate))
    (tmp_path / "frozen_selection.json").write_text("{}")
    (tmp_path / "ranker_manifest.json").write_text(json.dumps({
        "schema_version": 1, "semantic_sha256": "a" * 64, "rankers": {},
    }))
    scores = pd.DataFrame({
        "work_key": ["a"], "reader_name": ["content"], "l2": [0.1],
        "layer": [1], "readout_role": ["layer_select"],
    })
    scores.to_parquet(tmp_path / "layer_select_scores.parquet", index=False)
    controls = scores.assign(reader_name="majority")
    controls.to_parquet(tmp_path / "layer_select_control_scores.parquet", index=False)
    names = [
        "semantic_identity.json", "frozen_selection.json", "gate_report.json",
        "ranker_manifest.json", "layer_select_scores.parquet",
        "layer_select_control_scores.parquet",
    ]
    artifacts = {}
    for name in names:
        path = tmp_path / name
        artifacts[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        if path.suffix == ".parquet":
            artifacts[name]["rows"] = 1
            artifacts[name]["work_keys"] = 1
    manifest = {
        "status": "content_readout_complete",
        "semantic_identity": identity,
        "canary": False,
        "role_items": {"probe_train": 1801, "layer_select": 300, "reader_gate": 300},
        "tier_1_pass": False,
        "tier_2_pass": False,
        "patch_eligible": False,
        "final_confirmation_opened": False,
        "reader_gate_opened": False,
        "artifacts": artifacts,
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))

    verified = verify_run_root(tmp_path, expected_run_id="run-id", mode="complete")

    assert verified["status"] == "content_readout_complete"
    complete_artifacts = dict(artifacts)
    manifest["artifacts"] = {
        name: receipt
        for name, receipt in complete_artifacts.items()
        if name != "frozen_selection.json"
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="required artifacts"):
        verify_run_root(tmp_path, expected_run_id="run-id", mode="complete")
    manifest["artifacts"] = complete_artifacts
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "gate_report.json").write_text("{}")
    with pytest.raises(RuntimeError, match="checksum"):
        verify_run_root(tmp_path, expected_run_id="run-id", mode="complete")
