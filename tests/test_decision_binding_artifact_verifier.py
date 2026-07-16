from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from interface_formatting_study.model_profiles import get_model_profile
from interface_formatting_study.run_identity import build_semantic_identity, sha256_file
from interface_formatting_study.shards import ShardStore


SCRIPT = Path(__file__).parents[1] / "scripts" / "verify_decision_binding_artifacts.py"
SPEC = importlib.util.spec_from_file_location("verify_decision_binding_artifacts", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _fixture(root: Path):
    dataset = root.parent / "ledger.parquet"
    source = root.parent / "semantic.py"
    pd.DataFrame({"value": [1]}).to_parquet(dataset, index=False)
    source.write_text("meaning = 1\n")
    profile = get_model_profile("phi")
    identity = build_semantic_identity(
        profile,
        config={"study": "decision_binding_v1"},
        dataset_path=dataset,
        source_paths=[source],
    )
    root.mkdir()
    (root / "semantic_identity.json").write_text(json.dumps(identity.as_dict()))
    readout = ShardStore(root / "shards" / "readout", identity)
    readout_rows = pd.DataFrame(
        [
            {
                "_work_key": "chunk-1",
                "readout_work_key": "row-1",
                "layer": layer,
                "checkpoint": checkpoint,
            }
            for layer in (0, 1)
            for checkpoint in ("format_end", "answer_prefix_end")
        ]
    )
    readout.write_shard(
        readout_rows,
        work_keys=["chunk-1"],
    )
    patches = ShardStore(root / "shards" / "patches", identity)
    patch_rows = pd.DataFrame(
        [
            {
                "_work_key": "pair-1",
                "pair_work_key": "pair-1",
                "mechanism": mechanism,
                "layer": layer,
                "condition": condition,
            }
            for mechanism, layer in (("content", 0), ("label", 1))
            for condition in ("unpatched", "identity", "probe", "full", "random")
        ]
    )
    patches.write_shard(
        patch_rows,
        work_keys=["pair-1"],
    )
    readout.merge().to_parquet(root / "readout_scores.parquet", index=False)
    patches.merge().to_parquet(root / "patch_results.parquet", index=False)
    (root / "probe_bank.npz").write_bytes(b"probe-bank")
    (root / "frozen_selection.json").write_text(
        json.dumps(
            {
                "probe_bank_sha256": sha256_file(root / "probe_bank.npz"),
                "readout_scores_sha256": sha256_file(root / "readout_scores.parquet"),
                "selection": {
                    "content": {"patch_layers": [0]},
                    "label": {"patch_layers": [1]},
                },
            }
        )
    )
    (root / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "stage": "discovery",
                "selected_pairs": 1,
                "completed_pairs": 1,
                "expected_patch_rows": 10,
                "patch_rows_per_pair": 10,
                "expected_readout_rows": 4,
                "expected_readout_chunks": 1,
                "completed_readout_chunks": 1,
                "semantic_identity": identity.as_dict(),
                "artifacts": {
                    "readout_scores_sha256": sha256_file(root / "readout_scores.parquet"),
                    "patch_results_sha256": sha256_file(root / "patch_results.parquet"),
                    "probe_bank_sha256": sha256_file(root / "probe_bank.npz"),
                    "frozen_selection_sha256": sha256_file(root / "frozen_selection.json"),
                },
            }
        )
    )
    return identity.semantic_run_id, profile.slug


def test_decision_binding_verifier_reconciles_identity_shards_and_merged_outputs(tmp_path):
    root = tmp_path / "run"
    run_id, slug = _fixture(root)

    result = MODULE.verify(root, run_id, slug, "complete")

    assert result == {"status": "complete", "readout_chunks": 1, "patch_pairs": 1}
    assert (root / "LOCAL_SHA256SUMS.txt").exists()


def test_decision_binding_verifier_rejects_corruption_and_incomplete_completion(tmp_path):
    root = tmp_path / "run"
    run_id, slug = _fixture(root)
    shard = next((root / "shards" / "patches").glob("shard-*/data.parquet"))
    shard.write_bytes(shard.read_bytes() + b"corrupt")

    with pytest.raises(ValueError, match="checksum mismatch"):
        MODULE.verify(root, run_id, slug, "complete")

    root2 = tmp_path / "partial"
    run_id2, slug2 = _fixture(root2)
    manifest = json.loads((root2 / "run_manifest.json").read_text())
    manifest["status"] = "interrupted"
    (root2 / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not complete"):
        MODULE.verify(root2, run_id2, slug2, "complete")
    assert MODULE.verify(root2, run_id2, slug2, "partial")["status"] == "interrupted"


def test_decision_binding_verifier_rejects_semantically_changed_merged_artifact(tmp_path):
    root = tmp_path / "run"
    run_id, slug = _fixture(root)
    patch = pd.read_parquet(root / "patch_results.parquet")
    patch.loc[0, "condition"] = "changed"
    patch.to_parquet(root / "patch_results.parquet", index=False)

    with pytest.raises(ValueError, match="artifact checksum mismatch"):
        MODULE.verify(root, run_id, slug, "complete")


def test_decision_binding_verifier_rejects_declared_complete_but_structurally_short_output(tmp_path):
    root = tmp_path / "run"
    run_id, slug = _fixture(root)
    manifest = json.loads((root / "run_manifest.json").read_text())
    manifest["expected_patch_rows"] = 10
    manifest["expected_readout_rows"] = 8
    (root / "run_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="expected row count"):
        MODULE.verify(root, run_id, slug, "complete")
