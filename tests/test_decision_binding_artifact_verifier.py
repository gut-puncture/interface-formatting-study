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
    readout.write_shard(
        pd.DataFrame({"_work_key": ["chunk-1", "chunk-1"], "value": [1, 2]}),
        work_keys=["chunk-1"],
    )
    patches = ShardStore(root / "shards" / "patches", identity)
    patches.write_shard(
        pd.DataFrame({"_work_key": ["pair-1", "pair-1"], "value": [3, 4]}),
        work_keys=["pair-1"],
    )
    readout.merge().to_parquet(root / "readout_scores.parquet", index=False)
    patches.merge().to_parquet(root / "patch_results.parquet", index=False)
    (root / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "selected_pairs": 1,
                "semantic_identity": identity.as_dict(),
                "artifacts": {
                    "readout_scores_sha256": sha256_file(root / "readout_scores.parquet"),
                    "patch_results_sha256": sha256_file(root / "patch_results.parquet"),
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
    patch["value"] = [30, 40]
    patch.to_parquet(root / "patch_results.parquet", index=False)

    with pytest.raises(ValueError, match="artifact checksum mismatch"):
        MODULE.verify(root, run_id, slug, "complete")
