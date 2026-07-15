from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from interface_formatting_study.model_profiles import get_model_profile
from interface_formatting_study.run_identity import build_semantic_identity


SCRIPT = Path(__file__).parents[1] / "scripts" / "verify_interface_formatting_study_artifacts.py"
SPEC = importlib.util.spec_from_file_location("verify_interface_formatting_study_artifacts", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _complete_fixture(root: Path) -> tuple[str, str]:
    profile = get_model_profile("phi")
    dataset = root.parent / "dataset.jsonl"
    source = root.parent / "semantic.py"
    dataset.write_text("one row\n")
    source.write_text("meaning = 1\n")
    identity_object = build_semantic_identity(
        profile,
        config={"seed": 1729, "labels": ["A", "B", "C", "D"]},
        dataset_path=dataset,
        source_paths=[source],
    )
    identity = identity_object.as_dict()
    model_id = profile.model_id
    revision = profile.revision
    semantic_id = identity_object.semantic_run_id
    semantic_sha = identity_object.semantic_sha256
    root.mkdir()
    (root / "semantic_identity.json").write_text(json.dumps(identity))
    counts = {"behavioral": 24000, "vanilla": 2, "controls": 2, "attention": 2}
    for phase, count in counts.items():
        shard = root / "shards" / phase / "shard-one"
        shard.mkdir(parents=True)
        pd.DataFrame({"value": range(count)}).to_parquet(shard / "data.parquet", index=False)
        (shard / "manifest.json").write_text(
            json.dumps(
                {
                    "semantic_run_id": semantic_id,
                    "semantic_sha256": semantic_sha,
                    "model_id": model_id,
                    "model_revision": revision,
                    "work_keys": [f"{phase}:{index}" for index in range(count)],
                    "data_sha256": _sha(shard / "data.parquet"),
                }
            )
        )
    artifacts = [
        root / "raw" / "behavioral_scores.parquet",
        root / "processed" / "conflict_pairs.parquet",
        root / "processed" / "vanilla_convergence.parquet",
        root / "processed" / "focused_patching_controls.parquet",
        root / "processed" / "attention_diagnostics.parquet",
    ]
    for artifact in artifacts:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"value": [1]}).to_parquet(artifact, index=False)
        artifact.with_name(artifact.name + ".manifest.json").write_text(
            json.dumps(
                {
                    "semantic_run_id": semantic_id,
                    "semantic_sha256": semantic_sha,
                    "model_id": model_id,
                    "model_revision": revision,
                    "sha256": _sha(artifact),
                }
            )
        )
    (root / "metadata").mkdir()
    (root / "metadata" / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "canary": False,
                "dataset_rows": 24000,
                "model": {"slug": profile.slug, "id": model_id, "revision": revision},
                "semantic_identity": identity,
                "phase_completion": {
                    phase: {"completed": count, "total": count, "complete": True}
                    for phase, count in counts.items()
                },
            }
        )
    )
    return semantic_id, profile.slug


def test_complete_fetch_reconciles_every_phase_and_identity(tmp_path):
    root = tmp_path / "run"
    semantic_id, slug = _complete_fixture(root)

    result = MODULE.verify(root, semantic_id, slug, "complete")

    assert result["work_keys_by_phase"] == {
        "attention": 2,
        "behavioral": 24000,
        "controls": 2,
        "vanilla": 2,
    }


def test_complete_fetch_rejects_missing_mechanistic_shard(tmp_path):
    root = tmp_path / "run"
    semantic_id, slug = _complete_fixture(root)
    shard = root / "shards" / "attention" / "shard-one"
    for path in shard.iterdir():
        path.unlink()
    shard.rmdir()

    with pytest.raises(ValueError, match="attention shard work-key count mismatch"):
        MODULE.verify(root, semantic_id, slug, "complete")


def test_complete_fetch_rejects_wrong_shard_model_identity(tmp_path):
    root = tmp_path / "run"
    semantic_id, slug = _complete_fixture(root)
    manifest_path = root / "shards" / "vanilla" / "shard-one" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model_revision"] = "wrong"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="model_revision mismatch"):
        MODULE.verify(root, semantic_id, slug, "complete")
