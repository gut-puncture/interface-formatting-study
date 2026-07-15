#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from interface_formatting_study.run_identity import sha256_file


def verify(root: Path, semantic_run_id: str, model_slug: str, mode: str) -> dict[str, object]:
    manifest_path = root / "run_manifest.json"
    identity_path = root / "semantic_identity.json"
    if not manifest_path.exists() or not identity_path.exists():
        raise ValueError("Fetched causal run is missing its run manifest or semantic identity")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity.get("semantic_run_id") != semantic_run_id:
        raise ValueError("semantic_run_id mismatch")
    if identity.get("model", {}).get("slug") != model_slug:
        raise ValueError("model slug mismatch")
    if manifest.get("semantic_identity") != identity:
        raise ValueError("run manifest semantic identity mismatch")

    model_id = identity["model"]["id"]
    revision = identity["model"]["revision"]
    shard_root = root / "shards" / "causal_behavior"
    keys: set[str] = set()
    shard_rows = 0
    for shard in sorted(shard_root.glob("shard-*")):
        shard_manifest = json.loads((shard / "manifest.json").read_text(encoding="utf-8"))
        if (
            shard_manifest.get("semantic_run_id") != semantic_run_id
            or shard_manifest.get("model_id") != model_id
            or shard_manifest.get("model_revision") != revision
        ):
            raise ValueError(f"shard identity mismatch: {shard.name}")
        data = shard / "data.parquet"
        if sha256_file(data) != shard_manifest.get("data_sha256"):
            raise ValueError(f"shard checksum mismatch: {shard.name}")
        shard_keys = [str(value) for value in shard_manifest.get("work_keys", [])]
        overlap = keys.intersection(shard_keys)
        if overlap:
            raise ValueError(f"duplicate causal work keys across shards: {sorted(overlap)[:3]}")
        if len(shard_keys) != int(shard_manifest.get("row_count", -1)):
            raise ValueError(f"causal shard work-key/row mismatch: {shard.name}")
        keys.update(shard_keys)
        shard_rows += len(pd.read_parquet(data))

    artifact = root / "raw" / "causal_behavior.parquet"
    artifact_manifest = json.loads(artifact.with_name(artifact.name + ".manifest.json").read_text(encoding="utf-8"))
    if artifact_manifest.get("semantic_run_id") != semantic_run_id or sha256_file(artifact) != artifact_manifest.get("sha256"):
        raise ValueError("merged causal artifact identity or checksum mismatch")
    merged = pd.read_parquet(artifact)
    if len(merged) != len(keys) or len(merged) != shard_rows or set(merged["work_key"].astype(str)) != keys:
        raise ValueError("merged causal artifact does not reconcile with shards")
    expected = int(manifest["design"]["run_rows"])
    if mode == "complete" and (manifest.get("status") != "complete" or len(keys) != expected):
        raise ValueError(f"complete causal fetch expected {expected} work keys; found {len(keys)}")
    return {"status": manifest.get("status"), "work_keys": len(keys), "expected_work_keys": expected}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("semantic_run_id")
    parser.add_argument("model_slug")
    parser.add_argument("mode", choices=("complete", "partial"))
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.semantic_run_id, args.model_slug, args.mode), sort_keys=True))


if __name__ == "__main__":
    main()
