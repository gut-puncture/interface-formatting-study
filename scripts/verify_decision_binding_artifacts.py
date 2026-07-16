#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path, run_id: str, slug: str, mode: str) -> dict[str, object]:
    identity = json.loads((root / "semantic_identity.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    if identity.get("semantic_run_id") != run_id or identity.get("model", {}).get("slug") != slug:
        raise ValueError("decision-binding semantic identity mismatch")
    if manifest.get("semantic_identity") != identity:
        raise ValueError("run manifest semantic identity mismatch")
    phase_keys: dict[str, set[str]] = {}
    phase_rows: dict[str, int] = {}
    for phase in ("readout", "patches"):
        keys: set[str] = set()
        rows = 0
        for shard in sorted((root / "shards" / phase).glob("shard-*")):
            metadata = json.loads((shard / "manifest.json").read_text(encoding="utf-8"))
            if (
                metadata.get("semantic_run_id") != run_id
                or metadata.get("model_id") != identity["model"]["id"]
                or metadata.get("model_revision") != identity["model"]["revision"]
            ):
                raise ValueError(f"{phase} shard identity mismatch")
            data = shard / "data.parquet"
            if _sha(data) != metadata.get("data_sha256"):
                raise ValueError(f"{phase} shard checksum mismatch")
            data_frame = pd.read_parquet(data)
            if int(metadata.get("row_count", -1)) != len(data_frame):
                raise ValueError(f"{phase} shard row count mismatch")
            work_keys = {str(value) for value in metadata.get("work_keys", [])}
            if not work_keys or keys.intersection(work_keys):
                raise ValueError(f"{phase} contains duplicate or empty work keys")
            keys.update(work_keys)
            rows += len(data_frame)
        phase_keys[phase] = keys
        phase_rows[phase] = rows
    for phase, artifact_name in (("readout", "readout_scores.parquet"), ("patches", "patch_results.parquet")):
        artifact = root / artifact_name
        if artifact.exists():
            expected_sha = manifest.get("artifacts", {}).get(f"{artifact.stem}_sha256")
            if expected_sha != _sha(artifact):
                raise ValueError(f"{phase} artifact checksum mismatch")
            frame = pd.read_parquet(artifact)
            if len(frame) != phase_rows[phase] or set(frame["_work_key"].astype(str)) != phase_keys[phase]:
                raise ValueError(f"{phase} merged artifact does not reconcile with shards")
        elif mode == "complete":
            raise ValueError(f"complete fetch is missing {artifact_name}")
    if mode == "complete" and manifest.get("status") != "complete":
        raise ValueError("decision-binding run is not complete")
    if mode == "complete":
        expected_readout_rows = int(manifest.get("expected_readout_rows", -1))
        expected_patch_rows = int(manifest.get("expected_patch_rows", -1))
        if phase_rows["readout"] != expected_readout_rows or phase_rows["patches"] != expected_patch_rows:
            raise ValueError("complete artifact does not match expected row count")
        if (
            len(phase_keys["readout"]) != int(manifest.get("expected_readout_chunks", -1))
            or len(phase_keys["readout"]) != int(manifest.get("completed_readout_chunks", -2))
        ):
            raise ValueError("completed readout work count does not match manifest")
        if (
            len(phase_keys["patches"]) != int(manifest.get("selected_pairs", -1))
            or len(phase_keys["patches"]) != int(manifest.get("completed_pairs", -2))
        ):
            raise ValueError("completed patch work count does not match manifest")

        readout_frame = pd.read_parquet(root / "readout_scores.parquet")
        readout_key = ["readout_work_key", "layer", "checkpoint"]
        if set(readout_key) - set(readout_frame.columns) or readout_frame.duplicated(readout_key).any():
            raise ValueError("readout artifact has missing or duplicate structural rows")

        bank_path = root / "probe_bank.npz"
        selection_path = root / "frozen_selection.json"
        if not bank_path.exists() or not selection_path.exists():
            raise ValueError("complete fetch is missing probe or frozen selection artifacts")
        artifacts = manifest.get("artifacts", {})
        if (
            artifacts.get("probe_bank_sha256") != _sha(bank_path)
            or artifacts.get("frozen_selection_sha256") != _sha(selection_path)
        ):
            raise ValueError("probe or frozen selection checksum mismatch")
        frozen = json.loads(selection_path.read_text(encoding="utf-8"))
        if frozen.get("probe_bank_sha256") != _sha(bank_path):
            raise ValueError("frozen selection is not bound to the probe bank")
        if manifest.get("stage") == "discovery" and frozen.get("readout_scores_sha256") != _sha(
            root / "readout_scores.parquet"
        ):
            raise ValueError("frozen selection is not bound to discovery readout scores")

        expected_combinations = {
            (mechanism, int(layer), condition)
            for mechanism in ("content", "label")
            for layer in frozen["selection"][mechanism]["patch_layers"]
            for condition in ("unpatched", "identity", "probe", "full", "random")
        }
        patch_frame = pd.read_parquet(root / "patch_results.parquet")
        structural_columns = {"pair_work_key", "mechanism", "layer", "condition"}
        if structural_columns - set(patch_frame.columns):
            raise ValueError("patch artifact is missing structural columns")
        for pair_key, group in patch_frame.groupby("pair_work_key", sort=False):
            observed = set(
                zip(
                    group["mechanism"].astype(str),
                    group["layer"].astype(int),
                    group["condition"].astype(str),
                    strict=True,
                )
            )
            if observed != expected_combinations or len(group) != len(expected_combinations):
                raise ValueError(f"patch pair {pair_key} has incomplete structural conditions")
        if int(manifest.get("patch_rows_per_pair", -1)) != len(expected_combinations):
            raise ValueError("patch row cardinality does not match frozen selection")
    checksums = [
        f"{_sha(path)}  {path.relative_to(root)}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "LOCAL_SHA256SUMS.txt"
    ]
    (root / "LOCAL_SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    return {"status": manifest.get("status"), "readout_chunks": len(phase_keys["readout"]), "patch_pairs": len(phase_keys["patches"])}


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
