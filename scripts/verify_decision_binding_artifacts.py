#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


BANK_NAMES = ("content", "position", "label", "legacy_content")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_phase(
    root: Path,
    phase: str,
    *,
    run_id: str,
    identity: dict[str, object],
) -> tuple[set[str], pd.DataFrame]:
    keys: set[str] = set()
    frames: list[pd.DataFrame] = []
    exact_shards: set[tuple[str, tuple[str, ...]]] = set()
    for shard in sorted((root / "shards" / phase).glob("shard-*")):
        manifest_path = shard / "manifest.json"
        data_path = shard / "data.parquet"
        if not manifest_path.exists() or not data_path.exists():
            raise ValueError(f"{phase} contains an incomplete shard")
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            metadata.get("semantic_run_id") != run_id
            or metadata.get("semantic_sha256") != identity.get("semantic_sha256")
            or metadata.get("model_id") != identity["model"]["id"]
            or metadata.get("model_revision") != identity["model"]["revision"]
        ):
            raise ValueError(f"{phase} shard identity mismatch")
        data_sha = _sha(data_path)
        if data_sha != metadata.get("data_sha256"):
            raise ValueError(f"{phase} shard checksum mismatch")
        frame = pd.read_parquet(data_path)
        if int(metadata.get("row_count", -1)) != len(frame):
            raise ValueError(f"{phase} shard row count mismatch")
        work_keys = tuple(str(value) for value in metadata.get("work_keys", []))
        if not work_keys or len(set(work_keys)) != len(work_keys):
            raise ValueError(f"{phase} contains duplicate or empty work keys")
        exact_key = (data_sha, work_keys)
        if exact_key in exact_shards:
            continue
        if keys.intersection(work_keys):
            raise ValueError(f"{phase} contains conflicting duplicate work keys")
        if "_work_key" not in frame.columns or set(frame["_work_key"].astype(str)) != set(work_keys):
            raise ValueError(f"{phase} shard rows do not match declared work keys")
        exact_shards.add(exact_key)
        keys.update(work_keys)
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return keys, merged


def _frames_match(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if len(left) != len(right) or set(left.columns) != set(right.columns):
        return False
    columns = sorted(left.columns)
    left_hashes = pd.util.hash_pandas_object(left[columns], index=False).value_counts().sort_index()
    right_hashes = pd.util.hash_pandas_object(right[columns], index=False).value_counts().sort_index()
    return left_hashes.equals(right_hashes)


def _verify_merged(
    root: Path,
    phase: str,
    artifact_name: str,
    shard_frame: pd.DataFrame,
    manifest: dict[str, object],
    *,
    required: bool,
) -> pd.DataFrame | None:
    path = root / artifact_name
    if not path.exists():
        if required:
            raise ValueError(f"fetch is missing {artifact_name}")
        return None
    expected_sha = manifest.get("artifacts", {}).get(f"{path.stem}_sha256")
    if expected_sha is not None and expected_sha != _sha(path):
        raise ValueError(f"{phase} artifact checksum mismatch")
    frame = pd.read_parquet(path)
    if not _frames_match(frame, shard_frame):
        raise ValueError(f"{phase} merged artifact does not reconcile with shards")
    return frame


def _verify_readout_completion(
    root: Path,
    manifest: dict[str, object],
    readout_keys: set[str],
    readout_frame: pd.DataFrame,
) -> dict[str, str]:
    if len(readout_frame) != int(manifest.get("expected_readout_rows", -1)):
        raise ValueError("readout artifact does not match expected row count")
    if (
        len(readout_keys) != int(manifest.get("expected_readout_chunks", -1))
        or len(readout_keys) != int(manifest.get("completed_readout_chunks", -2))
    ):
        raise ValueError("completed readout work count does not match manifest")
    structural_key = ["readout_work_key", "layer", "checkpoint"]
    if set(structural_key) - set(readout_frame.columns) or readout_frame.duplicated(structural_key).any():
        raise ValueError("readout artifact has missing or duplicate structural rows")

    bank_paths = {name: root / f"probe_bank_{name}.npz" for name in BANK_NAMES}
    metadata_path = root / "probe_banks.json"
    selection_path = root / "frozen_selection.json"
    if any(not path.exists() for path in bank_paths.values()) or not metadata_path.exists() or not selection_path.exists():
        raise ValueError("fetch is missing probe or frozen selection artifacts")
    bank_hashes = {name: _sha(path) for name, path in bank_paths.items()}
    artifacts = manifest.get("artifacts", {})
    expected = {
        **{f"probe_bank_{name}_sha256": digest for name, digest in bank_hashes.items()},
        "probe_banks_sha256": _sha(metadata_path),
        "frozen_selection_sha256": _sha(selection_path),
    }
    if any(artifacts.get(key) != digest for key, digest in expected.items()):
        raise ValueError("probe or frozen selection checksum mismatch")
    bank_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frozen = json.loads(selection_path.read_text(encoding="utf-8"))
    if bank_metadata.get("probe_bank_sha256") != bank_hashes:
        raise ValueError("probe bank metadata is not bound to all coordinate banks")
    if frozen.get("probe_bank_sha256") != bank_hashes:
        raise ValueError("frozen selection is not bound to all coordinate banks")
    if manifest.get("stage") == "discovery" and frozen.get("readout_scores_sha256") != _sha(
        root / "readout_scores.parquet"
    ):
        raise ValueError("frozen selection is not bound to discovery readout scores")
    return bank_hashes


def _verify_patch_completion(
    root: Path,
    manifest: dict[str, object],
    patch_keys: set[str],
    patch_frame: pd.DataFrame,
) -> None:
    if len(patch_frame) != int(manifest.get("expected_patch_rows", -1)):
        raise ValueError("complete artifact does not match expected row count")
    if (
        len(patch_keys) != int(manifest.get("selected_pairs", -1))
        or len(patch_keys) != int(manifest.get("completed_pairs", -2))
    ):
        raise ValueError("completed patch work count does not match manifest")
    frozen = json.loads((root / "frozen_selection.json").read_text(encoding="utf-8"))
    expected_combinations = {
        (mechanism, int(layer), condition)
        for mechanism in ("content", "label")
        for layer in frozen["selection"][mechanism]["patch_layers"]
        for condition in ("unpatched", "identity", "probe", "full", "random")
    }
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


def verify(root: Path, run_id: str, slug: str, mode: str) -> dict[str, object]:
    identity = json.loads((root / "semantic_identity.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    if identity.get("semantic_run_id") != run_id or identity.get("model", {}).get("slug") != slug:
        raise ValueError("decision-binding semantic identity mismatch")
    if manifest.get("semantic_identity") != identity:
        raise ValueError("run manifest semantic identity mismatch")

    readout_keys, readout_shards = _load_phase(root, "readout", run_id=run_id, identity=identity)
    patch_keys, patch_shards = _load_phase(root, "patches", run_id=run_id, identity=identity)
    strict_readout = mode in {"readout", "complete"}
    readout_frame = _verify_merged(
        root,
        "readout",
        "readout_scores.parquet",
        readout_shards,
        manifest,
        required=strict_readout,
    )
    patch_frame = _verify_merged(
        root,
        "patches",
        "patch_results.parquet",
        patch_shards,
        manifest,
        required=mode == "complete",
    )

    status = manifest.get("status")
    if mode == "readout" and status not in {"readout_complete", "complete"}:
        raise ValueError("decision-binding run is not readout complete")
    if mode == "complete" and status != "complete":
        raise ValueError("decision-binding run is not complete")
    if strict_readout:
        assert readout_frame is not None
        _verify_readout_completion(root, manifest, readout_keys, readout_frame)
    if mode == "complete":
        assert patch_frame is not None
        _verify_patch_completion(root, manifest, patch_keys, patch_frame)

    checksums = [
        f"{_sha(path)}  {path.relative_to(root)}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "LOCAL_SHA256SUMS.txt"
    ]
    (root / "LOCAL_SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    return {
        "status": status,
        "readout_chunks": len(readout_keys),
        "patch_pairs": len(patch_keys),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("semantic_run_id")
    parser.add_argument("model_slug")
    parser.add_argument("mode", choices=("complete", "readout", "partial"))
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.semantic_run_id, args.model_slug, args.mode), sort_keys=True))


if __name__ == "__main__":
    main()
