#!/usr/bin/env python3
"""Validate a fetched model run before the paid GPU is released."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED_PHASES = {"behavioral", "vanilla", "controls", "attention"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path, expected_run_id: str, expected_slug: str, fetch_mode: str) -> dict[str, object]:
    identity_path = root / "semantic_identity.json"
    if not identity_path.exists():
        raise ValueError(f"missing semantic identity: {identity_path}")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity.get("semantic_run_id") != expected_run_id:
        raise ValueError(f"semantic id mismatch: {identity.get('semantic_run_id')} != {expected_run_id}")
    expected_identity = {
        "semantic_run_id": expected_run_id,
        "semantic_sha256": identity.get("semantic_sha256"),
        "model_id": identity.get("model", {}).get("id"),
        "model_revision": identity.get("model", {}).get("revision"),
    }
    if not all(expected_identity.values()):
        raise ValueError("semantic identity is missing model or checksum fields")

    seen_by_phase: dict[str, dict[str, str]] = {}
    verified = 0
    for manifest_path in sorted(root.glob("shards/*/shard-*/manifest.json")):
        phase = manifest_path.parents[1].name
        if phase not in REQUIRED_PHASES:
            raise ValueError(f"unexpected full-run shard phase: {phase}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for field, expected in expected_identity.items():
            if manifest.get(field) != expected:
                raise ValueError(f"shard {field} mismatch: {manifest_path}")
        data_path = manifest_path.parent / "data.parquet"
        if not data_path.exists():
            raise ValueError(f"missing shard data: {data_path}")
        digest = _sha256(data_path)
        if digest != manifest.get("data_sha256"):
            raise ValueError(f"checksum mismatch: {data_path}")
        phase_seen = seen_by_phase.setdefault(phase, {})
        work_keys = [str(key) for key in manifest.get("work_keys", [])]
        if not work_keys or len(work_keys) != len(set(work_keys)):
            raise ValueError(f"shard has missing or duplicate work keys: {manifest_path}")
        for work_key in work_keys:
            owner = phase_seen.get(work_key)
            if owner is not None and owner != digest:
                raise ValueError(f"conflicting duplicate work key in {phase}: {work_key}")
            phase_seen[work_key] = digest
        verified += 1

    if fetch_mode == "complete":
        run_manifest_path = root / "metadata" / "run_manifest.json"
        if not run_manifest_path.exists():
            raise ValueError("missing full-run manifest")
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if run_manifest.get("status") != "complete" or run_manifest.get("canary") is not False:
            raise ValueError(f"run is not full-complete: {run_manifest.get('status')}")
        if run_manifest.get("dataset_rows") != 24000:
            raise ValueError(f"full run has wrong dataset row count: {run_manifest.get('dataset_rows')}")
        model = run_manifest.get("model", {})
        if (
            model.get("slug") != expected_slug
            or model.get("id") != expected_identity["model_id"]
            or model.get("revision") != expected_identity["model_revision"]
        ):
            raise ValueError("model identity mismatch in full-run manifest")
        manifest_identity = run_manifest.get("semantic_identity", {})
        if manifest_identity != identity:
            raise ValueError("full-run manifest semantic identity differs from root identity")
        completion = run_manifest.get("phase_completion", {})
        if set(completion) != REQUIRED_PHASES or not all(entry.get("complete") for entry in completion.values()):
            raise ValueError(f"required phase completion missing: {completion}")
        for phase in sorted(REQUIRED_PHASES):
            entry = completion[phase]
            completed = int(entry.get("completed", -1))
            total = int(entry.get("total", -1))
            fetched = len(seen_by_phase.get(phase, {}))
            if completed != total or fetched != completed:
                raise ValueError(
                    f"{phase} shard work-key count mismatch: fetched={fetched}, completed={completed}, total={total}"
                )
        if int(completion["behavioral"]["completed"]) != 24000:
            raise ValueError("behavioral phase does not contain exactly 24,000 completed work keys")

        required_artifacts = [
            "raw/behavioral_scores.parquet",
            "processed/conflict_pairs.parquet",
            "processed/vanilla_convergence.parquet",
            "processed/focused_patching_controls.parquet",
            "processed/attention_diagnostics.parquet",
        ]
        for relative in required_artifacts:
            path = root / relative
            artifact_manifest_path = path.with_name(path.name + ".manifest.json")
            if not path.exists() or not artifact_manifest_path.exists():
                raise ValueError(f"missing required final artifact: {relative}")
            artifact_manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
            if (
                any(artifact_manifest.get(field) != expected for field, expected in expected_identity.items())
                or artifact_manifest.get("sha256") != _sha256(path)
            ):
                raise ValueError(f"final artifact validation failed: {relative}")
    elif fetch_mode == "partial":
        if verified == 0:
            print("warning: partial fetch contains no full-run shards")
    else:
        raise ValueError("fetch mode must be complete or partial")

    checksum_lines = [
        f"{_sha256(path)}  {path.relative_to(root)}"
        for path in sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.name != "LOCAL_SHA256SUMS.txt"
        )
    ]
    (root / "LOCAL_SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return {
        "semantic_run_id": expected_run_id,
        "mode": fetch_mode,
        "verified_shards": verified,
        "work_keys_by_phase": {phase: len(keys) for phase, keys in sorted(seen_by_phase.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("semantic_run_id")
    parser.add_argument("model_slug")
    parser.add_argument("fetch_mode", choices=("complete", "partial"))
    args = parser.parse_args()
    try:
        result = verify(args.root, args.semantic_run_id, args.model_slug, args.fetch_mode)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
