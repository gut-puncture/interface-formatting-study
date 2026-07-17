from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import signal
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .decision_binding import (
    capture_layer_readouts,
    evaluate_probe_banks,
    fit_coordinate_probe_bank,
    load_probe_bank,
    prepare_patch_pair_ledger,
    prepare_readout_ledger,
    run_patch_pair,
    save_probe_bank,
    select_readout_layers,
    validate_canonical_sources,
)
from .model_loader import load_model_and_tokenizer
from .model_profiles import MODEL_PROFILES, get_model_profile
from .model_runner import StopState
from .run_identity import build_semantic_identity, default_semantic_source_paths, sha256_file
from .scoring import tokenize_text
from .shards import ShardStore, run_sharded_phase
from .utils import read_table, write_table_atomic


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_bundle(root: Path) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    manifest_path = root / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("bundle_schema_version") != 3 or "source_validation" not in manifest:
        raise RuntimeError("Decision-binding bundle lacks canonical source attestation")
    scored_attestation = manifest["source_validation"].get("scored_artifact", {})
    required_attestation = {
        "manifest_sha256",
        "semantic_run_id",
        "model_id",
        "model_revision",
        "rows",
        "sha256",
    }
    if not required_attestation.issubset(scored_attestation):
        raise RuntimeError("Decision-binding bundle lacks scored source attestation")
    bundle_model = manifest.get("model", {})
    if any(
        str(bundle_model.get(bundle_key)) != str(scored_attestation[source_key])
        for bundle_key, source_key in (
            ("id", "model_id"),
            ("revision", "model_revision"),
            ("semantic_run_id", "semantic_run_id"),
        )
    ):
        raise RuntimeError("Decision-binding scored source identity does not match bundle model")
    readout_path = root / "readout_ledger.parquet"
    pairs_path = root / "patch_pair_ledger.parquet"
    for key, path in (("readout", readout_path), ("pairs", pairs_path)):
        if sha256_file(path) != manifest[key]["sha256"]:
            raise RuntimeError(f"Decision-binding {key} checksum mismatch")
    readout = read_table(readout_path)
    pairs = read_table(pairs_path)
    if len(readout) != manifest["readout"]["rows"] or len(pairs) != manifest["pairs"]["rows"]:
        raise RuntimeError("Decision-binding bundle row count mismatch")
    stage = str(manifest.get("stage"))
    expected = {"train": 1801, "validation": 600} if stage == "discovery" else {"test": 599}
    if manifest["source_validation"].get("split_items") != expected:
        raise RuntimeError("Decision-binding bundle has wrong canonical split denominators")
    observed = {
        split: int(readout.loc[readout["split"].astype(str) == split, "item_id"].nunique())
        for split in expected
    }
    if observed != expected:
        raise RuntimeError("Decision-binding readout ledger does not preserve canonical denominators")
    return manifest, readout, pairs


def cmd_prepare(args) -> None:
    scored_path = Path(args.scored)
    applicability_path = Path(args.applicability)
    design_path = Path(args.design)
    scored = read_table(scored_path)
    applicability = read_table(applicability_path)
    source_validation = _validate_prepare_sources(
        scored_path, applicability_path, design_path, scored, applicability, args.stage
    )
    readout = prepare_readout_ledger(scored, applicability, stage=args.stage, seed=args.seed)
    pair_split = "validation" if args.stage == "discovery" else "test"
    pairs = prepare_patch_pair_ledger(
        readout,
        split=pair_split,
        stable_control_count=args.stable_controls,
        label_binding_control_count=args.label_binding_controls,
        seed=args.seed,
    )
    root = Path(args.output_dir)
    readout_path = root / "readout_ledger.parquet"
    pairs_path = root / "patch_pair_ledger.parquet"
    write_table_atomic(readout, readout_path)
    write_table_atomic(pairs, pairs_path)
    _atomic_json(
        {
            "bundle_schema_version": 3,
            "stage": args.stage,
            "seed": args.seed,
            "source_scored_sha256": sha256_file(scored_path),
            "source_applicability_sha256": sha256_file(applicability_path),
            "source_design_sha256": sha256_file(design_path),
            "source_validation": source_validation,
            "model": {
                "id": str(scored["model_id"].iloc[0]) if "model_id" in scored else None,
                "revision": str(scored["model_revision"].iloc[0]) if "model_revision" in scored else None,
                "semantic_run_id": str(scored["semantic_run_id"].iloc[0]) if "semantic_run_id" in scored else None,
            },
            "readout": {
                "path_name": readout_path.name,
                "sha256": sha256_file(readout_path),
                "rows": len(readout),
            },
            "pairs": {
                "path_name": pairs_path.name,
                "sha256": sha256_file(pairs_path),
                "rows": len(pairs),
                "selected_rows": int(pairs["selected_for_patching"].sum()),
                "kind_counts": {
                    str(key): int(value) for key, value in pairs["pair_kind"].value_counts().items()
                },
            },
        },
        root / "bundle_manifest.json",
    )


def _validate_prepare_sources(
    scored_path: Path,
    applicability_path: Path,
    design_path: Path,
    scored: pd.DataFrame,
    applicability: pd.DataFrame,
    stage: str,
) -> dict[str, object]:
    scored_attestation = _validate_scored_manifest(scored_path, scored)
    manifest_path = design_path.with_suffix(design_path.suffix + ".manifest.json")
    if not manifest_path.exists():
        raise ValueError(f"Canonical design manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sha256") != sha256_file(design_path) or manifest.get("rows") != len(
        read_table(design_path)
    ):
        raise ValueError("Canonical design manifest does not match the design parquet")
    applicability_manifest = manifest.get("applicability", {})
    if (
        applicability_manifest.get("path_name") != applicability_path.name
        or applicability_manifest.get("sha256") != sha256_file(applicability_path)
        or int(applicability_manifest.get("rows", -1)) != len(applicability)
    ):
        raise ValueError("Canonical applicability file does not match the design manifest")
    expected = {"train": 1801, "validation": 600} if stage == "discovery" else {"test": 599}
    if (
        set(manifest.get("splits", [])) != set(expected)
        or int(manifest.get("retained_items", -1)) != sum(expected.values())
        or int(manifest.get("source_items", -1)) != sum(expected.values())
    ):
        raise ValueError("Canonical design manifest has the wrong stage denominators")
    design = read_table(design_path)
    counts = validate_canonical_sources(
        scored,
        applicability,
        design,
        stage=stage,
        expected_items=expected,
    )
    return {
        "design_manifest_sha256": sha256_file(manifest_path),
        "split_items": counts,
        "scored_path_name": scored_path.name,
        "scored_artifact": scored_attestation,
    }


def _validate_scored_manifest(scored_path: Path, scored: pd.DataFrame) -> dict[str, object]:
    manifest_path = scored_path.with_name(scored_path.name + ".manifest.json")
    if not manifest_path.exists():
        raise ValueError(f"Causal scored artifact manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = ("semantic_run_id", "model_id", "model_revision")
    if (
        manifest.get("sha256") != sha256_file(scored_path)
        or int(manifest.get("row_count", -1)) != len(scored)
        or any(key not in manifest for key in required)
    ):
        raise ValueError("Causal scored artifact manifest does not match the scored parquet")
    for column in required:
        if column not in scored or not scored[column].astype(str).eq(str(manifest[column])).all():
            raise ValueError("Scored rows contain mixed or unauthenticated model identity")
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "semantic_run_id": str(manifest["semantic_run_id"]),
        "model_id": str(manifest["model_id"]),
        "model_revision": str(manifest["model_revision"]),
        "rows": int(manifest["row_count"]),
        "sha256": str(manifest["sha256"]),
    }


def _chunked(frame: pd.DataFrame, size: int):
    for start in range(0, len(frame), size):
        chunk = frame.iloc[start : start + size].copy()
        digest = hashlib.sha256("\n".join(chunk["readout_work_key"].astype(str)).encode()).hexdigest()
        yield f"readout|{digest[:20]}", chunk


def _stable_sort(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    available = [column for column in columns if column in frame.columns]
    return frame.sort_values(available, kind="mergesort").reset_index(drop=True) if available else frame


_BANK_TARGETS = ("content", "position", "label", "legacy_content")


def _bank_paths(root: Path) -> dict[str, Path]:
    return {target: root / f"probe_bank_{target}.npz" for target in _BANK_TARGETS}


def _bank_hashes(root: Path) -> dict[str, str]:
    return {target: sha256_file(path) for target, path in _bank_paths(root).items()}


def _bank_artifact_hashes(root: Path) -> dict[str, str]:
    return {
        **{
            f"probe_bank_{target}_sha256": digest
            for target, digest in _bank_hashes(root).items()
        },
        "probe_banks_sha256": sha256_file(root / "probe_banks.json"),
    }


def _load_frozen(run_root: Path, profile_name: str):
    identity = json.loads((run_root / "semantic_identity.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    profile = get_model_profile(profile_name)
    if identity["model"]["id"] != profile.model_id or identity["model"]["revision"] != profile.revision:
        raise RuntimeError("Frozen mechanism belongs to a different model identity")
    if (
        manifest.get("status") != "complete"
        or manifest.get("stage") != "discovery"
        or manifest.get("semantic_identity") != identity
    ):
        raise RuntimeError("Frozen mechanism must come from a complete discovery run")
    if int(manifest.get("completed_pairs", -1)) != int(manifest.get("selected_pairs", -2)):
        raise RuntimeError("Frozen discovery run did not complete every selected patch pair")
    limits = identity.get("experiment_config", {}).get("limits", {})
    if any(value is not None for value in limits.values()):
        raise RuntimeError("Frozen mechanism must come from an unrestricted full discovery run")
    selection_path = run_root / "frozen_selection.json"
    bank_paths = _bank_paths(run_root)
    readout_path = run_root / "readout_scores.parquet"
    patch_path = run_root / "patch_results.parquet"
    if not all(path.exists() for path in (selection_path, readout_path, patch_path, *bank_paths.values())):
        raise RuntimeError("Frozen discovery run is missing required artifacts")
    selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
    bank_hashes = _bank_hashes(run_root)
    if selection_payload.get("probe_bank_sha256") != bank_hashes:
        raise RuntimeError("Frozen probe bank checksums mismatch")
    if selection_payload.get("readout_scores_sha256") != sha256_file(readout_path):
        raise RuntimeError("Frozen selection is not bound to the discovery readout scores")
    artifacts = manifest.get("artifacts", {})
    expected_hashes = {
        "frozen_selection_sha256": sha256_file(selection_path),
        "readout_scores_sha256": sha256_file(readout_path),
        "patch_results_sha256": sha256_file(patch_path),
        **_bank_artifact_hashes(run_root),
    }
    if any(artifacts.get(key) != value for key, value in expected_hashes.items()):
        raise RuntimeError("Frozen discovery manifest artifact checksums do not reconcile")
    banks = {target: load_probe_bank(path) for target, path in bank_paths.items()}
    if any(bank.target_name != target for target, bank in banks.items()):
        raise RuntimeError("Frozen probe bank target identity mismatch")
    return banks, selection_payload["selection"], selection_path, bank_paths


def _fit_or_load_banks(root: Path, ledger: pd.DataFrame, model, tokenizer, args, *, stop=None):
    paths = _bank_paths(root)
    meta_path = root / "probe_banks.json"
    if all(path.exists() for path in paths.values()) and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("probe_bank_sha256") != _bank_hashes(root):
            raise RuntimeError("Persisted probe bank checksums mismatch")
        banks = {target: load_probe_bank(path) for target, path in paths.items()}
        if any(bank.target_name != target for target, bank in banks.items()):
            raise RuntimeError("Persisted probe bank target identity mismatch")
        return banks
    train = ledger[ledger["readout_role"].astype(str).eq("probe_train")].copy()
    if train.empty:
        raise RuntimeError("Discovery bundle has no probe-training rows")
    banks: dict[str, object] = {}
    for target, path in paths.items():
        if not path.exists():
            continue
        try:
            bank = load_probe_bank(path)
        except (OSError, ValueError):
            continue
        if bank.target_name == target:
            banks[target] = bank
    missing = [target for target in _BANK_TARGETS if target not in banks]
    fit_started = time.monotonic()
    captured = None
    if missing and not (stop is not None and stop.requested):
        captured = capture_layer_readouts(
            model,
            tokenizer,
            train["prompt"].astype(str).tolist(),
            batch_size=args.batch_size,
            max_batch_tokens=args.max_batch_tokens,
        )
        for target in missing:
            if stop is not None and stop.requested:
                break
            bank = fit_coordinate_probe_bank(captured.activations, train, target)
            path = paths[target]
            temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.npz")
            save_probe_bank(temporary, bank)
            os.replace(temporary, path)
            banks[target] = bank
            if stop is not None and stop.requested:
                break
    if set(banks) != set(_BANK_TARGETS):
        return banks
    eligible_counts = {
        target: int(train.get(f"{target}_evaluable", train["winner_unique"]).astype(bool).sum())
        for target in ("content", "position", "label")
    }
    eligible_counts["legacy_content"] = int(
        (
            train["winner_unique"].astype(bool)
            & train["manipulation"].astype(str).eq("controlled_baseline")
            & train.get("content_evaluable", train["winner_unique"]).astype(bool)
        ).sum()
    )
    _atomic_json(
        {
            "probe_bank_sha256": _bank_hashes(root),
            "training_rows": len(train),
            "eligible_rows": eligible_counts,
            "telemetry": {
                "actual_tokens": 0 if captured is None else captured.actual_tokens,
                "padded_tokens": 0 if captured is None else captured.padded_tokens,
                "padding_ratio": 0.0
                if captured is None else captured.padded_tokens / captured.actual_tokens,
                "input_preparation_seconds": 0.0
                if captured is None else captured.input_preparation_seconds,
                "forward_seconds": 0.0 if captured is None else captured.forward_seconds,
                "wall_seconds": time.monotonic() - fit_started,
            },
        },
        meta_path,
    )
    return banks


def cmd_run_model(args) -> None:
    bundle_root = Path(args.bundle)
    bundle_manifest, ledger, pairs = _load_bundle(bundle_root)
    profile = get_model_profile(args.profile)
    if bundle_manifest["model"]["id"] not in {None, profile.model_id}:
        raise RuntimeError("Bundle model identity does not match --profile")
    if bundle_manifest["model"]["revision"] not in {None, profile.revision}:
        raise RuntimeError("Bundle model revision does not match --profile")
    stage = str(bundle_manifest["stage"])
    canary_items = getattr(args, "canary_items", None)
    for name in ("readout_chunk_size", "patch_shard_size", "batch_size", "max_batch_tokens"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("max_readout_rows", "max_pairs", "canary_items"):
        value = getattr(args, name, None)
        if value is not None and int(value) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if canary_items is not None and (
        getattr(args, "max_readout_rows", None) is not None
        or getattr(args, "max_pairs", None) is not None
    ):
        raise ValueError("--canary-items cannot be combined with row/pair limits")
    frozen_hashes = None
    if stage == "confirmation":
        if not args.frozen_run:
            raise ValueError("Confirmation requires --frozen-run")
        banks, selection, selection_path, bank_paths = _load_frozen(
            Path(args.frozen_run), args.profile
        )
        frozen_hashes = {
            "selection_sha256": sha256_file(selection_path),
            "probe_bank_sha256": {
                target: sha256_file(path) for target, path in bank_paths.items()
            },
        }
    config = {
        "study": "decision_binding_v2",
        "stage": stage,
        "bundle_manifest_sha256": sha256_file(bundle_root / "bundle_manifest.json"),
        "readout_sha256": bundle_manifest["readout"]["sha256"],
        "pairs_sha256": bundle_manifest["pairs"]["sha256"],
        "probe_c": 1e-2,
        "bootstrap_samples": args.bootstrap_samples,
        "permutation_samples": args.permutation_samples,
        "stable_controls": args.stable_controls,
        "label_binding_controls": args.label_binding_controls,
        "seed": args.seed,
        "readout_chunk_size": args.readout_chunk_size,
        "frozen": frozen_hashes,
        "limits": {
            "readout_rows": args.max_readout_rows,
            "pairs": args.max_pairs,
            "canary_items": canary_items,
        },
        "runtime": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "tokenizers", "numpy", "scikit-learn")
        },
    }
    source_paths = default_semantic_source_paths(PROJECT_ROOT)
    lock_path = PROJECT_ROOT / "requirements-gpu.lock"
    if lock_path.exists():
        source_paths.append(lock_path)
    identity = build_semantic_identity(
        profile,
        config=config,
        dataset_path=bundle_root / "readout_ledger.parquet",
        source_paths=source_paths,
    )
    root = Path(args.output_base) / profile.slug / identity.semantic_run_id
    root.mkdir(parents=True, exist_ok=True)
    previous_manifest_path = root / "run_manifest.json"
    previous_manifest = (
        json.loads(previous_manifest_path.read_text(encoding="utf-8"))
        if previous_manifest_path.exists()
        else {}
    )
    previous_telemetry = previous_manifest.get("telemetry", {})
    identity_path = root / "semantic_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity.as_dict():
        raise RuntimeError("Run-root semantic identity mismatch")
    _atomic_json(identity.as_dict(), identity_path)
    if stage == "confirmation":
        shutil.copy2(selection_path, root / "frozen_selection.json")
        for target, bank_path in bank_paths.items():
            shutil.copy2(bank_path, root / f"probe_bank_{target}.npz")
        _atomic_json(
            {
                "probe_bank_sha256": _bank_hashes(root),
            },
            root / "probe_banks.json",
        )
    _atomic_json(
        {
            "status": "starting",
            "stage": stage,
            "profile": args.profile,
            "semantic_identity": identity.as_dict(),
        },
        root / "run_manifest.json",
    )
    started = time.monotonic()
    try:
        model, tokenizer, device = load_model_and_tokenizer(
            profile.model_id,
            revision=profile.revision,
            tokenizer_name=profile.tokenizer_id,
            tokenizer_revision=profile.tokenizer_revision,
            local_files_only=args.local_files_only,
            attn_implementation=profile.attention_backend,
            expected_layers=profile.expected_layers,
            trust_remote_code=profile.trust_remote_code,
        )
    except BaseException as error:
        _atomic_json(
            {
                "status": "failed",
                "stage": stage,
                "profile": args.profile,
                "semantic_identity": identity.as_dict(),
                "failure": {"type": type(error).__name__, "message": str(error)},
            },
            root / "run_manifest.json",
        )
        raise
    stop = StopState()
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in previous:
        signal.signal(sig, stop.request)
    try:
        if not args.allow_cpu and (not torch.cuda.is_available() or device.type != "cuda"):
            raise RuntimeError("Decision-binding model runs require CUDA")
        if stage == "discovery":
            banks = _fit_or_load_banks(root, ledger, model, tokenizer, args, stop=stop)
            if set(banks) != set(_BANK_TARGETS):
                _atomic_json(
                    {
                        "status": "interrupted",
                        "stage": stage,
                        "profile": args.profile,
                        "semantic_identity": identity.as_dict(),
                        "phase": "probe_fit",
                        "completed_probe_banks": sorted(banks),
                        "artifacts": {
                            f"probe_bank_{target}_sha256": sha256_file(
                                root / f"probe_bank_{target}.npz"
                            )
                            for target in banks
                        },
                        "wall_seconds": time.monotonic() - started,
                        "stop_signal": stop.signal_name,
                    },
                    root / "run_manifest.json",
                )
                return

        def progress(phase: str, completed: int, total: int, shard: Path | None = None) -> None:
            elapsed = max(time.monotonic() - started, 1e-9)
            _atomic_json(
                {
                    "phase": phase,
                    "completed": completed,
                    "total": total,
                    "last_shard": None if shard is None else shard.name,
                    "elapsed_seconds": elapsed,
                    "completed_units_per_second": completed / elapsed,
                    "peak_vram_bytes": int(torch.cuda.max_memory_allocated())
                    if torch.cuda.is_available()
                    else 0,
                },
                root / "progress.json",
            )

        evaluation = ledger[
            ledger["readout_role"].isin(["layer_select", "reader_gate", "confirmation"])
        ].copy()
        selected = pairs[pairs["selected_for_patching"].astype(bool)].copy()
        if canary_items is not None:
            item_ids = _select_canary_items(
                selected, int(canary_items), seed=args.seed, tokenizer=tokenizer
            )
            evaluation = evaluation[evaluation["item_id"].astype(str).isin(item_ids)]
            selected = selected[selected["item_id"].astype(str).isin(item_ids)]
        if args.max_readout_rows:
            evaluation = evaluation.iloc[: args.max_readout_rows]
        readout_store = ShardStore(root / "shards" / "readout", identity)
        readout_work = list(_chunked(evaluation, args.readout_chunk_size))
        prior_readout = previous_telemetry.get("readout", {})
        readout_metrics = {
            "batches": int(prior_readout.get("batches", 0)),
            "actual_tokens": int(prior_readout.get("actual_tokens", 0)),
            "padded_tokens": int(prior_readout.get("padded_tokens", 0)),
            "input_preparation_seconds": float(
                prior_readout.get("input_preparation_seconds", 0.0)
            ),
            "forward_seconds": float(prior_readout.get("forward_seconds", 0.0)),
        }

        def process_readout(_key: str, chunk: pd.DataFrame) -> pd.DataFrame:
            captured = capture_layer_readouts(
                model,
                tokenizer,
                chunk["prompt"].astype(str).tolist(),
                batch_size=args.batch_size,
                max_batch_tokens=args.max_batch_tokens,
            )
            readout_metrics["batches"] += captured.batches
            readout_metrics["actual_tokens"] += captured.actual_tokens
            readout_metrics["padded_tokens"] += captured.padded_tokens
            readout_metrics["input_preparation_seconds"] += captured.input_preparation_seconds
            readout_metrics["forward_seconds"] += captured.forward_seconds
            return evaluate_probe_banks(banks, captured.activations, chunk)

        readout_scores = run_sharded_phase(
            readout_store,
            readout_work,
            process_readout,
            max_work_units=1,
            max_seconds=300,
            should_stop=lambda: stop.requested,
            on_flush=lambda completed, shard: progress(
                "readout", len(completed), len(readout_work), shard
            ),
        )
        progress("readout", len(readout_store.completed_work_keys()), len(readout_work))
        readout_scores = _stable_sort(
            readout_scores, ("readout_work_key", "layer", "checkpoint", "_work_key")
        )
        write_table_atomic(readout_scores, root / "readout_scores.parquet")
        if readout_store.completed_work_keys() != {key for key, _ in readout_work}:
            actual_tokens = int(readout_metrics["actual_tokens"])
            padded_tokens = int(readout_metrics["padded_tokens"])
            _atomic_json(
                {
                    "status": "interrupted",
                    "stage": stage,
                    "profile": args.profile,
                    "semantic_identity": identity.as_dict(),
                    "phase": "readout",
                    "expected_readout_chunks": len(readout_work),
                    "completed_readout_chunks": len(readout_store.completed_work_keys()),
                    "readout_rows": len(readout_scores),
                    "telemetry": {
                        "readout": {
                            **readout_metrics,
                            "padding_ratio": padded_tokens / actual_tokens
                            if actual_tokens else 0.0,
                            "tokens_per_forward_second": actual_tokens
                            / max(float(readout_metrics["forward_seconds"]), 1e-9),
                            "write_seconds": float(prior_readout.get("write_seconds", 0.0))
                            + readout_store.write_seconds,
                        }
                    },
                    "artifacts": {
                        "readout_scores_sha256": sha256_file(root / "readout_scores.parquet")
                    },
                    "wall_seconds": time.monotonic() - started,
                    "stop_signal": stop.signal_name,
                },
                root / "run_manifest.json",
            )
            return
        if stage == "discovery":
            selection = select_readout_layers(
                readout_scores,
                bootstrap_samples=args.bootstrap_samples,
                permutation_samples=args.permutation_samples,
                seed=args.seed,
            )
            selection_payload = {
                "selection": selection,
                "probe_bank_sha256": _bank_hashes(root),
                "readout_scores_sha256": sha256_file(root / "readout_scores.parquet"),
            }
            _atomic_json(selection_payload, root / "frozen_selection.json")
        patch_eligible = bool(
            selection.get("content", {}).get("usable")
            and selection.get("label", {}).get("usable")
        )
        if getattr(args, "stop_after_readout", False) or not patch_eligible:
            actual_tokens = int(readout_metrics["actual_tokens"])
            padded_tokens = int(readout_metrics["padded_tokens"])
            _atomic_json(
                {
                    "status": "readout_complete",
                    "stage": stage,
                    "profile": args.profile,
                    "semantic_identity": identity.as_dict(),
                    "patch_eligible": patch_eligible,
                    "readout_rows": len(readout_scores),
                    "expected_readout_rows": len(evaluation)
                    * int(banks["content"].weights.shape[0]) * 2,
                    "readout_chunks": len(readout_work),
                    "expected_readout_chunks": len(readout_work),
                    "completed_readout_chunks": len(readout_store.completed_work_keys()),
                    "selection": selection,
                    "telemetry": {
                        "readout": {
                            **readout_metrics,
                            "padding_ratio": padded_tokens / actual_tokens if actual_tokens else 0.0,
                            "tokens_per_forward_second": actual_tokens
                            / max(float(readout_metrics["forward_seconds"]), 1e-9),
                            "write_seconds": float(prior_readout.get("write_seconds", 0.0))
                            + readout_store.write_seconds,
                        }
                    },
                    "artifacts": {
                        "readout_scores_sha256": sha256_file(root / "readout_scores.parquet"),
                        "frozen_selection_sha256": sha256_file(root / "frozen_selection.json"),
                        **_bank_artifact_hashes(root),
                    },
                    "wall_seconds": time.monotonic() - started,
                    "stop_signal": stop.signal_name,
                },
                root / "run_manifest.json",
            )
            return
        if args.max_pairs:
            selected = selected.iloc[: args.max_pairs]
        patch_store = ShardStore(root / "shards" / "patches", identity)
        patch_work = [(str(row["pair_work_key"]), row) for _, row in selected.iterrows()]
        prior_patches = previous_telemetry.get("patches", {})
        patch_seconds = float(prior_patches.get("wall_seconds", 0.0))

        def process_patch(_key: str, row: pd.Series) -> pd.DataFrame:
            nonlocal patch_seconds
            patch_started = time.monotonic()
            result = run_patch_pair(model, tokenizer, row, banks, selection, device=device)
            patch_seconds += time.monotonic() - patch_started
            return result

        patch_results = run_sharded_phase(
            patch_store,
            patch_work,
            process_patch,
            max_work_units=args.patch_shard_size,
            max_seconds=300,
            should_stop=lambda: stop.requested,
            on_flush=lambda completed, shard: progress(
                "patches", len(completed), len(patch_work), shard
            ),
        )
        progress("patches", len(patch_store.completed_work_keys()), len(patch_work))
        patch_results = _stable_sort(
            patch_results, ("pair_work_key", "mechanism", "layer", "condition", "_work_key")
        )
        write_table_atomic(patch_results, root / "patch_results.parquet")
        complete = patch_store.completed_work_keys() == {key for key, _ in patch_work}
        patch_rows_per_pair = 5 * sum(
            len(selection[mechanism]["patch_layers"]) for mechanism in ("content", "label")
        )
        expected_readout_rows = len(evaluation) * int(banks["content"].weights.shape[0]) * 2
        expected_patch_rows = len(selected) * patch_rows_per_pair
        actual_tokens = int(readout_metrics["actual_tokens"])
        padded_tokens = int(readout_metrics["padded_tokens"])
        readout_telemetry = {
            **readout_metrics,
            "padding_ratio": padded_tokens / actual_tokens if actual_tokens else 0.0,
            "tokens_per_forward_second": actual_tokens
            / max(float(readout_metrics["forward_seconds"]), 1e-9),
            "write_seconds": float(prior_readout.get("write_seconds", 0.0))
            + readout_store.write_seconds,
        }
        patch_telemetry = {
            "wall_seconds": patch_seconds,
            "completed_pairs_per_second": len(patch_store.completed_work_keys())
            / max(patch_seconds, 1e-9),
            "write_seconds": float(prior_patches.get("write_seconds", 0.0))
            + patch_store.write_seconds,
        }
        _atomic_json(
            {
                "status": "complete" if complete else "interrupted",
                "stage": stage,
                "profile": args.profile,
                "semantic_identity": identity.as_dict(),
                "readout_rows": len(readout_scores),
                "readout_chunks": len(readout_work),
                "expected_readout_rows": expected_readout_rows,
                "expected_readout_chunks": len(readout_work),
                "completed_readout_chunks": len(readout_store.completed_work_keys()),
                "selected_pairs": len(selected),
                "completed_pairs": len(patch_store.completed_work_keys()),
                "patch_rows": len(patch_results),
                "patch_rows_per_pair": patch_rows_per_pair,
                "expected_patch_rows": expected_patch_rows,
                "telemetry": {"readout": readout_telemetry, "patches": patch_telemetry},
                "artifacts": {
                    "readout_scores_sha256": sha256_file(root / "readout_scores.parquet"),
                    "patch_results_sha256": sha256_file(root / "patch_results.parquet"),
                    "frozen_selection_sha256": sha256_file(root / "frozen_selection.json")
                    if (root / "frozen_selection.json").exists()
                    else frozen_hashes["selection_sha256"],
                    **_bank_artifact_hashes(root),
                },
                "wall_seconds": time.monotonic() - started,
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
                "stop_signal": stop.signal_name,
            },
            root / "run_manifest.json",
        )
    except BaseException as error:
        _atomic_json(
            {
                "status": "failed",
                "stage": stage,
                "profile": args.profile,
                "semantic_identity": identity.as_dict(),
                "failure": {"type": type(error).__name__, "message": str(error)},
                "wall_seconds": time.monotonic() - started,
                "stop_signal": stop.signal_name,
            },
            root / "run_manifest.json",
        )
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _select_canary_items(
    pairs: pd.DataFrame,
    count: int,
    *,
    seed: int,
    tokenizer=None,
) -> list[str]:
    if count <= 0:
        raise ValueError("Canary item count must be positive")
    candidates = pairs[pairs["selected_for_patching"].astype(bool)].copy()
    required = {"item_id", "subject", "wrapper_name", "donor_prompt", "receiver_prompt"}
    if required - set(candidates.columns):
        raise ValueError("Canary selection requires item, subject, and wrapper columns")
    def prompt_length(value: object) -> int:
        text = str(value)
        return len(text) if tokenizer is None else len(tokenize_text(tokenizer, text))

    candidates["_prompt_length"] = [
        max(prompt_length(donor), prompt_length(receiver))
        for donor, receiver in zip(
            candidates["donor_prompt"], candidates["receiver_prompt"], strict=True
        )
    ]
    items = candidates.groupby("item_id", as_index=False, sort=False).agg(
        subject=("subject", "first"),
        wrapper_name=("wrapper_name", "first"),
        _prompt_length=("_prompt_length", "max"),
    )
    if len(items) < count:
        raise ValueError(f"Canary requires {count} patchable items, found {len(items)}")
    items["_digest"] = items["item_id"].astype(str).map(
        lambda value: hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()
    )
    remaining = items.to_dict("records")
    longest = min(
        remaining,
        key=lambda row: (-int(row["_prompt_length"]), str(row["_digest"])),
    )
    remaining.remove(longest)
    chosen: list[dict[str, object]] = [longest]
    subjects: set[str] = {str(longest["subject"])}
    wrappers: set[str] = {str(longest["wrapper_name"])}
    while remaining and len(chosen) < count:
        best = min(
            remaining,
            key=lambda row: (
                -(str(row["subject"]) not in subjects),
                -(str(row["wrapper_name"]) not in wrappers),
                str(row["_digest"]),
            ),
        )
        remaining.remove(best)
        chosen.append(best)
        subjects.add(str(best["subject"]))
        wrappers.add(str(best["wrapper_name"]))
    return [str(row["item_id"]) for row in chosen]


def _item_clustered_macro_accuracy(
    frame: pd.DataFrame, target: str, prediction: str = "probe_pred_class"
) -> float:
    item_class = frame[["item_id", target, prediction]].copy()
    item_class["correct"] = (
        item_class[target].astype(int) == item_class[prediction].astype(int)
    )
    item_equal = item_class.groupby(["item_id", target], sort=False)["correct"].mean()
    return float(item_equal.groupby(level=1).mean().mean())


def _confirmation_selectivity(
    frame: pd.DataFrame, *, reader: str, target_coordinate: str, nuisance: str
) -> float:
    target_columns = {
        "content": "winner_content_id",
        "position": "winner_position",
        "label": "winner_label_index",
    }
    rows = np.arange(len(frame))

    def values(coordinate: str) -> np.ndarray:
        classes = frame[target_columns[coordinate]].to_numpy(dtype=int)
        scores = frame[
            [f"{reader}_log_prob_{class_id}" for class_id in range(4)]
        ].to_numpy(float)
        return scores[rows, classes]

    differences = pd.DataFrame({
        "item_id": frame["item_id"].astype(str),
        "wrapper_name": frame.get("wrapper_name", "__single_wrapper__"),
        "difference": values(target_coordinate) - values(nuisance),
    })
    per_item_wrapper = differences.groupby(
        ["item_id", "wrapper_name"], sort=False
    )["difference"].mean()
    return float(per_item_wrapper.groupby(level="item_id", sort=False).mean().mean())


def cmd_analyze(args) -> None:
    frames: list[pd.DataFrame] = []
    confirmation_runs: list[tuple[Path, dict[str, object], dict[str, object]]] = []
    for value in args.run:
        root = Path(value)
        identity = json.loads((root / "semantic_identity.json").read_text(encoding="utf-8"))
        manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or manifest.get("semantic_identity") != identity:
            raise RuntimeError(f"Analysis requires a complete identity-bound run: {root}")
        stage = str(manifest.get("stage"))
        if stage not in {"discovery", "confirmation"}:
            raise RuntimeError(f"Unknown decision-binding stage in {root}")
        frame = read_table(root / "patch_results.parquet")
        frame["model"] = str(identity["model"]["slug"])
        frame["semantic_run_id"] = str(identity["semantic_run_id"])
        frame["stage"] = stage
        frames.append(frame)
        if stage == "confirmation":
            confirmation_runs.append((root, identity, manifest))
    patches = pd.concat(frames, ignore_index=True)
    keys = [
        "model", "semantic_run_id", "stage", "split", "item_id", "pair_work_key",
        "mechanism", "layer", "pair_kind",
    ]
    baseline = patches[patches["condition"] == "unpatched"][
        [*keys, "content_target_margin", "symbol_target_margin"]
    ].rename(
        columns={
            "content_target_margin": "baseline_content",
            "symbol_target_margin": "baseline_symbol",
        }
    )
    compared = patches.merge(baseline, on=keys, validate="many_to_one")
    compared["content_change"] = compared["content_target_margin"] - compared["baseline_content"]
    compared["symbol_change"] = compared["symbol_target_margin"] - compared["baseline_symbol"]
    rng = np.random.default_rng(args.seed)
    records = []
    group_keys = [
        "model", "semantic_run_id", "stage", "split", "mechanism", "layer",
        "pair_kind", "condition",
    ]
    for group_key, group in compared.groupby(group_keys, sort=True):
        record = dict(zip(group_keys, group_key, strict=True))
        record["pairs"] = len(group)
        item_effects = group.groupby("item_id", sort=False)[
            ["content_change", "symbol_change"]
        ].mean()
        record["items"] = len(item_effects)
        for name in ("content", "symbol"):
            values = item_effects[f"{name}_change"].to_numpy(float)
            draws = rng.choice(values, size=(args.bootstrap_samples, len(values)), replace=True).mean(axis=1)
            record[f"mean_{name}_margin_change"] = float(values.mean())
            record[f"{name}_ci_low"] = float(np.quantile(draws, 0.025))
            record[f"{name}_ci_high"] = float(np.quantile(draws, 0.975))
        records.append(record)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output / "patch_effects.csv", index=False)
    confirmation_records: list[dict[str, object]] = []
    for root, identity, _manifest in confirmation_runs:
        scores = read_table(root / "readout_scores.parquet")
        frozen = json.loads((root / "frozen_selection.json").read_text(encoding="utf-8"))[
            "selection"
        ]
        total_items = int(scores["item_id"].nunique())
        text_ambiguous_items = int(
            scores.loc[scores["text_identity_ambiguous"].astype(bool), "item_id"].nunique()
        )
        for mechanism, target in (
            ("content", "winner_content_id"),
            ("label", "winner_label_index"),
        ):
            specification = frozen[mechanism]
            subset = scores[
                (scores["layer"].astype(int) == int(specification["selected_layer"]))
                & (scores["checkpoint"].astype(str) == str(specification["checkpoint"]))
                & scores["manipulation"].astype(str).isin(
                    {"position_only", "label_only"}
                    if mechanism == "content" else {"label_only"}
                )
                & scores[f"{mechanism}_evaluable"].astype(bool)
            ].copy()
            if subset.empty:
                raise RuntimeError(f"Confirmation readout is missing frozen {mechanism} rows")
            selected_site = scores[
                (scores["layer"].astype(int) == int(specification["selected_layer"]))
                & (scores["checkpoint"].astype(str) == str(specification["checkpoint"]))
            ]
            ambiguous_items_excluded = int(
                selected_site.loc[
                    selected_site["text_identity_ambiguous"].astype(bool)
                    & ~selected_site[f"{mechanism}_evaluable"].astype(bool),
                    "item_id",
                ].nunique()
            )
            subset["correct"] = (
                subset[f"{mechanism}_pred_class"].astype(int) == subset[target].astype(int)
            )
            macro_accuracy = _item_clustered_macro_accuracy(
                subset, target, f"{mechanism}_pred_class"
            )
            per_item = subset.groupby("item_id", sort=False)["correct"].mean()
            contrasts = (
                (("position_only", "position"), ("label_only", "label"))
                if mechanism == "content" else (("label_only", "content"),)
            )
            arm_metrics = {}
            for manipulation, nuisance in contrasts:
                arm = subset[subset["manipulation"].astype(str) == manipulation]
                if arm.empty:
                    raise RuntimeError(
                        f"Confirmation readout is missing frozen {mechanism} {manipulation} rows"
                    )
                arm_metrics[f"{manipulation}_rows"] = int(len(arm))
                arm_metrics[f"{manipulation}_selectivity"] = _confirmation_selectivity(
                    arm,
                    reader=mechanism,
                    target_coordinate=mechanism,
                    nuisance=nuisance,
                )
            item_values = per_item.to_numpy(float)
            draws = rng.choice(
                item_values,
                size=(args.bootstrap_samples, len(item_values)),
                replace=True,
            ).mean(axis=1)
            confirmation_records.append(
                {
                    "model": identity["model"]["slug"],
                    "semantic_run_id": identity["semantic_run_id"],
                    "stage": "confirmation",
                    "split": "test",
                    "mechanism": mechanism,
                    "layer": int(specification["selected_layer"]),
                    "checkpoint": specification["checkpoint"],
                    "total_items": total_items,
                    "text_ambiguous_items": text_ambiguous_items,
                    "ambiguous_items_excluded": ambiguous_items_excluded,
                    "evaluable_items": int(len(per_item)),
                    "controlled_variant_rows": int(len(subset)),
                    "macro_accuracy": macro_accuracy,
                    "mean_item_accuracy": float(item_values.mean()),
                    "item_accuracy_ci_low": float(np.quantile(draws, 0.025)),
                    "item_accuracy_ci_high": float(np.quantile(draws, 0.975)),
                    "coordinate_selectivity": min(
                        value for key, value in arm_metrics.items()
                        if key.endswith("_selectivity")
                    ),
                    **arm_metrics,
                }
            )
    pd.DataFrame(confirmation_records).to_csv(output / "readout_confirmation.csv", index=False)
    _atomic_json(
        {
            "runs": [str(value) for value in args.run],
            "patch_rows": len(patches),
            "confirmation_readout_rows": len(confirmation_records),
        },
        output / "analysis_manifest.json",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="interface-formatting-decision-binding")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--scored", required=True)
    prepare.add_argument("--applicability", required=True)
    prepare.add_argument("--design", required=True)
    prepare.add_argument("--stage", choices=("discovery", "confirmation"), required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--seed", default="decision-binding-v1")
    prepare.add_argument("--stable-controls", type=int, default=96)
    prepare.add_argument("--label-binding-controls", type=int, default=96)
    prepare.set_defaults(func=cmd_prepare)
    run = sub.add_parser("run-model")
    run.add_argument("--profile", choices=sorted(MODEL_PROFILES), required=True)
    run.add_argument("--bundle", required=True)
    run.add_argument("--frozen-run")
    run.add_argument("--output-base", default="results/decision_binding_runs")
    run.add_argument("--batch-size", type=int, default=16)
    run.add_argument("--max-batch-tokens", type=int, default=24000)
    run.add_argument("--readout-chunk-size", type=int, default=128)
    run.add_argument("--patch-shard-size", type=int, default=16)
    run.add_argument("--bootstrap-samples", type=int, default=5000)
    run.add_argument("--permutation-samples", type=int, default=1000)
    run.add_argument("--stable-controls", type=int, default=96)
    run.add_argument("--label-binding-controls", type=int, default=96)
    run.add_argument("--max-readout-rows", type=int)
    run.add_argument("--max-pairs", type=int)
    run.add_argument("--canary-items", type=int)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--local-files-only", action="store_true")
    run.add_argument("--allow-cpu", action="store_true", help=argparse.SUPPRESS)
    run.add_argument("--stop-after-readout", action="store_true")
    run.set_defaults(func=cmd_run_model)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--run", action="append", required=True)
    analyze.add_argument("--output-dir", default="results/decision_binding_analysis")
    analyze.add_argument("--bootstrap-samples", type=int, default=5000)
    analyze.add_argument("--seed", type=int, default=0)
    analyze.set_defaults(func=cmd_analyze)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
