from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import time
import uuid
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .causal_option_audit import load_causal_option_annotations
from .decision_binding_cli import _load_bundle
from .decision_binding_content import (
    CandidateRanker,
    candidate_score_rows,
    capture_candidate_states,
    evaluate_candidate_ranker,
    fit_candidate_ranker,
    gate_candidate_reader,
    load_candidate_ranker,
    locate_content_token_indices,
    metadata_candidate_features,
    prepare_candidate_sites,
    save_candidate_ranker,
    select_candidate_reader,
)
from .model_loader import load_model_and_tokenizer
from .model_profiles import MODEL_PROFILES, get_model_profile
from .model_runner import StopState
from .run_identity import build_semantic_identity, default_semantic_source_paths, sha256_file
from .shards import ShardStore, run_sharded_phase
from .utils import read_table, write_table_atomic


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROLE_ITEM_COUNTS = {"probe_train": 1801, "layer_select": 300, "reader_gate": 300}
L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1)


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _role_items(frame: pd.DataFrame) -> dict[str, int]:
    return {
        role: int(frame.loc[frame["readout_role"].astype(str) == role, "item_id"].nunique())
        for role in ROLE_ITEM_COUNTS
    }


def _profile_for_bundle(manifest: Mapping[str, object]):
    model = manifest.get("model", {})
    for profile in MODEL_PROFILES.values():
        if (
            isinstance(model, Mapping)
            and str(model.get("id")) == profile.model_id
            and str(model.get("revision")) == profile.revision
        ):
            return profile
    raise ValueError("decision-binding source model does not match a pinned profile")


def cmd_prepare(args) -> None:
    source_root = Path(args.bundle)
    source_manifest, ledger, _ = _load_bundle(source_root)
    if source_manifest.get("stage") != "discovery":
        raise ValueError("candidate-local preparation accepts discovery data only")
    applicability_path = Path(args.applicability)
    if sha256_file(applicability_path) != source_manifest.get("source_applicability_sha256"):
        raise ValueError("applicability checksum does not match the source bundle")
    applicability = read_table(applicability_path)
    option_maps = load_causal_option_annotations(args.option_audit)
    sites = prepare_candidate_sites(ledger, applicability, option_maps=option_maps)
    if len(sites) != len(ledger) or set(sites["work_key"]) != set(ledger["work_key"]):
        raise RuntimeError("candidate preparation did not preserve every source work key")
    observed_roles = _role_items(sites)
    if observed_roles != ROLE_ITEM_COUNTS:
        raise RuntimeError(f"candidate preparation changed frozen roles: {observed_roles}")
    profile = _profile_for_bundle(source_manifest)
    output = Path(args.output_dir)
    path = output / "candidate_sites.parquet"
    write_table_atomic(sites, path)
    _atomic_json({
        "schema_version": 1,
        "stage": "discovery",
        "rows": int(len(sites)),
        "sha256": sha256_file(path),
        "work_keys_sha256": hashlib.sha256(
            "\n".join(sorted(sites["work_key"].astype(str))).encode()
        ).hexdigest(),
        "role_items": observed_roles,
        "target_evaluable_rows": int(sites["content_target_evaluable"].sum()),
        "mapping_mismatch_rows": int((~sites["mapping_matches_ledger"].astype(bool)).sum()),
        "source_bundle_manifest_sha256": sha256_file(source_root / "bundle_manifest.json"),
        "source_readout_sha256": source_manifest["readout"]["sha256"],
        "source_applicability_sha256": sha256_file(applicability_path),
        "option_audit_manifest_sha256": sha256_file(Path(args.option_audit) / "manifest.json"),
        "source_model": {
            "id": profile.model_id,
            "revision": profile.revision,
            "slug": profile.slug,
        },
    }, output / "prepared_manifest.json")


def _load_prepared_bundle(root: str | Path) -> tuple[dict[str, object], pd.DataFrame]:
    bundle = Path(root)
    manifest_path = bundle / "prepared_manifest.json"
    sites_path = bundle / "candidate_sites.parquet"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("stage") != "discovery":
        raise RuntimeError("candidate prepared manifest has an unsupported identity")
    if sha256_file(sites_path) != manifest.get("sha256"):
        raise RuntimeError("candidate prepared bundle checksum mismatch")
    sites = read_table(sites_path)
    if int(manifest.get("rows", -1)) != len(sites) or sites["work_key"].duplicated().any():
        raise RuntimeError("candidate prepared bundle row identity mismatch")
    if _role_items(sites) != {
        str(key): int(value) for key, value in manifest.get("role_items", {}).items()
    }:
        raise RuntimeError("candidate prepared bundle role counts mismatch")
    return manifest, sites


def _select_canary_sites(frame: pd.DataFrame, *, items_per_role: int, seed: int) -> pd.DataFrame:
    if items_per_role <= 0:
        raise ValueError("canary items per role must be positive")
    chosen: list[pd.DataFrame] = []
    for role in ROLE_ITEM_COUNTS:
        group = frame[frame["readout_role"].astype(str) == role]
        item_ids = sorted(
            group["item_id"].astype(str).unique(),
            key=lambda item: hashlib.sha256(f"{seed}|{role}|{item}".encode()).hexdigest(),
        )[:items_per_role]
        if len(item_ids) != items_per_role:
            raise ValueError(f"canary lacks {items_per_role} items for {role}")
        chosen.append(group[group["item_id"].astype(str).isin(item_ids)])
    return pd.concat(chosen, ignore_index=True).sort_values("work_key", kind="mergesort")


def _chunks(frame: pd.DataFrame, size: int) -> list[tuple[str, pd.DataFrame]]:
    work: list[tuple[str, pd.DataFrame]] = []
    ordered = frame.sort_values("work_key", kind="mergesort").reset_index(drop=True)
    for start in range(0, len(ordered), size):
        chunk = ordered.iloc[start : start + size].copy()
        digest = hashlib.sha256("\n".join(chunk["work_key"].astype(str)).encode()).hexdigest()
        work.append((digest[:20], chunk))
    return work


def _verify_raw_winners(chunk: pd.DataFrame, raw_log_probs: torch.Tensor) -> None:
    predictions = np.asarray(raw_log_probs.float()).argmax(axis=1)
    expected = chunk["raw_predicted_label"].astype(str).map(
        {"A": 0, "B": 1, "C": 2, "D": 3}
    ).to_numpy()
    unique = chunk["winner_unique"].astype(bool).to_numpy()
    if np.isnan(expected).any() or not np.array_equal(predictions[unique], expected[unique]):
        mismatches = int(np.sum(predictions[unique] != expected[unique]))
        raise RuntimeError(f"captured forward pass changes {mismatches} stored raw winners")


def _token_positions(tokenizer, chunk: pd.DataFrame) -> list[list[int]]:
    return [
        locate_content_token_indices(tokenizer, str(row.prompt), row.content_char_spans)
        for row in chunk.itertuples()
    ]


def _save_activation_shard(
    path: Path,
    *,
    activations: torch.Tensor,
    work_keys: Sequence[str],
    semantic_sha256: str,
) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": 1,
        "semantic_sha256": semantic_sha256,
        "work_keys": list(map(str, work_keys)),
        "activations": activations.cpu(),
    }, temporary)
    os.replace(temporary, path)


def _load_activation_shard(
    path: Path,
    *,
    work_keys: Sequence[str],
    semantic_sha256: str,
) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema_version") != 1
        or payload.get("semantic_sha256") != semantic_sha256
        or payload.get("work_keys") != list(map(str, work_keys))
    ):
        raise RuntimeError(f"training activation shard identity mismatch: {path}")
    values = payload.get("activations")
    if not isinstance(values, torch.Tensor) or values.ndim != 4 or values.shape[2] != 4:
        raise RuntimeError(f"training activation shard is malformed: {path}")
    return values


def _capture_training(
    root: Path,
    training: pd.DataFrame,
    model,
    tokenizer,
    *,
    identity,
    batch_size: int,
    max_batch_tokens: int,
    chunk_size: int,
    stop: StopState,
    progress,
) -> torch.Tensor | None:
    values: list[torch.Tensor] = []
    work = _chunks(training, chunk_size)
    for index, (key, chunk) in enumerate(work):
        path = root / "activation_shards" / "training" / f"{key}.pt"
        if path.exists():
            captured = _load_activation_shard(
                path, work_keys=chunk["work_key"], semantic_sha256=identity.semantic_sha256
            )
        else:
            if stop.requested:
                return None
            result = capture_candidate_states(
                model,
                tokenizer,
                chunk["prompt"].astype(str).tolist(),
                _token_positions(tokenizer, chunk),
                batch_size=batch_size,
                max_batch_tokens=max_batch_tokens,
            )
            _verify_raw_winners(chunk, result.raw_log_probs)
            captured = result.activations
            _save_activation_shard(
                path,
                activations=captured,
                work_keys=chunk["work_key"],
                semantic_sha256=identity.semantic_sha256,
            )
        values.append(captured)
        progress("training_capture", index + 1, len(work))
    return torch.cat(values, dim=0)


def _fit_rankers(
    root: Path,
    activations: torch.Tensor,
    targets: np.ndarray,
    *,
    l2_grid: Sequence[float],
    semantic_sha256: str,
    prefix: str,
    max_iter: int,
    fit_device=None,
) -> dict[float, CandidateRanker]:
    rankers: dict[float, CandidateRanker] = {}
    for l2 in l2_grid:
        path = root / "rankers" / f"{prefix}-l2-{float(l2):.0e}.npz"
        if path.exists():
            ranker = load_candidate_ranker(path, semantic_sha256=semantic_sha256)
        else:
            ranker = fit_candidate_ranker(
                activations,
                targets,
                l2=float(l2),
                max_iter=max_iter,
                device=fit_device,
            )
            save_candidate_ranker(path, ranker, semantic_sha256=semantic_sha256)
        rankers[float(l2)] = ranker
    return rankers


def _score_model_phase(
    root: Path,
    phase: str,
    sites: pd.DataFrame,
    rankers: Mapping[float, CandidateRanker],
    model,
    tokenizer,
    *,
    identity,
    batch_size: int,
    max_batch_tokens: int,
    chunk_size: int,
    stop: StopState,
    progress,
    selected_layer: int | None = None,
    extra_rankers: Mapping[str, CandidateRanker] | None = None,
) -> tuple[pd.DataFrame, bool]:
    store = ShardStore(root / "shards" / phase, identity)
    work = _chunks(sites, chunk_size)

    def process(_key: str, chunk: pd.DataFrame) -> pd.DataFrame:
        captured = capture_candidate_states(
            model,
            tokenizer,
            chunk["prompt"].astype(str).tolist(),
            _token_positions(tokenizer, chunk),
            batch_size=batch_size,
            max_batch_tokens=max_batch_tokens,
        )
        _verify_raw_winners(chunk, captured.raw_log_probs)
        rows = []
        for l2, ranker in rankers.items():
            scored = candidate_score_rows(
                chunk,
                evaluate_candidate_ranker(ranker, captured.activations),
                l2=l2,
                reader_name="content",
            )
            if selected_layer is not None:
                scored = scored[scored["layer"].astype(int) == selected_layer]
            rows.append(scored)
        for name, ranker in (extra_rankers or {}).items():
            scored = candidate_score_rows(
                chunk,
                evaluate_candidate_ranker(ranker, captured.activations),
                l2=ranker.l2,
                reader_name=name,
            )
            if selected_layer is not None:
                scored = scored[scored["layer"].astype(int) == selected_layer]
            rows.append(scored)
        return pd.concat(rows, ignore_index=True)

    merged = run_sharded_phase(
        store,
        work,
        process,
        max_work_units=1,
        max_seconds=300,
        should_stop=lambda: stop.requested,
        on_flush=lambda completed, _path: progress(phase, len(completed), len(work)),
    )
    complete = store.completed_work_keys() == {key for key, _ in work}
    return merged, complete


def _score_metadata_rankers(
    sites: pd.DataFrame,
    rankers: Mapping[float, CandidateRanker],
    feature_name: str,
) -> pd.DataFrame:
    features = metadata_candidate_features(sites)[feature_name]
    return pd.concat([
        candidate_score_rows(
            sites,
            evaluate_candidate_ranker(ranker, features),
            l2=l2,
            reader_name=feature_name,
        )
        for l2, ranker in rankers.items()
    ], ignore_index=True)


def _best_control(frame: pd.DataFrame) -> pd.DataFrame:
    candidates = []
    for l2, group in frame.groupby("l2", sort=True):
        probabilities = group[[f"content_prob_{value}" for value in range(4)]].to_numpy()
        target = group["actual_winner_content_id"].to_numpy(dtype=int)
        correct = probabilities.argmax(axis=1) == target
        scored = group.assign(_correct=correct)
        arms = []
        for manipulation in ("position_only", "label_only"):
            arm = scored[scored["manipulation"].astype(str) == manipulation]
            arms.append(float(arm.groupby("item_id")["_correct"].mean().mean()))
        candidates.append((min(arms), float(l2), group))
    return max(candidates, key=lambda value: (value[0], value[1]))[2].copy()


def _manifest_artifacts(root: Path, names: Sequence[str]) -> dict[str, dict[str, object]]:
    return {
        name: {"sha256": sha256_file(root / name), "bytes": (root / name).stat().st_size}
        for name in names
    }


def verify_run_root(
    root: str | Path,
    *,
    expected_run_id: str,
    mode: str,
) -> dict[str, object]:
    run_root = Path(root)
    identity = json.loads((run_root / "semantic_identity.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    expected_status = "canary_complete" if mode == "canary" else "content_readout_complete"
    if mode not in {"canary", "complete"} or manifest.get("status") != expected_status:
        raise RuntimeError("candidate run has not reached the requested terminal state")
    if (
        identity.get("semantic_run_id") != expected_run_id
        or manifest.get("semantic_identity") != identity
    ):
        raise RuntimeError("candidate run semantic identity mismatch")
    if bool(manifest.get("canary")) != (mode == "canary"):
        raise RuntimeError("candidate run mode mismatch")
    for name, metadata in manifest.get("artifacts", {}).items():
        path = run_root / str(name)
        if not path.is_file() or sha256_file(path) != metadata.get("sha256"):
            raise RuntimeError(f"candidate artifact checksum mismatch: {name}")
        if path.stat().st_size != int(metadata.get("bytes", -1)):
            raise RuntimeError(f"candidate artifact size mismatch: {name}")
    ranker_manifest_path = run_root / "ranker_manifest.json"
    if ranker_manifest_path.exists():
        ranker_manifest = json.loads(ranker_manifest_path.read_text(encoding="utf-8"))
        if ranker_manifest.get("semantic_sha256") != identity.get("semantic_sha256"):
            raise RuntimeError("candidate ranker manifest identity mismatch")
        for name, digest in ranker_manifest.get("rankers", {}).items():
            path = run_root / "rankers" / str(name)
            if not path.is_file() or sha256_file(path) != digest:
                raise RuntimeError(f"candidate ranker checksum mismatch: {name}")
    gate = json.loads((run_root / "gate_report.json").read_text(encoding="utf-8"))
    if (
        gate.get("claim") != "candidate_local_linear_decodability"
        or bool(gate.get("patch_eligible"))
        or bool(manifest.get("patch_eligible"))
        or bool(gate.get("opens_final_confirmation")) != bool(gate.get("tier_2_pass"))
        or bool(manifest.get("final_confirmation_opened"))
        or bool(manifest.get("tier_1_pass")) != bool(gate.get("tier_1_pass"))
        or bool(manifest.get("tier_2_pass")) != bool(gate.get("tier_2_pass"))
    ):
        raise RuntimeError("candidate run violates the frozen scientific claim boundary")
    if mode == "complete" and {
        str(key): int(value) for key, value in manifest.get("role_items", {}).items()
    } != ROLE_ITEM_COUNTS:
        raise RuntimeError("candidate run changed the frozen discovery split")
    return manifest


def cmd_verify(args) -> None:
    manifest = verify_run_root(
        args.run_root, expected_run_id=args.run_id, mode=args.mode
    )
    print(json.dumps({
        "status": manifest["status"],
        "tier_1_pass": manifest["tier_1_pass"],
        "tier_2_pass": manifest["tier_2_pass"],
    }, sort_keys=True))


def cmd_run_model(args) -> None:
    prepared_manifest, all_sites = _load_prepared_bundle(args.bundle)
    profile = get_model_profile(args.profile)
    source_model = prepared_manifest["source_model"]
    if (
        source_model.get("id") != profile.model_id
        or source_model.get("revision") != profile.revision
        or source_model.get("slug") != profile.slug
    ):
        raise ValueError("CLI-selected model does not match the prepared source model")
    canary = args.canary_items is not None
    sites = (
        _select_canary_sites(all_sites, items_per_role=args.canary_items, seed=args.seed)
        if canary else all_sites
    )
    config = {
        "experiment": "decision_binding_candidate_local_content_v3",
        "prepared_manifest_sha256": sha256_file(Path(args.bundle) / "prepared_manifest.json"),
        "token_position": "final_full_token_within_audited_candidate_payload",
        "l2_grid": list(map(float, args.l2_grid)),
        "seed": int(args.seed),
        "bootstrap_samples": int(args.bootstrap_samples),
        "permutation_samples": int(args.permutation_samples),
        "canary_items": args.canary_items,
    }
    identity = build_semantic_identity(
        profile,
        config=config,
        dataset_path=Path(args.bundle) / "candidate_sites.parquet",
        source_paths=default_semantic_source_paths(PROJECT_ROOT),
    )
    root = Path(args.output_base) / profile.slug / identity.semantic_run_id
    root.mkdir(parents=True, exist_ok=True)
    identity_path = root / "semantic_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity.as_dict():
        raise RuntimeError("candidate run-root semantic identity mismatch")
    _atomic_json(identity.as_dict(), identity_path)
    started = time.monotonic()
    previous_manifest = root / "run_manifest.json"
    _atomic_json({
        "status": "starting",
        "profile": args.profile,
        "semantic_identity": identity.as_dict(),
        "canary": canary,
    }, previous_manifest)

    stop = StopState()
    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in previous_handlers:
        signal.signal(sig, stop.request)

    def progress(phase: str, completed: int, total: int) -> None:
        elapsed = max(time.monotonic() - started, 1e-9)
        _atomic_json({
            "phase": phase,
            "completed": int(completed),
            "total": int(total),
            "elapsed_seconds": elapsed,
            "units_per_second": completed / elapsed,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        }, root / "progress.json")

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
        if not args.allow_cpu and device.type != "cuda":
            raise RuntimeError("candidate-local model runs require CUDA")
        training_pool = sites[
            sites["readout_role"].astype(str).eq("probe_train")
            & sites["manipulation"].astype(str).eq("controlled_baseline")
            & sites["wrapper_name"].astype(str).eq("plain")
        ].sort_values("work_key", kind="mergesort")
        if training_pool["item_id"].duplicated().any() or (
            not canary and len(training_pool) != 1801
        ):
            raise RuntimeError("candidate training rows do not match the frozen plain baseline")
        training = training_pool[training_pool["content_target_evaluable"].astype(bool)].copy()
        if training.empty:
            raise RuntimeError("candidate training pool has no unique content targets")
        activations = _capture_training(
            root, training, model, tokenizer,
            identity=identity,
            batch_size=args.batch_size,
            max_batch_tokens=args.max_batch_tokens,
            chunk_size=args.capture_chunk_size,
            stop=stop,
            progress=progress,
        )
        if activations is None:
            _atomic_json({
                "status": "interrupted", "phase": "training_capture",
                "semantic_identity": identity.as_dict(), "stop_signal": stop.signal_name,
            }, previous_manifest)
            return
        targets = training["winner_position"].to_numpy(dtype=int)
        rankers = _fit_rankers(
            root, activations, targets,
            l2_grid=args.l2_grid,
            semantic_sha256=identity.semantic_sha256,
            prefix="content",
            max_iter=args.max_iter,
            fit_device=device,
        )
        train_features = metadata_candidate_features(training)
        control_rankers = {
            name: _fit_rankers(
                root,
                torch.from_numpy(features),
                targets,
                l2_grid=args.l2_grid,
                semantic_sha256=identity.semantic_sha256,
                prefix=f"control-{name}",
                max_iter=args.max_iter,
                fit_device=device,
            )
            for name, features in train_features.items()
        }
        layer_sites = sites[
            sites["readout_role"].astype(str).eq("layer_select")
            & sites["content_target_evaluable"].astype(bool)
        ].copy()
        layer_scores, complete = _score_model_phase(
            root, "layer_select", layer_sites, rankers, model, tokenizer,
            identity=identity,
            batch_size=args.batch_size,
            max_batch_tokens=args.max_batch_tokens,
            chunk_size=args.capture_chunk_size,
            stop=stop,
            progress=progress,
        )
        if not complete:
            _atomic_json({
                "status": "interrupted", "phase": "layer_select",
                "semantic_identity": identity.as_dict(), "stop_signal": stop.signal_name,
            }, previous_manifest)
            return
        layer_controls = {
            name: _best_control(_score_metadata_rankers(layer_sites, fitted, name))
            for name, fitted in control_rankers.items()
        }
        selection_stop = False
        try:
            selection = select_candidate_reader(
                layer_scores,
                layer_controls,
                allow_ineligible=canary,
            )
        except ValueError as error:
            if "beats both isolated nuisance controls" not in str(error):
                raise
            selection = select_candidate_reader(
                layer_scores, layer_controls, allow_ineligible=True
            )
            selection["stop_reason"] = "no_reader_beat_both_isolated_nuisance_controls"
            selection_stop = True
        selection_payload = {
            **selection,
            "semantic_sha256": identity.semantic_sha256,
            "prepared_manifest_sha256": config["prepared_manifest_sha256"],
            "token_position": config["token_position"],
            "l2_grid": config["l2_grid"],
            "seed": args.seed,
            "final_confirmation_opened": False,
            "ranker_sha256": {
                path.name: sha256_file(path) for path in sorted((root / "rankers").glob("*.npz"))
            },
        }
        _atomic_json(selection_payload, root / "frozen_selection.json")

        if selection_stop:
            report = {
                "schema_version": 1,
                "claim": "candidate_local_linear_decodability",
                "selection_eligible": False,
                "stop_reason": selection["stop_reason"],
                "tier_1_pass": False,
                "tier_2_pass": False,
                "content_reader_usable": False,
                "patch_eligible": False,
                "opens_final_confirmation": False,
                "reader_gate_opened": False,
            }
            write_table_atomic(layer_scores, root / "layer_select_scores.parquet")
            write_table_atomic(
                pd.concat(layer_controls.values(), ignore_index=True),
                root / "layer_select_control_scores.parquet",
            )
            _atomic_json(report, root / "gate_report.json")
            _atomic_json({
                "schema_version": 1,
                "semantic_sha256": identity.semantic_sha256,
                "rankers": {
                    path.name: sha256_file(path)
                    for path in sorted((root / "rankers").glob("*.npz"))
                },
            }, root / "ranker_manifest.json")
            artifact_names = (
                "semantic_identity.json", "frozen_selection.json", "gate_report.json",
                "ranker_manifest.json", "layer_select_scores.parquet",
                "layer_select_control_scores.parquet",
            )
            _atomic_json({
                "status": "content_readout_complete",
                "profile": args.profile,
                "semantic_identity": identity.as_dict(),
                "canary": False,
                "role_items": _role_items(sites),
                "training_pool_items": int(training_pool["item_id"].nunique()),
                "training_evaluable_items": int(training["item_id"].nunique()),
                "tier_1_pass": False,
                "tier_2_pass": False,
                "content_reader_usable": False,
                "patch_eligible": False,
                "final_confirmation_opened": False,
                "reader_gate_opened": False,
                "stop_reason": selection["stop_reason"],
                "artifacts": _manifest_artifacts(root, artifact_names),
                "wall_seconds": time.monotonic() - started,
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available() else 0,
            }, previous_manifest)
            return

        selected_l2 = float(selection["selected_l2"])
        selected_layer = int(selection["selected_layer"])
        gate_sites = sites[
            sites["readout_role"].astype(str).eq("reader_gate")
            & sites["content_target_evaluable"].astype(bool)
        ].copy()
        random_rankers: dict[str, CandidateRanker] = {}
        for random_index in range(3):
            rng = np.random.default_rng(args.seed + 1000 + random_index)
            random_targets = (targets + rng.integers(0, 4, size=len(targets))) % 4
            random_rankers[f"random_{random_index}"] = _fit_rankers(
                root, activations, random_targets,
                l2_grid=[selected_l2],
                semantic_sha256=identity.semantic_sha256,
                prefix=f"random-{random_index}",
                max_iter=args.max_iter,
                fit_device=device,
            )[selected_l2]
        all_gate_scores, complete = _score_model_phase(
            root, "reader_gate", gate_sites, {selected_l2: rankers[selected_l2]}, model, tokenizer,
            identity=identity,
            batch_size=args.batch_size,
            max_batch_tokens=args.max_batch_tokens,
            chunk_size=args.capture_chunk_size,
            stop=stop,
            progress=progress,
            selected_layer=selected_layer,
            extra_rankers=random_rankers,
        )
        if not complete:
            _atomic_json({
                "status": "interrupted", "phase": "reader_gate",
                "semantic_identity": identity.as_dict(), "stop_signal": stop.signal_name,
            }, previous_manifest)
            return
        gate_scores = all_gate_scores[all_gate_scores["reader_name"].astype(str) == "content"].copy()
        random_frames = [
            all_gate_scores[all_gate_scores["reader_name"].astype(str) == name].copy()
            for name in sorted(random_rankers)
        ]
        gate_controls = {
            name: _score_metadata_rankers(
                gate_sites, {float(control.iloc[0]["l2"]): fitted[float(control.iloc[0]["l2"])]}, name
            )
            for name, (control, fitted) in {
                key: (layer_controls[key], control_rankers[key]) for key in layer_controls
            }.items()
        }
        bootstrap = min(args.bootstrap_samples, 300) if canary else args.bootstrap_samples
        permutations = min(args.permutation_samples, 100) if canary else args.permutation_samples
        report = gate_candidate_reader(
            gate_scores,
            selection,
            gate_controls,
            random_frames,
            bootstrap_samples=bootstrap,
            permutation_samples=permutations,
            seed=args.seed,
        )
        write_table_atomic(layer_scores, root / "layer_select_scores.parquet")
        write_table_atomic(gate_scores, root / "reader_gate_scores.parquet")
        write_table_atomic(
            pd.concat(layer_controls.values(), ignore_index=True),
            root / "layer_select_control_scores.parquet",
        )
        write_table_atomic(
            pd.concat(gate_controls.values(), ignore_index=True),
            root / "reader_gate_control_scores.parquet",
        )
        write_table_atomic(
            pd.concat(random_frames, ignore_index=True),
            root / "reader_gate_random_scores.parquet",
        )
        _atomic_json(report, root / "gate_report.json")
        _atomic_json({
            "schema_version": 1,
            "semantic_sha256": identity.semantic_sha256,
            "rankers": {
                path.name: sha256_file(path) for path in sorted((root / "rankers").glob("*.npz"))
            },
        }, root / "ranker_manifest.json")
        artifact_names = (
            "semantic_identity.json", "frozen_selection.json", "gate_report.json",
            "ranker_manifest.json", "layer_select_scores.parquet", "reader_gate_scores.parquet",
            "layer_select_control_scores.parquet", "reader_gate_control_scores.parquet",
            "reader_gate_random_scores.parquet",
        )
        _atomic_json({
            "status": "canary_complete" if canary else "content_readout_complete",
            "profile": args.profile,
            "semantic_identity": identity.as_dict(),
            "canary": canary,
            "role_items": _role_items(sites),
            "training_pool_items": int(training_pool["item_id"].nunique()),
            "training_evaluable_items": int(training["item_id"].nunique()),
            "target_evaluable_rows_by_role": {
                role: int(sites.loc[
                    sites["readout_role"].astype(str).eq(role)
                    & sites["content_target_evaluable"].astype(bool)
                ].shape[0])
                for role in ROLE_ITEM_COUNTS
            },
            "tier_1_pass": bool(report["tier_1_pass"]),
            "tier_2_pass": bool(report["tier_2_pass"]),
            "content_reader_usable": bool(report["content_reader_usable"]),
            "patch_eligible": False,
            "final_confirmation_opened": False,
            "artifacts": _manifest_artifacts(root, artifact_names),
            "wall_seconds": time.monotonic() - started,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        }, previous_manifest)
    except BaseException as error:
        _atomic_json({
            "status": "failed",
            "profile": args.profile,
            "semantic_identity": identity.as_dict(),
            "failure": {"type": type(error).__name__, "message": str(error)},
            "wall_seconds": time.monotonic() - started,
            "stop_signal": stop.signal_name,
        }, previous_manifest)
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="interface-formatting-decision-content")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--bundle", required=True)
    prepare.add_argument("--applicability", required=True)
    prepare.add_argument("--option-audit", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.set_defaults(func=cmd_prepare)
    run = sub.add_parser("run-model")
    run.add_argument("--profile", choices=sorted(MODEL_PROFILES), required=True)
    run.add_argument("--bundle", required=True)
    run.add_argument("--output-base", default="results/decision_binding_content_runs")
    run.add_argument("--batch-size", type=int, default=16)
    run.add_argument("--max-batch-tokens", type=int, default=24000)
    run.add_argument("--capture-chunk-size", type=int, default=64)
    run.add_argument("--max-iter", type=int, default=100)
    run.add_argument("--l2-grid", type=float, nargs="+", default=list(L2_GRID))
    run.add_argument("--bootstrap-samples", type=int, default=5000)
    run.add_argument("--permutation-samples", type=int, default=1000)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--canary-items", type=int)
    run.add_argument("--local-files-only", action="store_true")
    run.add_argument("--allow-cpu", action="store_true", help=argparse.SUPPRESS)
    run.set_defaults(func=cmd_run_model)
    verify = sub.add_parser("verify")
    verify.add_argument("--run-root", required=True)
    verify.add_argument("--run-id", required=True)
    verify.add_argument("--mode", choices=("canary", "complete"), required=True)
    verify.set_defaults(func=cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if getattr(args, "capture_chunk_size", 1) <= 0:
        raise ValueError("capture chunk size must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
