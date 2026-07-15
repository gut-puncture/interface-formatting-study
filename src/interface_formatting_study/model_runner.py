from __future__ import annotations

import gc
import json
import os
import platform
import signal
import socket
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd
import torch

from .anchors import resolve_anchor_positions
from .behavioral_eval import evaluate_behavioral
from .component_patching import run_focused_patching_controls, summarize_focused_controls
from .conflicts import construct_conflict_pairs, item_outcome_summary
from .data_loading import dataset_audit_from_frame
from .diagnostics import (
    run_attention_diagnostics,
    run_vanilla_convergence,
    summarize_attention_diagnostics,
    summarize_vanilla_convergence,
)
from .experiment import prepare_dataset, save_split_manifest
from .hooks import find_transformer_blocks
from .model_loader import load_model_and_tokenizer
from .model_profiles import ModelProfile, get_model_profile
from .run_identity import SemanticIdentity, build_semantic_identity, default_semantic_source_paths, sha256_file
from .scoring import ScoringTelemetry, single_token_label_ids
from .shards import ShardStore, run_sharded_phase
from .utils import read_table, read_yaml, write_table_atomic


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASES = ("preflight", "behavioral", "conflicts", "vanilla", "controls", "attention", "manifest")
EXPECTED_ACTIVE_WRAPPERS = {
    "csv_inline",
    "graphql_query",
    "html_form",
    "ini_file",
    "key_equals",
    "protobuf_msg",
    "shell_heredoc",
    "toml_config",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@dataclass
class StopState:
    requested: bool = False
    signal_name: str | None = None

    def request(self, signum: int, _frame) -> None:
        self.requested = True
        self.signal_name = signal.Signals(signum).name


class ForwardTimer:
    """Optional synchronized timing for representative profiling canaries."""

    def __init__(self, model, *, enabled: bool):
        self.model = model
        self.enabled = enabled
        self.seconds = 0.0
        self.calls = 0
        self._started = 0.0
        self._handles = []

    @staticmethod
    def _sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def __enter__(self):
        if not self.enabled:
            return self

        def before(_module, _inputs):
            self._sync()
            self._started = time.monotonic()

        def after(_module, _inputs, output):
            self._sync()
            self.seconds += time.monotonic() - self._started
            self.calls += 1
            return output

        self._handles = [self.model.register_forward_pre_hook(before), self.model.register_forward_hook(after)]
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        return False


def _identity_columns(frame: pd.DataFrame, identity: SemanticIdentity) -> pd.DataFrame:
    out = frame.copy()
    out["semantic_run_id"] = identity.semantic_run_id
    out["model_id"] = str(identity.payload["model"]["id"])
    out["model_revision"] = str(identity.payload["model"]["revision"])
    return out


def _artifact_manifest_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.manifest.json")


def _write_bound_table(frame: pd.DataFrame, path: Path, identity: SemanticIdentity) -> None:
    write_table_atomic(frame, path)
    _atomic_json(
        {
            "artifact_schema_version": 1,
            "semantic_run_id": identity.semantic_run_id,
            "semantic_sha256": identity.semantic_sha256,
            "model_id": identity.payload["model"]["id"],
            "model_revision": identity.payload["model"]["revision"],
            "row_count": int(len(frame)),
            "sha256": sha256_file(path),
        },
        _artifact_manifest_path(path),
    )


def _read_bound_table(path: Path, identity: SemanticIdentity, *, expected_rows: int | None = None) -> pd.DataFrame:
    manifest_path = _artifact_manifest_path(path)
    if not path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Missing identity-bound artifact or manifest: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "semantic_run_id": identity.semantic_run_id,
        "semantic_sha256": identity.semantic_sha256,
        "model_id": identity.payload["model"]["id"],
        "model_revision": identity.payload["model"]["revision"],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Artifact identity mismatch for {path}: {key}")
    if sha256_file(path) != manifest.get("sha256"):
        raise RuntimeError(f"Artifact checksum mismatch: {path}")
    frame = read_table(path)
    if int(manifest.get("row_count", -1)) != len(frame):
        raise RuntimeError(f"Artifact row-count mismatch: {path}")
    if expected_rows is not None and len(frame) != expected_rows:
        raise RuntimeError(f"Artifact {path} has {len(frame)} rows; expected {expected_rows}")
    for column, value in (
        ("semantic_run_id", identity.semantic_run_id),
        ("model_id", identity.payload["model"]["id"]),
        ("model_revision", identity.payload["model"]["revision"]),
    ):
        if column not in frame or set(frame[column].dropna().astype(str)) != {str(value)}:
            raise RuntimeError(f"Artifact row identity mismatch for {path}: {column}")
    return frame


def resolve_run_root(
    profile: ModelProfile,
    identity: SemanticIdentity,
    *,
    project_root: str | Path,
    canary: bool,
    canary_name: str,
) -> Path:
    semantic_root = profile.output_base(project_root) / identity.semantic_run_id
    if not canary:
        return semantic_root
    if not canary_name or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in canary_name
    ):
        raise ValueError("--canary-name must contain only letters, numbers, dashes, and underscores")
    return semantic_root / "canaries" / canary_name


def assert_full_dataset_contract(dataset: pd.DataFrame, configured_wrappers: Sequence[str]) -> None:
    configured = {str(wrapper) for wrapper in configured_wrappers}
    observed_counts = dataset["wrapper_name"].astype(str).value_counts().to_dict()
    if (
        configured != EXPECTED_ACTIVE_WRAPPERS
        or len(dataset) != 24000
        or dataset["item_id"].nunique() != 3000
        or set(observed_counts) != EXPECTED_ACTIVE_WRAPPERS
        or any(int(count) != 3000 for count in observed_counts.values())
    ):
        raise RuntimeError(
            "Full source dataset contract failed: expected 24,000 rows, 3,000 items, "
            "and exactly 3,000 rows for each of the eight active wrappers"
        )


def _release_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _peak_vram_bytes() -> int:
    return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0


def _reset_peak_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _phase_metrics(
    *,
    phase: str,
    wall_seconds: float,
    forward_seconds: float,
    forward_calls: int,
    output_write_seconds: float,
    completed_units: int,
    total_units: int,
    peak_vram_bytes: int,
    input_tokens: int | None = None,
    padding_ratio: float | None = None,
    input_preparation_seconds: float | None = None,
) -> dict[str, object]:
    observed_preparation = (
        max(0.0, wall_seconds - forward_seconds - output_write_seconds)
        if input_preparation_seconds is None
        else input_preparation_seconds
    )
    other_seconds = max(0.0, wall_seconds - forward_seconds - output_write_seconds - observed_preparation)
    return {
        "phase": phase,
        "recorded_at": _utc_now(),
        "wall_seconds": wall_seconds,
        "forward_seconds": forward_seconds,
        "forward_calls": forward_calls,
        "input_preparation_seconds": observed_preparation,
        "input_preparation_is_residual_estimate": input_preparation_seconds is None,
        "other_seconds": other_seconds,
        "output_write_seconds": output_write_seconds,
        "completed_work_units": completed_units,
        "total_work_units": total_units,
        "completed_work_units_per_second": completed_units / wall_seconds if wall_seconds else None,
        "input_tokens": input_tokens,
        "input_tokens_per_second": input_tokens / wall_seconds if input_tokens is not None and wall_seconds else None,
        "padding_ratio": padding_ratio,
        "peak_vram_bytes": peak_vram_bytes,
    }


def _write_progress(
    root: Path,
    *,
    phase: str,
    completed: int,
    total: int,
    status: str,
    stop_state: StopState,
    last_shard: str | None = None,
) -> None:
    _atomic_json(
        {
            "phase": phase,
            "status": status,
            "completed_work_units": completed,
            "total_work_units": total,
            "last_shard": last_shard,
            "stop_requested": stop_state.requested,
            "stop_signal": stop_state.signal_name,
            "updated_at": _utc_now(),
        },
        root / "progress.json",
    )


def _load_profile_model(profile: ModelProfile, *, backend: str, local_files_only: bool):
    return load_model_and_tokenizer(
        profile.model_id,
        revision=profile.revision,
        tokenizer_name=profile.tokenizer_id,
        tokenizer_revision=profile.tokenizer_revision,
        local_files_only=local_files_only,
        attn_implementation=backend,
        expected_layers=profile.expected_layers,
        trust_remote_code=profile.trust_remote_code,
    )


def _run_preflight(
    root: Path,
    identity: SemanticIdentity,
    profile: ModelProfile,
    dataset: pd.DataFrame,
    model,
    tokenizer,
    device,
    *,
    scoring_config: dict[str, object],
) -> None:
    observed_layers = len(find_transformer_blocks(model))
    label_ids = single_token_label_ids(tokenizer)
    anchors: dict[str, dict[str, list[int]]] = {}
    for wrapper, group in dataset.groupby("wrapper_name", sort=True):
        row = group.iloc[0]
        wrapper_result: dict[str, list[int]] = {}
        for anchor in ("options_end", "all_option_ends"):
            positions = resolve_anchor_positions(
                tokenizer,
                str(row["wrapped_prompt"]),
                anchor,
                question=row.get("question"),
                choices=row.get("choices"),
            )
            if positions is None:
                raise RuntimeError(f"Preflight could not resolve {anchor} for wrapper {wrapper}")
            wrapper_result[anchor] = [position.position for position in positions]
        anchors[str(wrapper)] = wrapper_result

    tiny = evaluate_behavioral(
        dataset.head(1),
        model,
        tokenizer,
        sequence_batch_size=1,
        checkpoint_size=1,
        max_batch_tokens=int(scoring_config.get("max_batch_tokens", 40000)),
        device=device,
    )
    if len(tiny) != 1:
        raise RuntimeError("Preflight tiny behavioral output did not contain exactly one row")
    _atomic_json(
        {
            "status": "passed",
            "recorded_at": _utc_now(),
            "semantic_run_id": identity.semantic_run_id,
            "model_id": profile.model_id,
            "model_revision": profile.revision,
            "tokenizer_id": profile.tokenizer_id,
            "tokenizer_revision": profile.tokenizer_revision,
            "device": str(device),
            "dtype": str(next(model.parameters()).dtype),
            "attention_backend": profile.attention_backend,
            "observed_transformer_layers": observed_layers,
            "expected_transformer_layers": profile.expected_layers,
            "label_token_ids": label_ids,
            "generic_label_fallback_required": label_ids is None,
            "anchor_positions": anchors,
            "tiny_output": tiny[["item_id", "wrapper_name", "raw_pred_label", "cal_pred_label"]].to_dict("records"),
        },
        root / "metadata" / "preflight.json",
    )


def _behavioral_work_key(row: pd.Series) -> str:
    return f"behavioral:{row['item_id']}:{row['wrapper_name']}"


def _run_behavioral(
    root: Path,
    identity: SemanticIdentity,
    source: pd.DataFrame,
    model,
    tokenizer,
    device,
    *,
    scoring_config: dict[str, object],
    stop_state: StopState,
    profile_timings: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    store = ShardStore(root / "shards" / "behavioral", identity)
    completed = store.completed_work_keys()
    checkpoint_size = int(scoring_config.get("checkpoint_size", 256))
    sequence_batch_size = int(scoring_config.get("sequence_batch_size", scoring_config.get("batch_size", 4)))
    max_batch_tokens = int(scoring_config.get("max_batch_tokens", 40000))
    started = time.monotonic()
    _reset_peak_vram()
    forward_seconds = 0.0
    forward_calls = 0
    scoring_telemetry = ScoringTelemetry()
    for start in range(0, len(source), checkpoint_size):
        chunk = source.iloc[start : start + checkpoint_size].copy()
        keys = [_behavioral_work_key(row) for _, row in chunk.iterrows()]
        pending_mask = [key not in completed for key in keys]
        if not any(pending_mask):
            continue
        if stop_state.requested:
            break
        pending = chunk.loc[pending_mask].reset_index(drop=True)
        pending_keys = [key for key, keep in zip(keys, pending_mask, strict=True) if keep]
        with ForwardTimer(model, enabled=profile_timings) as timer:
            scored = evaluate_behavioral(
                pending,
                model,
                tokenizer,
                sequence_batch_size=sequence_batch_size,
                checkpoint_size=len(pending),
                max_batch_tokens=max_batch_tokens,
                device=device,
                telemetry=scoring_telemetry,
            )
        forward_seconds += timer.seconds
        forward_calls += timer.calls
        scored = _identity_columns(scored, identity)
        scored["_work_key"] = pending_keys
        shard = store.write_shard(scored, work_keys=pending_keys)
        completed.update(pending_keys)
        _write_progress(
            root,
            phase="behavioral",
            completed=len(completed),
            total=len(source),
            status="stopping" if stop_state.requested else "running",
            stop_state=stop_state,
            last_shard=str(shard.relative_to(root)),
        )
        if stop_state.requested:
            break

    merged = store.merge(sort_by=["_work_key"])
    complete = len(completed) == len(source)
    if complete:
        _write_bound_table(merged, root / "raw" / "behavioral_scores.parquet", identity)
    wall = time.monotonic() - started
    input_tokens = scoring_telemetry.actual_tokens
    metrics = _phase_metrics(
        phase="behavioral",
        wall_seconds=wall,
        forward_seconds=forward_seconds,
        forward_calls=forward_calls,
        output_write_seconds=store.write_seconds,
        completed_units=len(completed),
        total_units=len(source),
        peak_vram_bytes=_peak_vram_bytes(),
        input_tokens=input_tokens,
        padding_ratio=scoring_telemetry.padding_ratio,
        input_preparation_seconds=scoring_telemetry.input_preparation_seconds,
    )
    _write_progress(
        root,
        phase="behavioral",
        completed=len(completed),
        total=len(source),
        status="complete" if complete else "stopped",
        stop_state=stop_state,
    )
    return merged, metrics


def _construct_conflicts(root: Path, identity: SemanticIdentity, behavioral: pd.DataFrame) -> pd.DataFrame:
    pairs = construct_conflict_pairs(
        behavioral,
        category="pure_interface",
        require_same_wrapper_calibration=True,
    )
    pairs = _identity_columns(pairs, identity)
    outcomes = _identity_columns(item_outcome_summary(behavioral), identity)
    _write_bound_table(pairs, root / "processed" / "conflict_pairs.parquet", identity)
    _write_bound_table(outcomes, root / "processed" / "item_outcomes.parquet", identity)
    return pairs


def _selected_pairs(pairs: pd.DataFrame, *, split: str, cap: int | None, seed: int) -> pd.DataFrame:
    selected = pairs[pairs["split"] == split].copy()
    if cap is not None and len(selected) > cap:
        selected = selected.sample(n=cap, random_state=seed)
    return selected.sort_values("item_id", kind="mergesort").reset_index(drop=True)


def _select_mechanistic_work(
    pairs: pd.DataFrame,
    *,
    phases: tuple[str, ...],
    canary: bool,
    split: str,
    diagnostic_cap: int,
    patching_cap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select conflict rows only when a requested phase needs them."""

    mechanistic_requested = bool(set(phases) & {"vanilla", "controls", "attention"})
    if not mechanistic_requested:
        empty = pairs.iloc[0:0].copy()
        return empty, empty.copy()
    if pairs.empty:
        raise RuntimeError(
            "mechanistic phases require a target-model conflict; rerun the canary with more --canary-items"
        )
    selected_split = str(pairs.iloc[0]["split"]) if canary else split
    diagnostic = _selected_pairs(pairs, split=selected_split, cap=diagnostic_cap, seed=seed)
    controls = _selected_pairs(pairs, split=selected_split, cap=patching_cap, seed=seed)
    return diagnostic, controls


def _run_incremental_mechanistic_phase(
    *,
    phase: str,
    root: Path,
    identity: SemanticIdentity,
    pairs: pd.DataFrame,
    selected: pd.DataFrame,
    process_pair: Callable[[str, pd.Series], pd.DataFrame],
    summary: Callable[[pd.DataFrame], pd.DataFrame],
    output_name: str,
    table_name: str,
    sort_by: Sequence[str],
    model,
    stop_state: StopState,
    max_shard_pairs: int,
    max_shard_seconds: float,
    profile_timings: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    store = ShardStore(root / "shards" / phase, identity)
    items = [(f"{phase}:{row['item_id']}", row) for _, row in selected.iterrows()]
    started = time.monotonic()
    _reset_peak_vram()

    def process(work_key: str, row: pd.Series) -> pd.DataFrame:
        return _identity_columns(process_pair(work_key, row), identity)

    def on_flush(completed: set[str], shard: Path) -> None:
        _write_progress(
            root,
            phase=phase,
            completed=len(completed),
            total=len(items),
            status="stopping" if stop_state.requested else "running",
            stop_state=stop_state,
            last_shard=str(shard.relative_to(root)),
        )

    with ForwardTimer(model, enabled=profile_timings) as timer:
        run_sharded_phase(
            store,
            items,
            process,
            max_work_units=max_shard_pairs,
            max_seconds=max_shard_seconds,
            should_stop=lambda: stop_state.requested,
            on_flush=on_flush,
        )
    merged = store.merge(sort_by=sort_by)
    completed = store.completed_work_keys()
    complete = len(completed) == len(items)
    if complete:
        _write_bound_table(merged, root / "processed" / output_name, identity)
        write_table_atomic(summary(merged), root / "tables" / table_name)
    wall = time.monotonic() - started
    metrics = _phase_metrics(
        phase=phase,
        wall_seconds=wall,
        forward_seconds=timer.seconds,
        forward_calls=timer.calls,
        output_write_seconds=store.write_seconds,
        completed_units=len(completed),
        total_units=len(items),
        peak_vram_bytes=_peak_vram_bytes(),
        padding_ratio=0.0,
    )
    _write_progress(
        root,
        phase=phase,
        completed=len(completed),
        total=len(items),
        status="complete" if complete else "stopped",
        stop_state=stop_state,
    )
    return merged, metrics


def _write_run_manifest(
    root: Path,
    identity: SemanticIdentity,
    profile: ModelProfile,
    *,
    config_path: str,
    dataset_rows: int,
    conflict_rows: int,
    timings: list[dict[str, object]],
    canary: bool,
    args,
    phase_totals: dict[str, int] | None = None,
) -> None:
    totals = dict(phase_totals or {})
    phase_completion: dict[str, dict[str, int | bool]] = {}
    for phase, total in totals.items():
        shard_root = root / "shards" / phase
        completed = len(ShardStore(shard_root, identity).completed_work_keys()) if shard_root.exists() else 0
        phase_completion[phase] = {
            "completed": completed,
            "total": int(total),
            "complete": completed == int(total),
        }
    required_outputs = [
        root / "raw" / "behavioral_scores.parquet",
        root / "processed" / "conflict_pairs.parquet",
        root / "processed" / "vanilla_convergence.parquet",
        root / "processed" / "focused_patching_controls.parquet",
        root / "processed" / "attention_diagnostics.parquet",
    ]
    full_complete = (
        not canary
        and dataset_rows == 24000
        and conflict_rows > 0
        and set(phase_completion) == {"behavioral", "vanilla", "controls", "attention"}
        and all(bool(entry["complete"]) for entry in phase_completion.values())
        and all(path.exists() and _artifact_manifest_path(path).exists() for path in required_outputs)
    )
    _atomic_json(
        {
            "manifest_schema_version": 2,
            "created_at": _utc_now(),
            "status": "complete" if full_complete else ("canary_complete" if canary else "partial"),
            "semantic_identity": identity.as_dict(),
            "model": {
                "profile": profile.name,
                "id": profile.model_id,
                "revision": profile.revision,
                "slug": profile.slug,
                "expected_layers": profile.expected_layers,
                "trust_remote_code": profile.trust_remote_code,
                "mechanistic_layers": list(profile.mechanistic_layers),
                "normal_attention_backend": profile.attention_backend,
                "diagnostic_attention_backend": profile.diagnostic_attention_backend,
            },
            "config_path": str(config_path),
            "dataset_rows": dataset_rows,
            "conflict_rows": conflict_rows,
            "canary": canary,
            "phase_completion": phase_completion,
            "timings": timings,
            "execution_provenance": {
                "argv": sys.argv,
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "sequence_batch_size": args.sequence_batch_size,
                "max_batch_tokens": args.max_batch_tokens,
                "behavioral_checkpoint_size": args.behavioral_checkpoint_size,
                "max_shard_pairs": args.max_shard_pairs,
                "max_shard_seconds": args.max_shard_seconds,
            },
        },
        root / "metadata" / "run_manifest.json",
    )
    _atomic_json(timings, root / "metadata" / "timings.json")


def run_model(args) -> None:
    config = read_yaml(args.config)
    profile = get_model_profile(args.profile)
    config = json.loads(json.dumps(config))
    scoring_config = config.setdefault("scoring", {})
    if args.sequence_batch_size is not None:
        scoring_config["sequence_batch_size"] = args.sequence_batch_size
    if args.max_batch_tokens is not None:
        scoring_config["max_batch_tokens"] = args.max_batch_tokens
    if args.behavioral_checkpoint_size is not None:
        scoring_config["checkpoint_size"] = args.behavioral_checkpoint_size
    identity = build_semantic_identity(
        profile,
        config=config,
        dataset_path=config["dataset_path"],
        source_paths=default_semantic_source_paths(PROJECT_ROOT),
    )
    root = resolve_run_root(
        profile,
        identity,
        project_root=PROJECT_ROOT,
        canary=args.canary,
        canary_name=args.canary_name,
    )
    semantic_root = profile.output_base(PROJECT_ROOT) / identity.semantic_run_id
    for directory in ("metadata", "raw", "processed", "tables", "shards"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    identity_path = semantic_root / "semantic_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity.as_dict():
        raise RuntimeError(f"Semantic run directory contains a conflicting identity: {semantic_root}")
    if not identity_path.exists():
        _atomic_json(identity.as_dict(), identity_path)

    dataset, wrapper_audit = prepare_dataset(config)
    assert_full_dataset_contract(dataset, config.get("experiment", {}).get("active_wrappers", []))
    if args.canary:
        item_ids = list(dict.fromkeys(dataset["item_id"].astype(str)))[: args.canary_items]
        dataset = dataset[dataset["item_id"].astype(str).isin(item_ids)].reset_index(drop=True)
    dataset_audit = dataset_audit_from_frame(
        dataset,
        dataset_path=config["dataset_path"],
        wrapper_policy=str(config.get("experiment", {}).get("wrapper_policy", "verified_pure_interface_only")),
    )
    _atomic_json(dataset_audit, root / "metadata" / "dataset_audit.json")
    write_table_atomic(wrapper_audit, root / "metadata" / "wrapper_audit.csv")
    save_split_manifest(dataset, root / "metadata" / "split_manifest.json")

    phases = tuple(part.strip() for part in args.phases.split(",") if part.strip())
    unknown = set(phases) - set(DEFAULT_PHASES)
    if unknown:
        raise ValueError(f"Unknown run-model phases: {sorted(unknown)}")
    stop_state = StopState()
    previous_handlers = {
        signum: signal.signal(signum, stop_state.request)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    timings: list[dict[str, object]] = []
    model = tokenizer = device = None
    behavioral = pd.DataFrame()
    pairs = pd.DataFrame()
    try:
        normal_phases = set(phases) & {"preflight", "behavioral", "vanilla", "controls"}
        if normal_phases:
            model, tokenizer, device = _load_profile_model(
                profile,
                backend=profile.attention_backend,
                local_files_only=args.local_files_only,
            )
        if "preflight" in phases:
            _run_preflight(
                root,
                identity,
                profile,
                dataset,
                model,
                tokenizer,
                device,
                scoring_config=scoring_config,
            )
        if "behavioral" in phases:
            behavioral, metrics = _run_behavioral(
                root,
                identity,
                dataset,
                model,
                tokenizer,
                device,
                scoring_config=scoring_config,
                stop_state=stop_state,
                profile_timings=args.profile_timings or args.canary,
            )
            timings.append(metrics)
            if stop_state.requested or len(behavioral) != len(dataset):
                _write_run_manifest(
                    root,
                    identity,
                    profile,
                    config_path=args.config,
                    dataset_rows=len(dataset),
                    conflict_rows=0,
                    timings=timings,
                    canary=args.canary,
                    args=args,
                    phase_totals={"behavioral": len(dataset)},
                )
                raise SystemExit(130)
        elif (root / "raw" / "behavioral_scores.parquet").exists():
            behavioral = _read_bound_table(
                root / "raw" / "behavioral_scores.parquet",
                identity,
                expected_rows=len(dataset),
            )

        if "conflicts" in phases:
            if behavioral.empty:
                raise RuntimeError("conflicts phase requires a complete behavioral table")
            pairs = _construct_conflicts(root, identity, behavioral)
        elif (root / "processed" / "conflict_pairs.parquet").exists():
            pairs = _read_bound_table(root / "processed" / "conflict_pairs.parquet", identity)

        seed = int(config.get("seed", 1729))
        focused = config.get("focused_mechanistic", {})
        anchors = tuple(str(anchor) for anchor in focused.get("anchors", ["options_end", "all_option_ends"]))
        split = str(pairs.iloc[0]["split"]) if args.canary and not pairs.empty else "validation"
        diagnostic_cap = 1 if args.canary else int(focused.get("diagnostic_cap", 300))
        patching_cap = 1 if args.canary else int(focused.get("patching_cap", 300))
        diagnostic_pairs, control_pairs = _select_mechanistic_work(
            pairs,
            phases=phases,
            canary=args.canary,
            split=split,
            diagnostic_cap=diagnostic_cap,
            patching_cap=patching_cap,
            seed=seed,
        )

        if "vanilla" in phases:
            _, metrics = _run_incremental_mechanistic_phase(
                phase="vanilla",
                root=root,
                identity=identity,
                pairs=pairs,
                selected=diagnostic_pairs,
                process_pair=lambda _key, row: run_vanilla_convergence(
                    model,
                    tokenizer,
                    pd.DataFrame([row.to_dict()]),
                    layers=list(profile.mechanistic_layers),
                    split=split,
                    cap=None,
                    anchors=anchors,
                    seed=seed,
                    device=device,
                ),
                summary=summarize_vanilla_convergence,
                output_name="vanilla_convergence.parquet",
                table_name="table_vanilla_convergence.csv",
                sort_by=["_work_key", "anchor", "layer"],
                model=model,
                stop_state=stop_state,
                max_shard_pairs=args.max_shard_pairs,
                max_shard_seconds=args.max_shard_seconds,
                profile_timings=args.profile_timings or args.canary,
            )
            timings.append(metrics)
        if stop_state.requested:
            raise SystemExit(130)

        if "controls" in phases:
            _, metrics = _run_incremental_mechanistic_phase(
                phase="controls",
                root=root,
                identity=identity,
                pairs=pairs,
                selected=control_pairs,
                process_pair=lambda _key, row: run_focused_patching_controls(
                    model,
                    tokenizer,
                    pairs,
                    layers=list(profile.mechanistic_layers),
                    split=split,
                    cap=patching_cap,
                    anchors=anchors,
                    seed=seed,
                    device=device,
                    target_item_ids={str(row["item_id"])},
                ),
                summary=summarize_focused_controls,
                output_name="focused_patching_controls.parquet",
                table_name="table_focused_controls.csv",
                sort_by=["_work_key", "condition", "anchor", "layer"],
                model=model,
                stop_state=stop_state,
                max_shard_pairs=args.max_shard_pairs,
                max_shard_seconds=args.max_shard_seconds,
                profile_timings=args.profile_timings or args.canary,
            )
            timings.append(metrics)
        if stop_state.requested:
            raise SystemExit(130)

        if model is not None:
            _release_model(model)
            model = tokenizer = device = None

        if "attention" in phases:
            model, tokenizer, device = _load_profile_model(
                profile,
                backend=profile.diagnostic_attention_backend,
                local_files_only=args.local_files_only,
            )
            _, metrics = _run_incremental_mechanistic_phase(
                phase="attention",
                root=root,
                identity=identity,
                pairs=pairs,
                selected=diagnostic_pairs,
                process_pair=lambda _key, row: run_attention_diagnostics(
                    model,
                    tokenizer,
                    pd.DataFrame([row.to_dict()]),
                    split=split,
                    cap=None,
                    anchors=anchors,
                    seed=seed,
                    device=device,
                ),
                summary=summarize_attention_diagnostics,
                output_name="attention_diagnostics.parquet",
                table_name="table_attention_diagnostics.csv",
                sort_by=["_work_key", "run_kind", "anchor", "layer", "head"],
                model=model,
                stop_state=stop_state,
                max_shard_pairs=args.max_shard_pairs,
                max_shard_seconds=args.max_shard_seconds,
                profile_timings=args.profile_timings or args.canary,
            )
            timings.append(metrics)
        if stop_state.requested:
            raise SystemExit(130)

        if "manifest" in phases:
            _write_run_manifest(
                root,
                identity,
                profile,
                config_path=args.config,
                dataset_rows=len(dataset),
                conflict_rows=len(pairs),
                timings=timings,
                canary=args.canary,
                args=args,
                phase_totals={
                    "behavioral": len(dataset),
                    "vanilla": len(diagnostic_pairs),
                    "controls": len(control_pairs),
                    "attention": len(diagnostic_pairs),
                },
            )
        print(json.dumps({"status": "complete", "model": profile.model_id, "run_root": str(root)}, sort_keys=True))
    finally:
        if model is not None:
            _release_model(model)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
