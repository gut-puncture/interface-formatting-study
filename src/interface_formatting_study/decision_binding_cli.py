from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .decision_binding import (
    capture_layer_readouts,
    evaluate_probe_bank,
    fit_probe_bank,
    load_probe_bank,
    prepare_patch_pair_ledger,
    prepare_readout_ledger,
    run_patch_pair,
    save_probe_bank,
    select_readout_layers,
)
from .model_loader import load_model_and_tokenizer
from .model_profiles import MODEL_PROFILES, get_model_profile
from .model_runner import StopState
from .run_identity import build_semantic_identity, default_semantic_source_paths, sha256_file
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
    readout_path = root / "readout_ledger.parquet"
    pairs_path = root / "patch_pair_ledger.parquet"
    for key, path in (("readout", readout_path), ("pairs", pairs_path)):
        if sha256_file(path) != manifest[key]["sha256"]:
            raise RuntimeError(f"Decision-binding {key} checksum mismatch")
    readout = read_table(readout_path)
    pairs = read_table(pairs_path)
    if len(readout) != manifest["readout"]["rows"] or len(pairs) != manifest["pairs"]["rows"]:
        raise RuntimeError("Decision-binding bundle row count mismatch")
    return manifest, readout, pairs


def cmd_prepare(args) -> None:
    scored_path = Path(args.scored)
    applicability_path = Path(args.applicability)
    scored = read_table(scored_path)
    applicability = read_table(applicability_path)
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
            "bundle_schema_version": 1,
            "stage": args.stage,
            "seed": args.seed,
            "source_scored_sha256": sha256_file(scored_path),
            "source_applicability_sha256": sha256_file(applicability_path),
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


def _chunked(frame: pd.DataFrame, size: int):
    for start in range(0, len(frame), size):
        chunk = frame.iloc[start : start + size].copy()
        digest = hashlib.sha256("\n".join(chunk["readout_work_key"].astype(str)).encode()).hexdigest()
        yield f"readout|{digest[:20]}", chunk


def _stable_sort(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    available = [column for column in columns if column in frame.columns]
    return frame.sort_values(available, kind="mergesort").reset_index(drop=True) if available else frame


def _load_frozen(run_root: Path, profile_name: str):
    identity = json.loads((run_root / "semantic_identity.json").read_text(encoding="utf-8"))
    profile = get_model_profile(profile_name)
    if identity["model"]["id"] != profile.model_id or identity["model"]["revision"] != profile.revision:
        raise RuntimeError("Frozen mechanism belongs to a different model identity")
    selection_path = run_root / "frozen_selection.json"
    bank_path = run_root / "probe_bank.npz"
    selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection_payload["probe_bank_sha256"] != sha256_file(bank_path):
        raise RuntimeError("Frozen probe bank checksum mismatch")
    return load_probe_bank(bank_path), selection_payload["selection"], selection_path, bank_path


def _fit_or_load_bank(root: Path, ledger: pd.DataFrame, model, tokenizer, args):
    bank_path = root / "probe_bank.npz"
    meta_path = root / "probe_bank.json"
    if bank_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["sha256"] != sha256_file(bank_path):
            raise RuntimeError("Persisted probe bank checksum mismatch")
        return load_probe_bank(bank_path)
    train = ledger[(ledger["readout_role"] == "probe_train") & ledger["winner_unique"].astype(bool)]
    if train.empty:
        raise RuntimeError("Discovery bundle has no unique-winner probe-training rows")
    captured = capture_layer_readouts(
        model,
        tokenizer,
        train["prompt"].astype(str).tolist(),
        batch_size=args.batch_size,
        max_batch_tokens=args.max_batch_tokens,
    )
    bank = fit_probe_bank(captured.activations, train["winner_content_id"].astype(int).tolist())
    save_probe_bank(bank_path, bank)
    _atomic_json({"sha256": sha256_file(bank_path), "training_rows": len(train)}, meta_path)
    return bank


def cmd_run_model(args) -> None:
    bundle_root = Path(args.bundle)
    bundle_manifest, ledger, pairs = _load_bundle(bundle_root)
    profile = get_model_profile(args.profile)
    if bundle_manifest["model"]["id"] not in {None, profile.model_id}:
        raise RuntimeError("Bundle model identity does not match --profile")
    if bundle_manifest["model"]["revision"] not in {None, profile.revision}:
        raise RuntimeError("Bundle model revision does not match --profile")
    stage = str(bundle_manifest["stage"])
    frozen_hashes = None
    if stage == "confirmation":
        if not args.frozen_run:
            raise ValueError("Confirmation requires --frozen-run")
        bank, selection, selection_path, bank_path = _load_frozen(Path(args.frozen_run), args.profile)
        frozen_hashes = {
            "selection_sha256": sha256_file(selection_path),
            "probe_bank_sha256": sha256_file(bank_path),
        }
    config = {
        "study": "decision_binding_v1",
        "stage": stage,
        "bundle_manifest_sha256": sha256_file(bundle_root / "bundle_manifest.json"),
        "readout_sha256": bundle_manifest["readout"]["sha256"],
        "pairs_sha256": bundle_manifest["pairs"]["sha256"],
        "probe_c": 1e-2,
        "bootstrap_samples": args.bootstrap_samples,
        "permutation_samples": args.permutation_samples,
        "stable_controls": args.stable_controls,
        "label_binding_controls": args.label_binding_controls,
        "frozen": frozen_hashes,
        "limits": {"readout_rows": args.max_readout_rows, "pairs": args.max_pairs},
    }
    identity = build_semantic_identity(
        profile,
        config=config,
        dataset_path=bundle_root / "readout_ledger.parquet",
        source_paths=default_semantic_source_paths(PROJECT_ROOT),
    )
    root = Path(args.output_base) / profile.slug / identity.semantic_run_id
    root.mkdir(parents=True, exist_ok=True)
    identity_path = root / "semantic_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity.as_dict():
        raise RuntimeError("Run-root semantic identity mismatch")
    _atomic_json(identity.as_dict(), identity_path)
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
                "failure": {"type": type(error).__name__, "message": str(error)},
            },
            root / "run_manifest.json",
        )
        raise
    if not args.allow_cpu and (not torch.cuda.is_available() or device.type != "cuda"):
        raise RuntimeError("Decision-binding model runs require CUDA")
    if stage == "discovery":
        bank = _fit_or_load_bank(root, ledger, model, tokenizer, args)
    stop = StopState()
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in previous:
        signal.signal(sig, stop.request)
    try:
        def progress(phase: str, completed: int, total: int, shard: Path | None = None) -> None:
            _atomic_json(
                {
                    "phase": phase,
                    "completed": completed,
                    "total": total,
                    "last_shard": None if shard is None else shard.name,
                },
                root / "progress.json",
            )

        evaluation = ledger[ledger["readout_role"].isin(["layer_select", "confirmation"])].copy()
        if args.max_readout_rows:
            evaluation = evaluation.iloc[: args.max_readout_rows]
        readout_store = ShardStore(root / "shards" / "readout", identity)
        readout_work = list(_chunked(evaluation, args.readout_chunk_size))
        readout_scores = run_sharded_phase(
            readout_store,
            readout_work,
            lambda _key, chunk: evaluate_probe_bank(
                bank,
                capture_layer_readouts(
                    model,
                    tokenizer,
                    chunk["prompt"].astype(str).tolist(),
                    batch_size=args.batch_size,
                    max_batch_tokens=args.max_batch_tokens,
                ).activations,
                chunk,
            ),
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
            _atomic_json({"status": "interrupted", "stage": "readout"}, root / "run_manifest.json")
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
                "probe_bank_sha256": sha256_file(root / "probe_bank.npz"),
                "readout_scores_sha256": sha256_file(root / "readout_scores.parquet"),
            }
            _atomic_json(selection_payload, root / "frozen_selection.json")
        selected = pairs[pairs["selected_for_patching"].astype(bool)].copy()
        if args.max_pairs:
            selected = selected.iloc[: args.max_pairs]
        patch_store = ShardStore(root / "shards" / "patches", identity)
        patch_work = [(str(row["pair_work_key"]), row) for _, row in selected.iterrows()]
        patch_results = run_sharded_phase(
            patch_store,
            patch_work,
            lambda _key, row: run_patch_pair(model, tokenizer, row, bank, selection, device=device),
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
        _atomic_json(
            {
                "status": "complete" if complete else "interrupted",
                "stage": stage,
                "profile": args.profile,
                "semantic_identity": identity.as_dict(),
                "readout_rows": len(readout_scores),
                "readout_chunks": len(readout_work),
                "selected_pairs": len(selected),
                "completed_pairs": len(patch_store.completed_work_keys()),
                "artifacts": {
                    "readout_scores_sha256": sha256_file(root / "readout_scores.parquet"),
                    "patch_results_sha256": sha256_file(root / "patch_results.parquet"),
                    "frozen_selection_sha256": sha256_file(root / "frozen_selection.json")
                    if (root / "frozen_selection.json").exists()
                    else frozen_hashes["selection_sha256"],
                    "probe_bank_sha256": sha256_file(root / "probe_bank.npz")
                    if (root / "probe_bank.npz").exists()
                    else frozen_hashes["probe_bank_sha256"],
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


def cmd_analyze(args) -> None:
    frames = []
    for value in args.run:
        root = Path(value)
        frame = read_table(root / "patch_results.parquet")
        frame["model"] = root.parent.name
        frames.append(frame)
    patches = pd.concat(frames, ignore_index=True)
    keys = ["model", "item_id", "pair_work_key", "mechanism", "layer", "pair_kind"]
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
    group_keys = ["model", "mechanism", "layer", "pair_kind", "condition"]
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
    _atomic_json(
        {"runs": [str(value) for value in args.run], "rows": len(patches)},
        output / "analysis_manifest.json",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="interface-formatting-decision-binding")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--scored", required=True)
    prepare.add_argument("--applicability", required=True)
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
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--local-files-only", action="store_true")
    run.add_argument("--allow-cpu", action="store_true", help=argparse.SUPPRESS)
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
